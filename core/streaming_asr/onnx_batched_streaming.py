"""
进程内多路批处理流式 ASR 引擎。

本引擎走 **官方 ONNX 导出物**:
  - model.onnx   : encoder + CIF-CNN (输入 speech (B,T,560), 输出 enc/enc_len/alphas,
                   动态 batch 轴, 无内部 cache —— look-back 由调用方拼 feats 重叠窗)
  - decoder.onnx : decoder (输入 enc/acoustic_embeds + 12 层 FSMN cache (B,320,10),
                   输出 logits/sample_ids/新 cache)

调用语义 (feats 重叠窗、scale、位置编码、CIF 积分、is_final 两段切分、last_chunk
padding、FSMN cache 裁剪) 严格镜像官方 Python 参考实现 funasr_onnx.Paraformer
(funasr-onnx 包, 与官方 C++ runtime 同构)。该参考实现本身是单流 batch=1 的;
本引擎把"每流准备 (frontend/PE/overlap, 无共享状态)"留在调用线程, 把
"encoder/decoder ORT 前向 + CIF 积分 + cache 合并/拆分"放到单一 batcher 线程,
多路同形请求沿 batch 维合并 —— ONNX 图的动态 batch 轴使其天然支持。

注意: int8 量化 (onnx_quant) 与 torch fp32 路径存在数值差异, 一致性校验
应使用 fp32 图严格比对, 量化图按字错率容差评估。

首遍热词在流式 Paraformer 路径本身不生效, 本引擎同样忽略 hotwords。
"""

import json
import os
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from core.engine_health import SelfHealingMixin
from core.streaming_asr.base import BaseStreamingASREngine
from core.metrics.prometheus_metrics import metrics
from config.settings import StreamingASRConfig


class _ForwardRequest:
    """一个流的一次前向请求 (调用线程构造, batcher 线程消费)。"""

    __slots__ = (
        "feats",
        "feats_len",
        "cache",
        "last_chunk",
        "tokens",
        "enqueued_at",
        "forward_started_at",
    )

    def __init__(self, feats: np.ndarray, feats_len: np.ndarray, cache: Dict[str, Any],
                 last_chunk: bool):
        self.feats = feats
        self.feats_len = feats_len
        self.cache = cache
        self.last_chunk = last_chunk
        self.tokens: List[str] = []
        self.enqueued_at = 0.0
        self.forward_started_at = 0.0


class _ForwardBatcher:
    """单线程前向合批调度器。"""

    def __init__(self, engine: "OnnxBatchedStreamingEngine"):
        self._engine = engine
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="asr-onnx-forward"
        )
        self._thread.start()

    def submit(self, req: _ForwardRequest) -> List[str]:
        req.enqueued_at = time.perf_counter()
        fut: Future = Future()
        self._queue.put((req, fut))
        try:
            tokens = fut.result(timeout=self._engine._submit_timeout_sec)
        except FutureTimeout:
            logger.warning("[OnnxForwardBatcher] submit timed out, returning empty tokens")
            return []
        except Exception as e:
            logger.error(f"[OnnxForwardBatcher] forward failed: {e}")
            return []
        wait = (
            req.forward_started_at - req.enqueued_at
            if req.forward_started_at > 0
            else 0.0
        )
        self._engine._record_wait(wait)
        return tokens

    def stop(self):
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _loop(self):
        max_batch = self._engine._max_batch
        window_sec = self._engine._batch_window_sec
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            batch = [item]
            deadline = time.perf_counter() + window_sec
            while len(batch) < max_batch:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop.set()
                    break
                batch.append(nxt)
            try:
                self._execute(batch)
            except Exception:
                # 保险丝: 线程死亡 = 全部会话的流式前向退化为 submit 30s
                # 超时静默返回空, 且无自愈路径。任何漏网异常都只废弃本批。
                logger.exception("[OnnxForwardBatcher] batch dispatch crashed; failing batch")
                for _req, fut in batch:
                    if not fut.done():
                        fut.set_result([])

    def _execute(self, batch):
        if self._engine._model is None:
            for req, fut in batch:
                fut.set_result([])
            return

        # 分组: last_chunk (尾块) 与稳态分开; 同 (T, cache 形状签名) 才合批
        # cache 可能被其他线程并发覆盖重置, 签名读取存在瞬时失败可能 ——
        # 只废弃本批 (调用方拿空 tokens), 绝不让异常逃出杀死 batcher 线程
        try:
            groups: Dict[Any, List] = {}
            for item in batch:
                req, fut = item
                key = (int(req.feats.shape[1]), req.last_chunk,
                       self._engine._cache_signature(req.cache))
                groups.setdefault(key, []).append(item)
        except Exception as e:
            logger.error(f"[OnnxForwardBatcher] batch grouping failed: {e}")
            for _req, fut in batch:
                if not fut.done():
                    fut.set_result([])
            return

        reload_reason = None
        for group in groups.values():
            reqs = [r for r, _ in group]
            try:
                t0 = time.perf_counter()
                for req, _fut in group:
                    req.forward_started_at = t0
                self._engine._forward_group(reqs)
                metrics.observe("asr_streaming_batch_size", float(len(group)))
            except Exception as e:
                if len(reqs) > 1:
                    logger.warning(
                        f"[OnnxForwardBatcher] merged forward failed ({e}); "
                        f"retrying {len(reqs)} requests solo"
                    )
                    for req, fut in group:
                        try:
                            self._engine._forward_group([req])
                            if not fut.done():
                                fut.set_result(req.tokens)
                        except Exception as e2:
                            logger.error(f"[OnnxForwardBatcher] solo forward failed: {e2}")
                            if not fut.done():
                                fut.set_result([])
                            reload_reason = self._engine._note_error(e2)
                    metrics.observe("asr_streaming_batch_size", 1.0)
                else:
                    logger.error(f"[OnnxForwardBatcher] forward error: {e}")
                    reload_reason = self._engine._note_error(e)
                    for req, fut in group:
                        if not fut.done():
                            fut.set_result([])
            else:
                metrics.observe(
                    "asr_streaming_forward_ms", (time.perf_counter() - t0) * 1000.0
                )
                self._engine._note_success()
                for req, fut in group:
                    if not fut.done():
                        fut.set_result(req.tokens)

        if reload_reason is not None:
            self._engine._start_reload(reload_reason)


def _ort_providers_for_device(
    device: Optional[str] = "cpu",
    available_providers: Optional[Sequence[str]] = None,
) -> List[Any]:
    """根据目标设备及当前环境支持的 provider 列表计算 ONNX Runtime providers 配置。

    支持设备格式:
      - 'cpu' -> ['CPUExecutionProvider']
      - 'cuda' / 'cuda:N' -> [('CUDAExecutionProvider', {'device_id': N}), 'CPUExecutionProvider'] (若可用)
      - 'auto' / None / '' -> 自动选择: 若 CUDAExecutionProvider 可用则走 cuda:0，否则走 cpu
    若请求 CUDA 但 CUDAExecutionProvider 不可用，记录 warning 并回退至 CPUExecutionProvider。
    """
    dev_raw = (device or "").strip()
    dev = dev_raw.lower()

    if available_providers is None:
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
        except Exception:
            available_providers = ["CPUExecutionProvider"]

    available_list = list(available_providers)
    has_cuda = "CUDAExecutionProvider" in available_list

    if not dev or dev == "auto":
        resolved = "cuda:0" if has_cuda else "cpu"
        logger.info(
            f"streaming_asr.device is '{device}', auto-resolved to '{resolved}' "
            f"(CUDAExecutionProvider available: {has_cuda})"
        )
        dev = resolved

    if dev == "cpu":
        return ["CPUExecutionProvider"]

    if dev == "cuda" or dev.startswith("cuda:"):
        device_id = 0
        if ":" in dev:
            part = dev.split(":", 1)[1].strip()
            try:
                device_id = int(part)
                if device_id < 0:
                    raise ValueError("negative device_id")
            except ValueError:
                logger.warning(
                    f"Invalid CUDA device ID in streaming_asr.device='{device}', "
                    f"defaulting to device_id=0"
                )
                device_id = 0

        if has_cuda:
            return [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            logger.warning(
                f"streaming_asr.device='{device}' requested CUDA, but 'CUDAExecutionProvider' "
                f"is not available in onnxruntime (available: {available_list}). "
                f"Falling back to CPUExecutionProvider."
            )
            return ["CPUExecutionProvider"]

    logger.warning(
        f"Unsupported streaming_asr.device='{device}', falling back to CPUExecutionProvider"
    )
    return ["CPUExecutionProvider"]


class OnnxBatchedStreamingEngine(SelfHealingMixin, BaseStreamingASREngine):
    """
    ONNX Runtime 多路合批流式 Paraformer 引擎 (唯一的首遍真实引擎)。

    模型目录需包含 funasr 导出物: model[_quant].onnx / decoder[_quant].onnx +
    config.yaml / am.mvn / tokens.json (scripts/export_onnx.py 产出)。
    """

    SUBMIT_TIMEOUT_SEC = 30.0
    _ort_providers_for_device = staticmethod(_ort_providers_for_device)

    def __init__(self, config: StreamingASRConfig, reload_after_errors: int = 3):
        self.config = config
        self.name = "onnx"
        self._lock = threading.Lock()  # SelfHealingMixin 契约
        self._model = None
        self._init_health(reload_after_errors)

        self._max_batch = max(1, int(getattr(config, "batch_size", 8) or 8))
        self._batch_window_sec = max(
            0.0, getattr(config, "batch_window_ms", 15) or 0
        ) / 1000.0
        self._submit_timeout_sec = self.SUBMIT_TIMEOUT_SEC
        self._tl = threading.local()
        # 合批排队 EWMA (ms), 准入控制探针读快照
        self._wait_ewma_ms = 0.0

        self._init_model()
        self._batcher = _ForwardBatcher(self)

    @property
    def _log_tag(self) -> str:
        return "OnnxBatchedStreamingEngine"

    @property
    def last_wait_sec(self) -> float:
        return getattr(self._tl, "last_wait_sec", 0.0)

    @property
    def wait_ewma_ms(self) -> float:
        with self._state_lock:
            return self._wait_ewma_ms

    def _record_wait(self, wait_sec: float):
        self._tl.last_wait_sec = wait_sec
        wait_ms = wait_sec * 1000.0
        with self._state_lock:
            self._wait_ewma_ms = 0.9 * self._wait_ewma_ms + 0.1 * wait_ms
        metrics.observe("asr_streaming_batch_wait_ms", wait_ms)

    # ------------------------------------------------------------------
    # 模型加载 (funasr_onnx 参考实现同款产物)
    # ------------------------------------------------------------------
    def _init_model(self):
        try:
            # 路径校验必须在 import onnxruntime / funasr_onnx 之前: 单元测试与
            # 缺依赖环境只应看到 ValueError/FileNotFoundError, 而不是
            # ModuleNotFoundError 抢先冒泡。
            model_dir = self.config.onnx_model_dir
            if not model_dir or not os.path.isdir(model_dir):
                raise ValueError(
                    f"streaming_asr.onnx_model_dir ('{model_dir}') 不存在; "
                    "先运行 scripts/export_onnx.py 导出模型"
                )
            suffix = "_quant" if self.config.onnx_quantize else ""
            enc_path = os.path.join(model_dir, f"model{suffix}.onnx")
            dec_path = os.path.join(model_dir, f"decoder{suffix}.onnx")
            for p in (enc_path, dec_path):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"ONNX 模型缺失: {p}")

            import yaml
            import onnxruntime as ort
            from funasr_onnx.utils.frontend import WavFrontendOnline
            from funasr_onnx.utils.frontend import SinusoidalPositionEncoderOnline

            with open(os.path.join(model_dir, "config.yaml"), "r", encoding="utf-8") as f:
                mcfg = yaml.safe_load(f)
            with open(os.path.join(model_dir, "tokens.json"), "r", encoding="utf-8") as f:
                token_list = json.load(f)
            self._tokens = token_list

            opts = ort.SessionOptions()
            threads = int(getattr(self.config, "onnx_intra_op_threads", 0) or 0)
            if threads > 0:
                opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            providers = _ort_providers_for_device(
                self.config.device, ort.get_available_providers()
            )

            def _create_sessions(provs):
                enc = ort.InferenceSession(enc_path, opts, providers=provs)
                dec = ort.InferenceSession(dec_path, opts, providers=provs)
                return enc, dec

            try:
                self._enc_sess, self._dec_sess = _create_sessions(providers)
            except Exception as e:
                if providers != ["CPUExecutionProvider"]:
                    logger.warning(
                        f"[{self._log_tag}] Failed to initialize ONNX InferenceSession with "
                        f"providers {providers}: {e}. Falling back to CPUExecutionProvider."
                    )
                    providers = ["CPUExecutionProvider"]
                    self._enc_sess, self._dec_sess = _create_sessions(providers)
                else:
                    raise

            self._enc_input_names = [i.name for i in self._enc_sess.get_inputs()]
            self._dec_input_names = [i.name for i in self._dec_sess.get_inputs()]

            fc = mcfg["frontend_conf"]
            self._frontend_conf = fc
            self._cmvn_file = os.path.join(model_dir, "am.mvn")
            self._feats_dims = int(fc["n_mels"]) * int(fc["lfr_m"])
            self._enc_output_size = int(mcfg["encoder_conf"]["output_size"])
            self._scale = float(self._enc_output_size) ** 0.5

            dc = mcfg["decoder_conf"]
            self._fsmn_layers = int(dc["num_blocks"])
            self._fsmn_lorder = int(dc["kernel_size"]) - 1
            self._fsmn_dims = self._enc_output_size

            pc = mcfg["predictor_conf"]
            self._cif_threshold = float(pc["threshold"])
            self._tail_threshold = float(pc["tail_threshold"])

            # 共享的无状态组件 (frontend 实例每个流分别创建, 因其有内部状态)
            self._pe = SinusoidalPositionEncoderOnline()
            self._frontend_cls = WavFrontendOnline

            self._chunk_size = list(self.config.chunk_size or [5, 10, 5])
            self._chunk_stride_samples = int(self._chunk_size[1] * 960)
            self._keep = self._chunk_size[0] + self._chunk_size[2]
            self._model = True  # 加载完成标记 (SelfHealingMixin is_available)
            enc_provs = self._enc_sess.get_providers()
            dec_provs = self._dec_sess.get_providers()
            provs_str = (
                f"{enc_provs}"
                if enc_provs == dec_provs
                else f"enc={enc_provs}, dec={dec_provs}"
            )
            logger.info(
                f"[{self._log_tag}] ONNX models loaded from {model_dir} "
                f"(device='{self.config.device}', providers={provs_str}, "
                f"quant={self.config.onnx_quantize}, B={self._max_batch}, "
                f"W={self._batch_window_sec * 1000:.0f}ms)"
            )
        except Exception as e:
            logger.error(f"[{self._log_tag}] Failed to load ONNX models: {e}")
            raise

    # ------------------------------------------------------------------
    # 每流状态
    # ------------------------------------------------------------------
    def _init_stream_cache(self, cache: Dict[str, Any]):
        """重建每流初始状态。

        不做 cache.clear(): 该 dict 可能仍被在途读者持有 (batcher 队列中的
        旧请求、断连后被取消的孤儿 chunk), 原地掏空会使其 KeyError 崩溃
        (线上实测 'cif_hidden' / 空 dict 上的 'start_idx')。改为全键覆盖写
        —— 本函数写入的键集是引擎全部读取键的超集, 覆盖后无残留旧键;
        覆盖过程中读者最坏读到瞬时新旧混合值, 由 batcher 的合批容错兜住,
        不会缺键。调用方传空 dict (新流) 或句末旧 dict (重置) 均可。
        """
        # dither=0: 推理服务要确定性输出。两套官方前端默认 dither=1.0 (波形加
        # 随机抖动), 特征扰动会让 CIF token 边界随机翻转 (实测同音频两次运行
        # 字符级输出不同); 官方 C++ runtime 同样默认关闭 dither。
        cache["frontend"] = self._frontend_cls(
            cmvn_file=self._cmvn_file, dither=0.0, **self._frontend_conf
        )
        cache["start_idx"] = 0
        cache["feats"] = np.zeros((1, self._keep, self._feats_dims), dtype=np.float32)
        cache["cif_hidden"] = np.zeros((1, 1, self._enc_output_size), dtype=np.float32)
        cache["cif_alphas"] = np.zeros((1, 1), dtype=np.float32)
        cache["decoder_fsmn"] = [
            np.zeros((1, self._fsmn_dims, self._fsmn_lorder), dtype=np.float32)
            for _ in range(self._fsmn_layers)
        ]
        cache["last_chunk"] = False
        # is_final 仅在 _prepare_requests 写入、_add_overlap_chunk 读取;
        # 取消 clear() 后必须在此显式复位, 否则上一句的 True 会泄漏到下一句
        cache["is_final"] = False
        cache["_prev_samples"] = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _cache_signature(cache: Dict[str, Any]) -> tuple:
        """cache 形状签名: 只有签名一致才能合批 (防形状演化导致的 cat 崩溃)。"""
        return (
            tuple(cache.get("feats").shape),
            tuple(cache.get("cif_hidden").shape),
            tuple(t.shape for t in cache.get("decoder_fsmn")),
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def process_chunk(
        self,
        audio_bytes: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        if not audio_bytes and not is_final:
            return ""
        if self._model is None:
            return ""

        if not cache:
            self._init_stream_cache(cache)

        # 空 flush 防御: 流内从未产出特征 (start_idx==0) 时, is_final 路径会在
        # funasr 前端 np.stack(空 lfr 缓存) 上崩溃 (frontend.py, 强制切句清缓存
        # 后紧跟 COMMIT 即触发)。缓存此刻本就是句末重置后的原始态, 直接收尾。
        if is_final and cache["start_idx"] == 0:
            return ""

        if audio_bytes:
            if len(audio_bytes) & 1:
                # 半个采样点: 截齐防 frombuffer ValueError (网关已挡一道, 此处兜底)
                audio_bytes = audio_bytes[:-1]
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.zeros(0, dtype=np.float32)
        audio = np.concatenate((cache["_prev_samples"], samples))

        stride = self._chunk_stride_samples
        n = int(len(audio) // stride) + int(is_final)
        m = int(len(audio) % stride) * (1 - int(is_final))

        tokens: List[str] = []
        for i in range(n):
            final_i = bool(is_final and i == n - 1)
            chunk_i = audio[i * stride : (i + 1) * stride]
            reqs = self._prepare_requests(cache, chunk_i, final_i)
            for req in reqs:
                tokens.extend(self._batcher.submit(req))

        cache["_prev_samples"] = audio[len(audio) - m :] if m > 0 else np.zeros(0, dtype=np.float32)
        if is_final:
            self._init_stream_cache(cache)  # 句末重置 (funasr 语义)

        if not tokens:
            return ""
        from funasr_onnx.utils.postprocess_utils import sentence_postprocess
        post = sentence_postprocess(tokens)
        # funasr_onnx 返回 (text, timestamp) 元组
        return post[0] if isinstance(post, tuple) else post

    def flush(self, cache: Dict[str, Any], hotwords: Optional[List[str]] = None) -> str:
        return self.process_chunk(b"", cache=cache, is_final=True)

    # ------------------------------------------------------------------
    # 每流请求准备 (调用线程, 镜像 funasr_onnx Paraformer.__call__ 前半段)
    # ------------------------------------------------------------------
    def _prepare_requests(self, cache, chunk_i: np.ndarray, is_final: bool):
        # 尾块不足一帧: 直接用缓存的 overlap feats (参考实现 <16*60 样本路径)
        if is_final and len(chunk_i) < 960 and cache["start_idx"] > 0:
            cache["last_chunk"] = True
            feats = cache["feats"]
            feats_len = np.array([feats.shape[1]]).astype(np.int32)
            return [_ForwardRequest(feats, feats_len, cache, last_chunk=True)]

        waveforms = chunk_i[None, :]
        lens = np.array([len(chunk_i)]).astype(np.int32)
        feats, feats_len = cache["frontend"].extract_fbank(waveforms, lens, is_final)
        feats = feats.astype(np.float32)
        if feats.shape[1] == 0:
            return []

        feats = feats * self._scale
        cache["is_final"] = is_final
        feats = self._pe.forward(feats, cache["start_idx"])
        cache["start_idx"] += feats.shape[1]

        cs = self._chunk_size
        if is_final:
            if feats.shape[1] + cs[2] <= cs[1]:
                # 小尾块: last_chunk, overlap 后补零到 sum(chunk_size)
                cache["last_chunk"] = True
                overlap = self._add_overlap_chunk(feats, cache)
                return [_ForwardRequest(overlap.astype(np.float32),
                                        np.array([overlap.shape[1]]).astype(np.int32),
                                        cache, last_chunk=True)]
            # 大尾块: 切成 chunk1 + last_chunk 两段 (参考实现语义)
            overlap1 = self._add_overlap_chunk(feats[:, : cs[1], :], cache)
            req1 = _ForwardRequest(overlap1.astype(np.float32),
                                   np.array([overlap1.shape[1]]).astype(np.int32),
                                   cache, last_chunk=False)
            cache["last_chunk"] = True
            overlap2 = self._add_overlap_chunk(
                feats[:, -(feats.shape[1] + cs[2] - cs[1]) :, :], cache
            )
            req2 = _ForwardRequest(overlap2.astype(np.float32),
                                   np.array([overlap2.shape[1]]).astype(np.int32),
                                   cache, last_chunk=True)
            return [req1, req2]

        overlap = self._add_overlap_chunk(feats, cache)
        return [_ForwardRequest(overlap.astype(np.float32),
                                np.array([overlap.shape[1]]).astype(np.int32),
                                cache, last_chunk=False)]

    def _add_overlap_chunk(self, feats: np.ndarray, cache: dict) -> np.ndarray:
        """镜像 funasr_onnx add_overlap_chunk (含 is_final/last_chunk 分支)。"""
        cs = self._chunk_size
        overlap = np.concatenate((cache["feats"], feats), axis=1)
        if cache.get("is_final"):
            cache["feats"] = overlap[:, -cs[0] :, :]
            if not cache.get("last_chunk"):
                padding_length = sum(cs) - overlap.shape[1]
                if padding_length > 0:
                    overlap = np.pad(overlap, ((0, 0), (0, padding_length), (0, 0)))
        else:
            cache["feats"] = overlap[:, -(cs[0] + cs[2]) :, :]
        return overlap

    # ------------------------------------------------------------------
    # 批前向 (batcher 线程): encoder ORT → CIF 积分 → decoder ORT → token
    # ------------------------------------------------------------------
    def _forward_group(self, requests: List[_ForwardRequest]):
        B = len(requests)
        caches = [r.cache for r in requests]
        last_chunk = requests[0].last_chunk

        feats = np.concatenate([r.feats for r in requests], axis=0)
        feats_len = np.concatenate([r.feats_len for r in requests]).astype(np.int32)

        # 1) encoder (动态 batch 轴)
        enc, enc_lens, alphas = self._enc_sess.run(
            ["enc", "enc_len", "alphas"], {self._enc_input_names[0]: feats,
                                           self._enc_input_names[1]: feats_len}
        )

        # 2) CIF 积分 (batched, cache 沿 batch 维 cat)
        cif_hidden = np.concatenate([c["cif_hidden"] for c in caches], axis=0)
        cif_alphas = np.concatenate([c["cif_alphas"] for c in caches], axis=0)
        acoustic_embeds, token_lens = self._cif_search(
            enc, alphas, cif_hidden, cif_alphas, last_chunk
        )
        # scatter CIF cache 回各流 (积分循环产出 (B,1,·), 直接行切片)
        for i, c in enumerate(caches):
            c["cif_hidden"] = cif_hidden[i : i + 1]
            c["cif_alphas"] = cif_alphas[i : i + 1]

        for r in requests:
            r.tokens = []
        if acoustic_embeds.shape[1] == 0:
            return

        # 3) decoder —— 必须按 CIF token 数分组执行:
        #    FSMN 记忆沿 acoustic_embeds 序列卷积, 把 token 少的行零填充到
        #    max 会让 padding 零帧卷进记忆 (实测导致后续 token 重复/丢失)。
        #    同 token 数的行无 padding, 可安全合批; 0-token 组跳过 decoder
        #    (与单流参考语义一致)。
        by_tokcount: Dict[int, List[int]] = {}
        for i, tl in enumerate(token_lens):
            by_tokcount.setdefault(int(tl), []).append(i)

        for tokcount, idxs in by_tokcount.items():
            if tokcount <= 0:
                continue
            sub = np.asarray(idxs)
            dec_inputs = {
                self._dec_input_names[0]: enc[sub],
                self._dec_input_names[1]: enc_lens[sub].astype(np.int32),
                self._dec_input_names[2]: acoustic_embeds[sub][:, :tokcount, :],
                self._dec_input_names[3]: np.full(len(sub), tokcount, dtype=np.int32),
            }
            for li in range(self._fsmn_layers):
                dec_inputs[self._dec_input_names[4 + li]] = np.concatenate(
                    [caches[i]["decoder_fsmn"][li] for i in idxs], axis=0
                )

            outs = self._dec_sess.run(None, dec_inputs)
            sample_ids = outs[1]
            out_caches = outs[2:]
            for row, i in enumerate(idxs):
                caches[i]["decoder_fsmn"] = [
                    out_caches[li][row : row + 1, :, -self._fsmn_lorder :]
                    for li in range(self._fsmn_layers)
                ]
                # 4) token 提取 (镜像参考 decode_one: 过滤 0/2, 截断到有效长度)
                ids = sample_ids[row][:tokcount]
                requests[i].tokens = [
                    self._tokens[t]
                    for t in ids
                    if t not in (0, 2) and 0 <= t < len(self._tokens)
                ]

    def _cif_search(self, hidden, alphas, cif_hidden, cif_alphas, last_chunk):
        """镜像 funasr_onnx cif_search 的 batched 版本。

        hidden: (B,T,D); alphas: (B,T); cif_hidden/cif_alphas: (B,1,·) —— 原地更新。
        返回 acoustic_embeds (B, maxTok, D) 与 token_lengths (B,)。
        """
        cs = self._chunk_size
        B = hidden.shape[0]
        alphas = alphas.copy()
        alphas[:, : cs[0]] = 0.0
        if not last_chunk:
            alphas[:, sum(cs[:2]) :] = 0.0

        hidden = np.concatenate((cif_hidden, hidden), axis=1)
        alphas = np.concatenate((cif_alphas, alphas), axis=1)
        if last_chunk:
            tail_hidden = np.zeros((B, 1, hidden.shape[2]), dtype=np.float32)
            tail_alphas = np.tile(
                np.array([[self._tail_threshold]], dtype=np.float32), (B, 1)
            )
            hidden = np.concatenate((hidden, tail_hidden), axis=1)
            alphas = np.concatenate((alphas, tail_alphas), axis=1)

        len_time = alphas.shape[1]
        token_lengths = []
        list_frames = []
        cache_alphas = []
        cache_hiddens = []
        for b in range(B):
            integrate = 0.0
            frames = np.zeros(hidden.shape[2], dtype=np.float32)
            list_frame = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self._cif_threshold:
                    integrate += alpha
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self._cif_threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    integrate -= self._cif_threshold
                    frames = integrate * hidden[b][t]
            cache_alphas.append(integrate)
            cache_hiddens.append(frames / integrate if integrate > 0.0 else frames)
            token_lengths.append(len(list_frame))
            list_frames.append(list_frame)

        # 写回 batched cache ((B,1,D)/(B,1), 行切片由调用方 scatter)
        cif_alphas[:, :] = np.stack(cache_alphas, axis=0).astype(np.float32).reshape(B, 1)
        cif_hidden[:, :] = np.stack(cache_hiddens, axis=0).astype(np.float32).reshape(B, 1, -1)

        max_token_len = max(token_lengths) if token_lengths else 0
        if max_token_len == 0:
            return (
                np.zeros((B, 0, hidden.shape[2]), dtype=np.float32),
                np.array(token_lengths, dtype=np.int32),
            )
        list_ls = []
        for b in range(B):
            pad = np.zeros((max_token_len - token_lengths[b], hidden.shape[2]), dtype=np.float32)
            if token_lengths[b] == 0:
                list_ls.append(pad)
            else:
                list_ls.append(np.concatenate((np.stack(list_frames[b]), pad), axis=0))
        return (
            np.stack(list_ls, axis=0).astype(np.float32),
            np.array(token_lengths, dtype=np.int32),
        )

    def close(self):
        if getattr(self, "_batcher", None) is not None:
            self._batcher.stop()

# TODO: select CPU/CUDA ExecutionProvider based on runtime device
