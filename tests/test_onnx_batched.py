import unittest

from config.settings import StreamingASRConfig


class TestOnnxBatchedEngineFactory(unittest.TestCase):
    """首遍引擎工厂分发 (不加载真实模型)。"""

    def test_factory_mock_backend(self):
        from core.streaming_asr import create_streaming_asr_engine
        from core.streaming_asr.mock_streaming import MockStreamingASREngine

        cfg = StreamingASRConfig(backend="mock")
        eng = create_streaming_asr_engine(cfg)
        self.assertIsInstance(eng, MockStreamingASREngine)

    def test_factory_unsupported_backend_raises(self):
        from core.streaming_asr import create_streaming_asr_engine

        with self.assertRaises(ValueError):
            create_streaming_asr_engine(StreamingASRConfig(backend="funasr"))

    def test_missing_model_dir_raises_with_explicit_backend(self):
        # 不依赖 onnxruntime: 引擎必须先校验目录, 再 import 重依赖。
        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        cfg = StreamingASRConfig(
            backend="onnx", onnx_model_dir="/nonexistent/onnx/dir"
        )
        with self.assertRaises((ValueError, FileNotFoundError)):
            OnnxBatchedStreamingEngine(cfg)

    def test_cache_signature_distinguishes_shapes(self):
        import numpy as np

        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        a = {
            "feats": np.zeros((1, 10, 560), dtype=np.float32),
            "cif_hidden": np.zeros((1, 1, 320), dtype=np.float32),
            "decoder_fsmn": [np.zeros((1, 320, 10), dtype=np.float32) for _ in range(12)],
        }
        b = {
            "feats": np.zeros((1, 5, 560), dtype=np.float32),
            "cif_hidden": np.zeros((1, 1, 320), dtype=np.float32),
            "decoder_fsmn": [np.zeros((1, 320, 10), dtype=np.float32) for _ in range(12)],
        }
        self.assertNotEqual(
            OnnxBatchedStreamingEngine._cache_signature(a),
            OnnxBatchedStreamingEngine._cache_signature(b),
        )
        c = dict(a)
        self.assertEqual(
            OnnxBatchedStreamingEngine._cache_signature(a),
            OnnxBatchedStreamingEngine._cache_signature(c),
        )

    def test_flush_on_pristine_cache_is_safe_noop(self):
        """强制切句清缓存后紧跟 COMMIT 的空 flush 不得崩溃。

        线上实测: 此时 funasr 前端会在 np.stack(空 lfr 缓存) 上抛
        "need at least one array to stack", 带断 WebSocket 连接。
        守卫应在触达前端之前按无结果收尾 (不依赖真实模型)。
        """
        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        eng = object.__new__(OnnxBatchedStreamingEngine)
        eng._model = object()  # 绕过未加载校验, 守卫在特征提取之前
        cache = {"start_idx": 0}
        self.assertEqual(eng.process_chunk(b"", cache, is_final=True), "")
        self.assertEqual(cache["start_idx"], 0)

    def test_non_final_empty_chunk_still_short_circuits(self):
        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        eng = object.__new__(OnnxBatchedStreamingEngine)
        eng._model = object()
        cache = {"start_idx": 0}
        self.assertEqual(eng.process_chunk(b"", cache, is_final=False), "")


class _FakeFrontend:
    def __init__(self, **kwargs):
        pass


def _make_engine_skeleton():
    """构造绕过模型加载的最小引擎骨架 (单测 _init_stream_cache 用)。"""
    from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

    eng = object.__new__(OnnxBatchedStreamingEngine)
    eng._frontend_cls = _FakeFrontend
    eng._frontend_conf = {}
    eng._cmvn_file = "/dev/null"
    eng._feats_dims = 560
    eng._keep = 10  # chunk_size[0] + chunk_size[2] (5+5)
    eng._enc_output_size = 320
    eng._fsmn_dims = 320
    eng._fsmn_layers = 12
    eng._fsmn_lorder = 10
    return eng


class TestStreamCacheReset(unittest.TestCase):
    """句末重置语义: 全键覆盖, 绝不清空共享 dict (线上 KeyError 回归)。"""

    def test_init_overwrites_all_engine_keys_without_clear(self):
        import numpy as np

        eng = _make_engine_skeleton()
        cache = {
            "frontend": "stale",
            "start_idx": 99,
            "feats": np.ones((1, 10, 560), dtype=np.float32),
            "cif_hidden": np.ones((1, 1, 320), dtype=np.float32),
            "cif_alphas": np.ones((1, 1), dtype=np.float32),
            "decoder_fsmn": [np.ones((1, 320, 10), dtype=np.float32) for _ in range(12)],
            "last_chunk": True,
            "is_final": True,  # 上一句遗留, 必须被复位 (否则泄漏到下一句)
            "_prev_samples": np.ones(4, dtype=np.float32),
        }
        eng._init_stream_cache(cache)
        self.assertEqual(cache["start_idx"], 0)
        self.assertFalse(cache["is_final"])
        self.assertFalse(cache["last_chunk"])
        self.assertEqual(cache["_prev_samples"].shape, (0,))
        self.assertIsInstance(cache["frontend"], _FakeFrontend)
        self.assertEqual(cache["feats"].shape, (1, 10, 560))
        # 引擎读取的键在重置后全部存在且类型正确
        self.assertEqual(cache["cif_hidden"].shape, (1, 1, 320))
        self.assertEqual(cache["cif_alphas"].shape, (1, 1))
        self.assertEqual(len(cache["decoder_fsmn"]), 12)
        self.assertEqual(cache["decoder_fsmn"][0].shape, (1, 320, 10))

    def test_reset_key_set_is_closed_over_engine_reads(self):
        """init 写入的键集必须覆盖引擎全部读取键 —— 去 clear() 后无残留旧键的前提。"""
        eng = _make_engine_skeleton()
        cache = {}
        eng._init_stream_cache(cache)
        expected = {
            "frontend", "start_idx", "feats", "cif_hidden", "cif_alphas",
            "decoder_fsmn", "last_chunk", "is_final", "_prev_samples",
        }
        self.assertTrue(expected.issubset(cache.keys()))


class TestBatcherThreadResilience(unittest.TestCase):
    """毒化 cache 的请求不得杀死 batcher 线程 (线程死亡 = 全会话前向静默超时)。"""

    def test_batcher_survives_poisoned_and_failing_requests(self):
        import numpy as np

        from core.streaming_asr.onnx_batched_streaming import (
            _ForwardBatcher,
            _ForwardRequest,
            OnnxBatchedStreamingEngine,
        )

        eng = object.__new__(OnnxBatchedStreamingEngine)
        eng._model = object()
        eng._max_batch = 4
        eng._batch_window_sec = 0.0
        eng._submit_timeout_sec = 5.0
        eng._record_wait = lambda wait: None
        reload_reasons = []
        eng._note_error = lambda e: "test-reload"
        eng._note_success = lambda: None
        eng._start_reload = lambda reason: reload_reasons.append(reason)

        calls = {"n": 0}

        def fake_forward(reqs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("forward boom")
            for r in reqs:
                r.tokens = ["ok"]

        eng._forward_group = fake_forward
        batcher = _ForwardBatcher(eng)
        try:
            def make_req(cache):
                return _ForwardRequest(
                    np.zeros((1, 10, 560), dtype=np.float32),
                    np.array([10], dtype=np.int32),
                    cache,
                    False,
                )

            # 1) 毒化 cache (缺签名键): 分组失败, 请求拿空 tokens, 线程存活
            self.assertEqual(batcher.submit(make_req({})), [])
            # 2) 前向抛错: 单路重试也失败, 走 reload 记录, 线程仍存活
            self.assertEqual(batcher.submit(make_req(_valid_cache())), [])
            self.assertEqual(reload_reasons, ["test-reload"])
            # 3) 后续正常请求照常出 token —— 线程未死
            self.assertEqual(batcher.submit(make_req(_valid_cache())), ["ok"])
            self.assertEqual(calls["n"], 2)
        finally:
            batcher.stop()


def _valid_cache():
    import numpy as np

    return {
        "feats": np.zeros((1, 10, 560), dtype=np.float32),
        "cif_hidden": np.zeros((1, 1, 320), dtype=np.float32),
        "cif_alphas": np.zeros((1, 1), dtype=np.float32),
        "decoder_fsmn": [np.zeros((1, 320, 10), dtype=np.float32) for _ in range(12)],
    }


class TestOrtProvidersForDevice(unittest.TestCase):
    """首遍 ONNX Runtime 执行设备 Provider 解析测试。"""

    def test_providers_for_cpu(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(_ort_providers_for_device("cpu", avail), ["CPUExecutionProvider"])
        self.assertEqual(_ort_providers_for_device("CPU", avail), ["CPUExecutionProvider"])

    def test_providers_for_cuda_default(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(
            _ort_providers_for_device("cuda", avail),
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )

    def test_providers_for_cuda_device_id(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(
            _ort_providers_for_device("cuda:0", avail),
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        self.assertEqual(
            _ort_providers_for_device("cuda:1", avail),
            [("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
        )
        self.assertEqual(
            _ort_providers_for_device(" CUDA:2 ", avail),
            [("CUDAExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"],
        )

    def test_providers_for_cuda_unavailable_fallback(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        # ORT 仅有 CPUExecutionProvider (如只安装了 cpu 版 onnxruntime)
        avail = ["CPUExecutionProvider"]
        self.assertEqual(_ort_providers_for_device("cuda:0", avail), ["CPUExecutionProvider"])
        self.assertEqual(_ort_providers_for_device("cuda", avail), ["CPUExecutionProvider"])

    def test_providers_for_auto_device(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        # auto 且 CUDA 可用 -> cuda:0
        self.assertEqual(
            _ort_providers_for_device("auto", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        # auto 且 CUDA 不可用 -> cpu
        self.assertEqual(
            _ort_providers_for_device("auto", ["CPUExecutionProvider"]),
            ["CPUExecutionProvider"],
        )
        # None / 空串 等价 auto
        self.assertEqual(
            _ort_providers_for_device(None, ["CPUExecutionProvider"]),
            ["CPUExecutionProvider"],
        )
        self.assertEqual(
            _ort_providers_for_device("", ["CPUExecutionProvider"]),
            ["CPUExecutionProvider"],
        )

    def test_providers_for_invalid_device(self):
        from core.streaming_asr.onnx_batched_streaming import _ort_providers_for_device

        avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(_ort_providers_for_device("unknown_dev", avail), ["CPUExecutionProvider"])
        self.assertEqual(
            _ort_providers_for_device("cuda:abc", avail),
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        self.assertEqual(
            _ort_providers_for_device("cuda:-1", avail),
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )

    def test_init_model_cuda_session_error_fallback(self):
        import os
        import sys
        import tempfile
        from unittest.mock import MagicMock, patch

        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname in ("model.onnx", "decoder.onnx", "tokens.json", "config.yaml", "am.mvn"):
                with open(os.path.join(tmp_dir, fname), "w", encoding="utf-8") as f:
                    if fname == "tokens.json":
                        f.write('["<blank>", "a", "b"]')
                    elif fname == "config.yaml":
                        f.write(
                            "frontend_conf: {n_mels: 80, lfr_m: 7}\n"
                            "encoder_conf: {output_size: 320}\n"
                            "decoder_conf: {num_blocks: 12, kernel_size: 11}\n"
                            "predictor_conf: {threshold: 1.0, tail_threshold: 0.45}\n"
                        )
                    else:
                        f.write("mock")

            cfg = StreamingASRConfig(
                backend="onnx",
                onnx_model_dir=tmp_dir,
                onnx_quantize=False,
                device="cuda:0",
            )

            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            mock_cpu_sess = MagicMock()
            mock_cpu_sess.get_providers.return_value = ["CPUExecutionProvider"]
            mock_input = MagicMock()
            mock_input.name = "in"
            mock_cpu_sess.get_inputs.return_value = [mock_input]

            def fake_session(path, opts, providers=None):
                if providers and any(
                    (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider") or p == "CUDAExecutionProvider"
                    for p in providers
                ):
                    raise RuntimeError("CUDA device init failed: out of memory")
                return mock_cpu_sess

            mock_ort.InferenceSession.side_effect = fake_session
            mock_frontend_mod = MagicMock()

            with patch.dict(
                sys.modules,
                {
                    "onnxruntime": mock_ort,
                    "funasr_onnx.utils.frontend": mock_frontend_mod,
                },
            ):
                eng = OnnxBatchedStreamingEngine(cfg)
                self.assertIsNotNone(eng._enc_sess)
                self.assertEqual(eng._enc_sess.get_providers(), ["CPUExecutionProvider"])
                batcher = getattr(eng, "_batcher", None)
                if batcher:
                    batcher.stop()


if __name__ == "__main__":
    unittest.main()
