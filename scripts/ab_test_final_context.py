#!/usr/bin/env python3
"""
二遍(final ASR) context 策略离线 A/B 对比测试。

- 标准结果(GT): mimo-v2.5-asr (OpenAI chat/completions)。实测两个坑:
  user 消息禁止携带 text part; 约 >90s 的大请求会随机丢内容(输出变短)。
  故按句边界切成 <=90s 的 mp3 小块(64kbps, 远低于 10MB 上限)并做双调用
  校验(两次字数差 >10% 时第三次仲裁取最长), 拼接为整段 GT, 按音频 sha 缓存复用。
- 二遍被测端点: 独立 vLLM Qwen3-ASR (transcriptions multipart API,
  config/config.default.yaml 的 final_asr), context 走 multipart prompt 字段,
  与 core/final_asr/qwen_engine.py 的 transcriptions 分支一致。

对比三种二遍调用方式:
  S1  不传 context（prompt 仅前缀）
  S2  旧线上：传该句首遍(Paraformer 流式)的 provisional 结果作 context，
      沿用当时 50 字截断（现网已改为 S3）
  S3  现网：传之前识别的历史结果（前序句子的二遍输出拼接，取尾部 --history_chars 字）

流程:
  1. 输入音频(默认 fixtures/sample_16k.wav) ffmpeg 转 16kHz/mono/PCM16;
  2. mimo 完整识别整段拿到 GT（已识别过则直接复用缓存）;
  3. 用与 pipeline/session_pipeline.py 相同的状态机本地跑真实链路分句
     (FSMN-VAD + force-cut, 默认 20s 对齐二遍可接受时长) 与真实首遍
     (ONNX Paraformer 流式, flush+ITN)，得到每句边界与首遍文本;
  4. 对每句按三种场景分别调 qwen transcriptions 接口
     (prompt = 前缀 + "(前文: ...)"，复刻 qwen_engine._build_prompt);
  5. 各场景分段结果拼接后与 GT 计算归一化 CER，输出对比报告，
     全部结果(每句文本/首遍/延迟)落盘 JSON。

用法（必须在仓库根目录执行）:
  .venv/bin/python scripts/ab_test_final_context.py
  .venv/bin/python scripts/ab_test_final_context.py --audio fixtures/sample_16k.wav --max_sec 60  # 冒烟
  .venv/bin/python scripts/ab_test_final_context.py --refresh                      # 忽略缓存重跑
"""

import os
import sys
import json
import time
import asyncio
import hashlib
import argparse
import subprocess
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp

from config.settings import VADConfig, StreamingASRConfig, ITNConfig
from core.vad import create_vad_engine
from core.streaming_asr import create_streaming_asr_engine
from core.itn.itn_processor import ITNProcessor
from core.final_asr.qwen_engine import pcm_to_wav_bytes, clean_qwen_asr_text

TARGET_SR = 16000
BYTES_PER_MS = TARGET_SR * 2 // 1000  # 32
PROMPT_PREFIX = "语音转写："  # 与 FinalASRConfig.prompt_prefix 线上默认一致
CACHE_VERSION = 3

DEFAULT_GT_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_GT_MODEL = "mimo-v2.5-asr"
DEFAULT_GT_API_KEY = os.environ.get("MIMO_API_KEY", "")
# 二遍端点 = 默认本地 vLLM transcriptions API
DEFAULT_QWEN_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8899/v1/audio/transcriptions")
DEFAULT_QWEN_MODEL = "qwen3-asr-1.7b"
DEFAULT_QWEN_API_KEY = "EMPTY"


def load_audio_16k_pcm(path: str, max_sec: float) -> bytes:
    """任意音频 -> 16kHz/mono/PCM16 字节流（压缩格式走 ffmpeg）。"""
    if path.lower().endswith(".wav"):
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != TARGET_SR:
            import soxr

            mono = soxr.resample(mono, sr, TARGET_SR)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    else:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", path,
                "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
                "-f", "s16le", "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        pcm = proc.stdout
    if max_sec and max_sec > 0:
        pcm = pcm[: int(TARGET_SR * 2 * max_sec)]
    return pcm


def build_prompt(context: Optional[str], ctx_limit: int) -> str:
    """复刻 qwen_engine._build_prompt（无热词场景）。ctx_limit=50 与线上一致。"""
    prompt = PROMPT_PREFIX
    ctx = (context or "").strip()[:ctx_limit]
    if ctx:
        prompt += f" (前文: {ctx})"
    return prompt


# ---------------------------------------------------------------------------
# 本地真实链路：VAD 分句 + 首遍流式识别（状态机复刻 pipeline/session_pipeline.py）
# ---------------------------------------------------------------------------

def run_first_pass(pcm: bytes, chunk_ms: int, max_speech_ms: int) -> List[Dict[str, Any]]:
    """
    驱动真实 FSMN-VAD + ONNX Paraformer 流式引擎，返回每个句子的
    {start_ms, end_ms, first_pass}。首遍文本 = 句内累计 partial + flush 尾巴，
    再过 ITN —— 与线上 _handle_sentence_endpoint 的 provisional 拼法一致。
    """
    vad_cfg = VADConfig()
    vad_cfg.max_speech_duration_ms = max_speech_ms
    vad = create_vad_engine(vad_cfg)
    asr = create_streaming_asr_engine(StreamingASRConfig())
    print(f"[FirstPass] VAD engine: {type(vad).__name__}, "
          f"streaming engine: {type(asr).__name__}")
    if "Mock" in type(vad).__name__ or "Mock" in type(asr).__name__:
        asr.close()
        raise RuntimeError(
            "VAD/首遍引擎回退到了 Mock（缺模型或依赖），本测试需要真实引擎；"
            "请在仓库根目录运行并确认 models/ 完整"
        )

    itn = ITNProcessor(ITNConfig())
    total_ms = len(pcm) // BYTES_PER_MS
    chunk_len = BYTES_PER_MS * chunk_ms

    vad_cache: Dict[str, Any] = {}
    asr_cache: Dict[str, Any] = {}
    partial = ""
    in_speech = False
    speech_start_ms = -1
    last_end_ms = 0
    segments: List[Dict[str, Any]] = []

    def sentence_endpoint(start_ms: int, end_ms: int) -> None:
        nonlocal partial, asr_cache
        if end_ms - start_ms < 100:  # 丢弃文件尾部的零长度/过短残留段
            partial = ""
            asr_cache.clear()
            return
        try:
            tail = asr.flush(asr_cache, None)
        except ValueError:
            # 句尾残留不足一个 LFR 窗时前端缓存为空, flush 无尾巴可吐
            tail = ""
        text = itn.normalize((partial + tail).strip())
        partial = ""
        asr_cache.clear()
        segments.append(
            {"start_ms": start_ms, "end_ms": end_ms, "first_pass": text}
        )

    try:
        for off in range(0, len(pcm), chunk_len):
            cur = pcm[off : off + chunk_len]
            t_ms = (off + len(cur)) // BYTES_PER_MS

            vad_segments = vad.process_chunk(cur, vad_cache, False)
            p = asr.process_chunk(cur, asr_cache, False, None)
            if p:
                partial += p

            for seg_start, seg_end in vad_segments:
                if seg_start != -1:
                    in_speech = True
                    speech_start_ms = seg_start
                if seg_end != -1:
                    in_speech = False
                    actual_start = (
                        speech_start_ms
                        if speech_start_ms != -1
                        else max(0, seg_end - 3000)
                    )
                    actual_start = max(actual_start, last_end_ms)
                    actual_end = max(seg_end, actual_start)
                    speech_start_ms = -1
                    last_end_ms = max(last_end_ms, actual_end)
                    sentence_endpoint(actual_start, actual_end)

            # force-cut：连续语音无停顿时的强切（同线上）
            if (
                max_speech_ms > 0
                and in_speech
                and speech_start_ms != -1
                and t_ms - speech_start_ms >= max_speech_ms
            ):
                sentence_endpoint(max(speech_start_ms, last_end_ms), t_ms)
                speech_start_ms = t_ms
                last_end_ms = t_ms

        if in_speech and speech_start_ms != -1:
            sentence_endpoint(max(speech_start_ms, last_end_ms), total_ms)
    finally:
        asr.close()

    print(f"[FirstPass] {total_ms/1000:.1f}s 音频 -> {len(segments)} 句, "
          f"平均 {total_ms/max(len(segments),1)/1000:.1f}s/句")
    return segments


# ---------------------------------------------------------------------------
# GT 客户端: mimo chat/completions
# 实测两个坑: 1) user 消息禁 text part; 2) 大请求(约 >90s 音频)会随机丢内容
# (同块两次调用字数大幅波动), 故用 mp3 小块 + 双调用校验规避。
# ---------------------------------------------------------------------------

def pcm_to_mp3_bytes(pcm: bytes, bitrate: str = "64k") -> bytes:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(TARGET_SR),
            "-ac", "1", "-i", "pipe:0", "-b:a", bitrate, "-f", "mp3", "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=True,
    )
    return proc.stdout


class MimoGTClient:
    def __init__(self, api_base: str, model: str, api_key: str):
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def _call_once(
        self, pcm_bytes: bytes, max_tokens: int, timeout_sec: float
    ) -> str:
        import base64

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(
                                    pcm_to_mp3_bytes(pcm_bytes)
                                ).decode("utf-8"),
                                "format": "mp3",
                            },
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        session = await self._get_session()
        last_err = ""
        for attempt in range(3):
            try:
                async with session.post(
                    self.url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_sec),
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        last_err = f"HTTP {resp.status}: {body[:300]}"
                        raise RuntimeError(last_err)
                    data = json.loads(body)
                    choices = data.get("choices", [])
                    if not choices or "message" not in choices[0]:
                        last_err = f"unexpected response: {body[:300]}"
                        raise RuntimeError(last_err)
                    if choices[0].get("finish_reason") == "length":
                        print("[GT] 警告: finish_reason=length, 该块可能被 "
                              "max_tokens 截断")
                    return clean_qwen_asr_text(
                        choices[0]["message"].get("content", "")
                    )
            except Exception as e:  # noqa: BLE001 - 重试一切瞬时错误
                last_err = str(e)
                if attempt < 2:
                    await asyncio.sleep(3.0 * (attempt + 1))
        raise RuntimeError(f"mimo GT 调用重试 3 次仍失败: {last_err}")

    async def transcribe_full(
        self, pcm_bytes: bytes, max_tokens: int, timeout_sec: float
    ) -> str:
        """双调用校验: 该端点会随机丢内容(输出显著变短)。两次长度差 >10%
        时追加第三次仲裁, 取最长结果; 一致则取较长者。"""
        best = await self._call_once(pcm_bytes, max_tokens, timeout_sec)
        again = await self._call_once(pcm_bytes, max_tokens, timeout_sec)
        if abs(len(again) - len(best)) > 0.10 * max(len(best), 1):
            third = await self._call_once(pcm_bytes, max_tokens, timeout_sec)
            best = max((best, again, third), key=len)
            print(f"[GT] 块结果不稳定, 三次取最长: {len(best)} 字")
        elif len(again) > len(best):
            best = again
        return best

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# 二遍客户端: vLLM transcriptions multipart（context 走 prompt 字段, 同线上）
# ---------------------------------------------------------------------------

class QwenFinalClient:
    def __init__(self, url: str, model: str, api_key: str):
        self.url = url
        self.model = model
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.latencies: List[float] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def transcribe(
        self, pcm_bytes: bytes, context: Optional[str], ctx_limit: int,
        timeout_sec: float,
    ) -> str:
        if not pcm_bytes:
            return ""
        wav_data = pcm_to_wav_bytes(pcm_bytes)
        data = aiohttp.FormData()
        data.add_field(
            "file", wav_data, filename="audio.wav", content_type="audio/wav"
        )
        data.add_field("model", self.model)
        data.add_field("prompt", build_prompt(context, ctx_limit))

        session = await self._get_session()
        last_err = ""
        for attempt in range(3):
            t0 = time.time()
            try:
                async with session.post(
                    self.url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout_sec),
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        last_err = f"HTTP {resp.status}: {body[:300]}"
                        raise RuntimeError(last_err)
                    result = json.loads(body)
                    self.latencies.append(time.time() - t0)
                    return clean_qwen_asr_text(result.get("text", ""))
            except Exception as e:  # noqa: BLE001 - 重试一切瞬时错误
                last_err = str(e)
                if attempt < 2:
                    await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"qwen 二遍调用重试 3 次仍失败: {last_err}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# 打分：归一化 CER
# ---------------------------------------------------------------------------

def normalize_text(t: str) -> str:
    """全角->半角、小写、去空白与标点，仅保留中英文数字。"""
    if not t:
        return ""
    out = []
    for ch in t:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        ch = chr(code)
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def levenshtein(ref: str, hyp: str) -> int:
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
            )
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    nref, nhyp = normalize_text(ref), normalize_text(hyp)
    if not nref:
        return 0.0
    return levenshtein(nref, nhyp) / len(nref)


def build_gt_chunks(
    segments: List[Dict[str, Any]], total_ms: int, chunk_sec: int
) -> List[tuple]:
    """长音频按句边界分块（贪婪累计到 chunk_sec 即在句末切开），
    保证单块 wav base64 后低于网关 10MB 上限。"""
    if not chunk_sec or chunk_sec <= 0:
        return [(0, total_ms)]
    limit = chunk_sec * 1000
    cuts = []
    last = 0
    for seg in segments:
        if seg["end_ms"] - last >= limit:
            cuts.append(seg["end_ms"])
            last = seg["end_ms"]
    if last < total_ms:
        cuts.append(total_ms)
    bounds = [0] + cuts
    return list(zip(bounds[:-1], bounds[1:]))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def cache_path(cache_dir: str, audio: str, audio_sha: str, max_sec: float) -> str:
    stem = os.path.splitext(os.path.basename(audio))[0]
    sec = int(max_sec) if max_sec else 0
    return os.path.join(cache_dir, f"{stem}_{audio_sha[:8]}_{sec}.json")


def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == CACHE_VERSION:
            return data
    return {"version": CACHE_VERSION}


def save_cache(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def run_scenario(
    client: QwenFinalClient,
    pcm: bytes,
    segments: List[Dict[str, Any]],
    contexts: List[Optional[str]],
    ctx_limit: int,
    timeout_sec: float,
    sem: asyncio.Semaphore,
    label: str,
) -> List[str]:
    results: List[Optional[str]] = [None] * len(segments)

    async def one(i: int) -> None:
        async with sem:
            seg = segments[i]
            audio_slice = pcm[
                seg["start_ms"] * BYTES_PER_MS : seg["end_ms"] * BYTES_PER_MS
            ]
            results[i] = await client.transcribe(
                audio_slice, contexts[i], ctx_limit, timeout_sec
            )

    t0 = time.time()
    await asyncio.gather(*(one(i) for i in range(len(segments))))
    print(f"[{label}] {len(segments)} 句完成, 耗时 {time.time()-t0:.1f}s")
    return [r or "" for r in results]


def write_report(
    args: argparse.Namespace,
    cache: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> str:
    """导出人工对比 markdown: 汇总表 + GT/三场景全文 + 逐句并排。"""
    os.makedirs(args.report_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    path = os.path.join(args.report_dir, f"{stem}_对比.md")
    total_ms = segments[-1]["end_ms"] if segments else 0

    fp_concat = "".join(s["first_pass"] for s in segments)
    rows = [
        ("首遍(Paraformer,参考)", fp_concat),
        ("S1 无context", "".join(cache["s1"])),
        ("S2 首遍结果(线上同款)", "".join(cache["s2"])),
        ("S3 历史结果", "".join(cache["s3"])),
    ]
    baseline = cer(cache["gt"], rows[1][1])

    lines: List[str] = []
    a = lines.append
    a(f"# 二遍 context 策略对比 — {args.audio}")
    a("")
    a(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- 音频时长: {total_ms/1000:.1f}s, 分句数: {len(segments)} "
      f"(force-cut {args.max_speech_ms/1000:.0f}s)")
    a(f"- GT(标准结果): {args.gt_model} 整段识别")
    a(f"- 二遍被测: {args.qwen_model} ({args.qwen_url})")
    a(f"- S2 截断 {args.s2_ctx_chars} 字(线上同款); "
      f"S3 历史 {args.history_chars} 字")
    a("")
    a("| 场景 | CER(vs GT) | vs S1 | 字数 |")
    a("|---|---|---|---|")
    for name, text in rows:
        c = cer(cache["gt"], text)
        delta = "" if name.startswith("S1") and "无" in name else (
            f"{(c - baseline) * 100:+.2f}pp"
        )
        a(f"| {name} | {c:.2%} | {delta} | {len(text)} |")
    a("")
    a("## GT (标准结果)")
    a("")
    a(cache["gt"])
    a("")
    for name, text in rows[1:]:
        a(f"## {name} 全文")
        a("")
        a(text if text.strip() else "（空）")
        a("")
    a("## 逐句对比")
    a("")
    for i, seg in enumerate(segments):
        dur = (seg["end_ms"] - seg["start_ms"]) / 1000
        a(f"### 句{i} [{seg['start_ms']/1000:.1f}s ~ "
          f"{seg['end_ms']/1000:.1f}s] {dur:.1f}s")
        a("")
        a(f"- 首遍(S2的context): {seg['first_pass'] or '（空）'}")
        a(f"- S1 无context: {cache['s1'][i] or '（空）'}")
        a(f"- S2 首遍结果: {cache['s2'][i] or '（空）'}")
        a(f"- S3 历史结果: {cache['s3'][i] or '（空）'}")
        a("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


async def main() -> int:
    ap = argparse.ArgumentParser(description="二遍 context 策略 A/B 对比测试")
    ap.add_argument("--audio", default="fixtures/sample_16k.wav", help="输入音频 (默认 fixtures/sample_16k.wav)")
    ap.add_argument("--max_sec", type=float, default=0,
                    help="只取前 N 秒做快速冒烟 (0=全长)")
    # GT (标准结果)
    ap.add_argument("--gt_api_base", default=DEFAULT_GT_API_BASE)
    ap.add_argument("--gt_model", default=DEFAULT_GT_MODEL)
    ap.add_argument("--gt_api_key", default=DEFAULT_GT_API_KEY,
                    help="默认取环境变量 MIMO_API_KEY")
    ap.add_argument("--gt_max_tokens", type=int, default=4096)
    ap.add_argument("--gt_timeout", type=float, default=300,
                    help="单块 GT 请求超时(秒)")
    ap.add_argument("--gt_chunk_sec", type=int, default=90,
                    help="GT 分块时长(s); 该端点 >90s 会随机丢内容, 勿调大")
    # 二遍 (被测端点, 默认本地 vLLM)
    ap.add_argument("--qwen_url", default=DEFAULT_QWEN_URL)
    ap.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--qwen_api_key", default=DEFAULT_QWEN_API_KEY)
    ap.add_argument("--timeout", type=float, default=60, help="二遍单句超时(秒)")
    # 场景参数
    ap.add_argument("--history_chars", type=int, default=200,
                    help="S3 历史上下文保留的尾部字数 (默认 200)")
    ap.add_argument("--s2_ctx_chars", type=int, default=50,
                    help="S2 上下文截断字数, 50 与线上 _build_prompt 一致")
    ap.add_argument("--max_speech_ms", type=int, default=20000,
                    help="force-cut 单句上限(ms)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--chunk_ms", type=int, default=100, help="首遍/VAD 喂入块大小")
    ap.add_argument("--cache_dir", default=".asr_ab_cache")
    ap.add_argument("--report_dir", default=".asr_ab_cache/reports",
                    help="人工对比 markdown 报告输出目录")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存全部重算")
    ap.add_argument("--refresh_gt", action="store_true",
                    help="只重算标准结果(GT), 分句/首遍/三场景走缓存")
    args = ap.parse_args()

    if not args.gt_api_key:
        print("[ERROR] 未设置 MIMO_API_KEY 环境变量或 --gt_api_key 参数。请设置有效密钥后重试。", file=sys.stderr)
        return 1

    with open(args.audio, "rb") as f:
        audio_sha = hashlib.sha256(f.read()).hexdigest()
    cpath = cache_path(args.cache_dir, args.audio, audio_sha, args.max_sec)
    cache = {"version": CACHE_VERSION} if args.refresh else load_cache(cpath)
    meta = {
        "gt_model": args.gt_model,
        "qwen_model": args.qwen_model,
        "qwen_url": args.qwen_url,
        "max_speech_ms": args.max_speech_ms,
    }
    if cache.get("meta") != meta:  # 端点/模型变了, 重算
        cache = {"version": CACHE_VERSION, "meta": meta}
    else:
        cache["meta"] = meta

    pcm = load_audio_16k_pcm(args.audio, args.max_sec)
    total_ms = len(pcm) // BYTES_PER_MS
    print(f"[Audio] {args.audio}: {total_ms/1000:.1f}s @16kHz mono PCM16")
    print(f"[Cache]  {cpath}")

    gt_client = MimoGTClient(args.gt_api_base, args.gt_model, args.gt_api_key)
    qwen_client = QwenFinalClient(
        args.qwen_url, args.qwen_model, args.qwen_api_key
    )
    sem = asyncio.Semaphore(args.concurrency)
    try:
        # ---- Step 1: 真实链路分句 + 首遍（本地 ONNX，无 API 调用） ----
        if cache.get("segments"):
            print(f"[FirstPass] 命中缓存: {len(cache['segments'])} 句")
        else:
            cache["segments"] = run_first_pass(
                pcm, args.chunk_ms, args.max_speech_ms
            )
            save_cache(cpath, cache)
        segments = [
            s for s in cache["segments"] if s["end_ms"] - s["start_ms"] >= 100
        ]
        cache["segments"] = segments
        if not segments:
            print("[Error] VAD 未检出任何语音段")
            return 1

        # ---- Step 2: 标准结果 GT（mimo 完整识别, 小块 mp3 + 双调用校验） ----
        if args.refresh_gt:
            cache.pop("gt", None)
            cache.pop("gt_chunks", None)
        if cache.get("gt"):
            print(f"[GT] 命中缓存: {cache['gt'][:60]}...")
        else:
            chunks = build_gt_chunks(segments, total_ms, args.gt_chunk_sec)
            texts: List[str] = cache.get("gt_chunks", [])
            print(f"[GT] 调用 {args.gt_model} 识别整段音频 ({len(chunks)} 块)...")
            for cs, ce in chunks[len(texts):]:
                texts.append(await gt_client.transcribe_full(
                    pcm[cs * BYTES_PER_MS: ce * BYTES_PER_MS],
                    args.gt_max_tokens, args.gt_timeout,
                ))
                cache["gt_chunks"] = texts
                save_cache(cpath, cache)
                print(f"  [GT] 块{len(texts)}/{len(chunks)} "
                      f"[{cs/1000:.0f}~{ce/1000:.0f}s]: {len(texts[-1])} 字")
            cache["gt"] = "".join(texts)
            save_cache(cpath, cache)
            print(f"[GT] 完成: {cache['gt'][:60]}...")

        # ---- Step 3: 三种场景二遍调用 (qwen transcriptions) ----
        scenarios: Dict[str, List[str]] = {}

        # S1 不传 context
        if cache.get("s1"):
            print(f"[S1] 命中缓存: {len(cache['s1'])} 句")
        else:
            scenarios["s1"] = await run_scenario(
                qwen_client, pcm, segments,
                contexts=[None] * len(segments),
                ctx_limit=0, timeout_sec=args.timeout,
                sem=sem, label="S1 无context",
            )

        # S2 与线上一致：context = 该句首遍结果
        if cache.get("s2"):
            print(f"[S2] 命中缓存: {len(cache['s2'])} 句")
        else:
            scenarios["s2"] = await run_scenario(
                qwen_client, pcm, segments,
                contexts=[s["first_pass"] for s in segments],
                ctx_limit=args.s2_ctx_chars, timeout_sec=args.timeout,
                sem=sem, label="S2 首遍结果",
            )

        # S3 传历史：前序句子的二遍输出(S2)拼接，取尾部 history_chars 字
        if cache.get("s3"):
            print(f"[S3] 命中缓存: {len(cache['s3'])} 句")
        else:
            s2_done = scenarios.get("s2") or cache["s2"]
            hist_ctx: List[Optional[str]] = []
            acc = ""
            for prev in s2_done:
                hist_ctx.append(acc or None)
                acc = (acc + prev)[-args.history_chars:]
            scenarios["s3"] = await run_scenario(
                qwen_client, pcm, segments,
                contexts=hist_ctx,
                ctx_limit=args.history_chars, timeout_sec=args.timeout,
                sem=sem, label="S3 历史结果",
            )

        for k, v in scenarios.items():
            cache[k] = v
        save_cache(cpath, cache)

        # ---- Step 4: 打分报告 ----
        gt = cache["gt"]
        first_pass_concat = "".join(s["first_pass"] for s in segments)
        rows = {
            "S1 无context": "".join(cache["s1"]),
            "S2 首遍结果(线上同款)": "".join(cache["s2"]),
            "S3 历史结果": "".join(cache["s3"]),
        }

        lat = qwen_client.latencies
        lat_str = (
            f"avg {sum(lat)/len(lat):.2f}s, max {max(lat):.2f}s, n={len(lat)}"
            if lat else "全部命中缓存"
        )

        print("\n" + "=" * 72)
        print(f"二遍 context 策略对比  |  音频: {args.audio} "
              f"({total_ms/1000:.1f}s, {len(segments)} 句)")
        print(f"GT = {args.gt_model} 整段识别  |  二遍 = {args.qwen_model} "
              f"({args.qwen_url})")
        print(f"二遍调用延迟: {lat_str}")
        print("=" * 72)
        print(f"--- GT (标准结果) ---\n{gt}\n")
        print(f"{'场景':<22}{'CER(vs GT)':>12}{'字数':>8}")
        print("-" * 42)
        print(f"{'首遍(Paraformer,参考)':<22}"
              f"{cer(gt, first_pass_concat):>11.2%}{len(first_pass_concat):>8}")
        baseline = cer(gt, rows["S1 无context"])
        for name, text in rows.items():
            c = cer(gt, text)
            delta = (c - baseline) * 100
            marker = "" if name.startswith("S1") else f"  ({delta:+.2f}pp vs S1)"
            print(f"{name:<22}{c:>11.2%}{len(text):>8}{marker}")

        # 差异样例：三场景输出不完全一致的句子（最多展示 5 句）
        diff_shown = 0
        for i, seg in enumerate(segments):
            if diff_shown >= 5:
                break
            texts = {cache["s1"][i], cache["s2"][i], cache["s3"][i]}
            if len(texts) > 1:
                diff_shown += 1
                dur = (seg["end_ms"] - seg["start_ms"]) / 1000
                print(f"\n--- 差异样例 {diff_shown} | 句{i} [{seg['start_ms']}~"
                      f"{seg['end_ms']}ms] {dur:.1f}s ---")
                print(f"首遍(S2 ctx): {seg['first_pass']}")
                print(f"S1: {cache['s1'][i]}")
                print(f"S2: {cache['s2'][i]}")
                print(f"S3: {cache['s3'][i]}")

        report_path = write_report(args, cache, segments)
        print(f"\n[Report] 完整结果已保存: {cpath}")
        print(f"[Report] 人工对比报告: {report_path}")
        return 0
    finally:
        await gt_client.close()
        await qwen_client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
