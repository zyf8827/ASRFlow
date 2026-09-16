#!/usr/bin/env python3
"""
ASRFlow 端到端真实链路测试客户端。

对运行中的 ASRFlow 网关推送本地音频（mp3/wav/flac，自动重采样为 16kHz/mono/PCM16），
验证 流式 Partial -> Provisional -> Final(Qwen3-ASR 二遍) -> session_finished 全链路，
并输出统计报告；全部断言通过返回退出码 0，否则非 0。

用法:
    python3 scripts/e2e_test.py --audio fixtures/sample_16k.wav --rate 2.0
    python3 scripts/e2e_test.py --uri ws://127.0.0.1:10095 --audio fixtures/sample_16k.wav
"""

import os
import sys
import json
import time
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import websockets


TARGET_SR = 16000


def load_audio_16k_pcm(path: str) -> bytes:
    """解码任意音频文件为 16kHz 单声道 PCM16 字节流 (mp3 等压缩格式走 ffmpeg)。"""
    if path.lower().endswith(".wav"):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != TARGET_SR:
            import soxr

            mono = soxr.resample(mono, sr, TARGET_SR)
        return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    import subprocess

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
            "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return proc.stdout


async def run_e2e(
    uri: str, audio_path: str, rate: float, chunk_ms: int, finish_timeout: float, max_sec: float = 0.0
) -> bool:
    audio_bytes = load_audio_16k_pcm(audio_path)
    if max_sec and max_sec > 0:
        audio_bytes = audio_bytes[: int(TARGET_SR * 2 * max_sec)]
    duration_sec = len(audio_bytes) / 2 / TARGET_SR
    chunk_size = int(TARGET_SR * 2 * chunk_ms / 1000)
    n_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
    print(f"[Audio] {audio_path}: {duration_sec:.1f}s @16kHz mono, {n_chunks} chunks, rate={rate}x")

    stats = {
        "partial": 0,
        "provisional": [],
        "final": [],
        "errors": [],
        "session_finished": None,
    }
    finished = asyncio.Event()

    async def receiver(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "partial":
                    stats["partial"] += 1
                elif mtype == "provisional":
                    stats["provisional"].append(msg)
                    print(f"  [Provisional #{msg.get('sentence_id')}] {msg.get('text')}")
                elif mtype == "final":
                    stats["final"].append(msg)
                    print(
                        f"  [Final #{msg.get('sentence_id')}] ({msg.get('speaker')}, "
                        f"src={msg.get('final_source')}, review={msg.get('needs_review')}) "
                        f"{msg.get('text')}"
                    )
                elif mtype == "error":
                    stats["errors"].append(msg)
                    print(f"  [Error] {msg}")
                elif mtype == "session_finished":
                    stats["session_finished"] = msg
                    print("[Session Finished]")
                    finished.set()
                    return
                elif mtype == "session_ready":
                    print(f"[Session Ready] id={msg.get('session_id')}")
        except websockets.ConnectionClosed as e:
            stats["errors"].append({"type": "error", "message": f"connection closed: {e}"})

    t_start = time.time()
    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "session_id": f"e2e-{int(time.time())}",
                    "language": "zh",
                    "enable_spk": True,
                    "expected_speakers": 2,
                }
            )
        )
        recv_task = asyncio.create_task(receiver(ws))
        await asyncio.sleep(0.5)  # 等待 session_ready

        t_stream = time.time()
        offset = 0
        interval = chunk_ms / 1000.0 / rate
        while offset < len(audio_bytes):
            if finished.is_set():
                break
            chunk = audio_bytes[offset : offset + chunk_size]
            offset += len(chunk)
            await ws.send(chunk)
            await asyncio.sleep(interval)
        stream_cost = time.time() - t_stream

        await ws.send(json.dumps({"type": "commit"}))
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(finished.wait(), timeout=finish_timeout)
        except asyncio.TimeoutError:
            stats["errors"].append(
                {"type": "error", "message": f"no session_finished within {finish_timeout}s"}
            )
        recv_task.cancel()

    total_cost = time.time() - t_start

    finals = stats["final"]
    qwen_finals = [f for f in finals if f.get("final_source") == "qwen3-asr"]
    fallback_finals = [f for f in finals if f.get("final_source") != "qwen3-asr"]
    speakers = sorted({str(f.get("speaker")) for f in finals})
    full_text = "".join(str(f.get("text")) for f in qwen_finals)

    print("\n" + "=" * 70)
    print("E2E 测试报告")
    print("=" * 70)
    print(f"音频时长          : {duration_sec:.1f}s | 推流耗时 {stream_cost:.1f}s | 总耗时 {total_cost:.1f}s")
    print(f"Partial 帧数      : {stats['partial']}")
    print(f"Provisional 句数  : {len(stats['provisional'])}")
    print(f"Final 句数        : {len(finals)} (Qwen3-ASR: {len(qwen_finals)}, fallback: {len(fallback_finals)})")
    print(f"说话人标签        : {speakers}")
    print(f"错误              : {len(stats['errors'])} {stats['errors'] if stats['errors'] else ''}")
    print(f"session_finished  : {'yes' if stats['session_finished'] else 'NO'}")
    print("-" * 70)
    preview = full_text[:300] + ("..." if len(full_text) > 300 else "")
    print(f"二遍最终文本预览:\n{preview}")

    checks = {
        "收到 Final 结果": len(finals) > 0,
        "Qwen3-ASR 二遍生效": len(qwen_finals) > 0,
        "无协议错误": len(stats["errors"]) == 0,
        "会话正常收尾": stats["session_finished"] is not None,
    }
    print("-" * 70)
    all_pass = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass &= ok
    print("=" * 70)
    print("E2E RESULT:", "PASS" if all_pass else "FAIL")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="ASRFlow E2E Test Client")
    parser.add_argument("--uri", default="ws://127.0.0.1:10095")
    parser.add_argument("--audio", default="fixtures/sample_16k.wav", help="mp3/wav/flac 音频文件")
    parser.add_argument("--rate", type=float, default=1.0, help="推流倍速 (1.0=实时)")
    parser.add_argument("--max-sec", type=float, default=0.0, help="只取音频前 N 秒 (0=完整)")
    parser.add_argument("--chunk-ms", type=int, default=60)
    parser.add_argument("--finish-timeout", type=float, default=180.0, help="stop 后等待收尾超时(s)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[ERROR] audio not found: {args.audio}", file=sys.stderr)
        sys.exit(2)

    ok = asyncio.run(
        run_e2e(args.uri, args.audio, args.rate, args.chunk_ms, args.finish_timeout, args.max_sec)
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
