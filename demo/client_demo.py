#!/usr/bin/env python3
"""
Interactive Client Demo for ASRFlow Inference Service.
Supports local WAV files, synthetic audio generation, and live streaming.
"""

import os
import sys
import json
import time
import math
import wave
import struct
import asyncio
import argparse
from typing import Optional, List
import websockets
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ANSI color codes for rich terminal display
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_GREEN = "\033[92m"
C_BLUE = "\033[94m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"


def print_banner():
    print(f"\n{C_BOLD}{C_BLUE}================================================================={C_RESET}")
    print(f"{C_BOLD}{C_GREEN}       ASRFlow: 2-Pass Speech Recognition Client Demo   {C_RESET}")
    print(f"{C_BOLD}{C_BLUE}================================================================={C_RESET}\n")


def generate_synthetic_audio(duration_sec: float = 8.0, sample_rate: int = 16000) -> bytes:
    """Generate tone-modulated simulated speech waveforms with intermittent pauses."""
    samples = []
    total_samples = int(duration_sec * sample_rate)
    for i in range(total_samples):
        t = i / sample_rate
        cycle_t = t % 4.0
        if cycle_t < 3.0:
            val = (
                0.5 * math.sin(2 * math.pi * 260 * t)
                + 0.3 * math.sin(2 * math.pi * 520 * t)
                + 0.2 * math.sin(2 * math.pi * 780 * t)
            )
            int_val = int(val * 16000)
        else:
            int_val = 0
        samples.append(max(-32767, min(32767, int_val)))
    return struct.pack(f"<{len(samples)}h", *samples)


def read_wav_pcm16(wav_path: str) -> bytes:
    """Read a standard WAV file into 16kHz mono PCM16 raw bytes."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

        if n_channels == 1 and sampwidth == 2 and framerate == 16000:
            return raw_bytes

        print(
            f"{C_YELLOW}[Notice] Input WAV ({framerate}Hz, {n_channels}ch, {sampwidth*8}bit) will be normalized to 16kHz Mono PCM16.{C_RESET}"
        )
        import numpy as np
        audio_arr = np.frombuffer(raw_bytes, dtype=np.int16 if sampwidth == 2 else np.int8)
        if n_channels > 1:
            audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        if framerate != 16000:
            from scipy import signal
            num_target_samples = int(len(audio_arr) * 16000 / framerate)
            audio_arr = signal.resample(audio_arr, num_target_samples).astype(np.int16)
        return audio_arr.tobytes()


async def stream_audio_client(
    uri: str,
    wav_path: Optional[str] = None,
    hotwords: Optional[List[str]] = None,
    postprocess_hotwords: Optional[str] = None,
    expected_speakers: int = 2,
    chunk_ms: int = 60,
    speed_rate: float = 1.0,
):
    print_banner()
    logger.info(f"Connecting to WebSocket: {C_BOLD}{uri}{C_RESET}")

    # Prepare Audio Data
    if wav_path and os.path.exists(wav_path):
        logger.info(f"Loading WAV file: {wav_path}")
        audio_data = read_wav_pcm16(wav_path)
    else:
        logger.info("No audio file provided; generating 8.0s synthetic speech audio...")
        audio_data = generate_synthetic_audio(duration_sec=8.0)

    total_sec = len(audio_data) / 32000.0
    bytes_per_chunk = int(16000 * 2 * (chunk_ms / 1000.0))  # 1920 bytes for 60ms

    logger.info(
        f"Audio payload ready: {len(audio_data)} bytes ({total_sec:.2f}s). Starting real-time stream..."
    )

    async with websockets.connect(uri) as ws:
        # 1. Send START initialization
        session_id = f"demo-session-{int(time.time()*1000)}"
        start_payload = {
            "type": "start",
            "session_id": session_id,
            "language": "zh",
            "enable_spk": True,
            "expected_speakers": expected_speakers,
            "hotwords": hotwords or ["星巴克", "电影院", "周末爬山"],
        }
        await ws.send(json.dumps(start_payload))

        if postprocess_hotwords:
            await ws.send(f"POSTPROCESS_HOTWORDS:{postprocess_hotwords}")

        # Receiver Task
        async def response_listener():
            async for message in ws:
                try:
                    data = json.loads(message)
                    mtype = data.get("type") or data.get("mode")

                    if mtype == "session_ready" or data.get("event") == "started":
                        print(
                            f"\n{C_BOLD}{C_GREEN}>>> [SESSION READY]{C_RESET} ID: {data.get('session_id', session_id)}"
                        )

                    elif mtype in ("partial", "2pass-online") and not data.get("is_final", False):
                        text = data.get("text", "")
                        start_ms = data.get("start_ms", data.get("begin_time", 0))
                        end_ms = data.get("end_ms", data.get("end_time", 0))
                        print(
                            f"\r{C_CYAN} [Partial {start_ms/1000:.1f}s~{end_ms/1000:.1f}s]{C_RESET} {text:<40}",
                            end="",
                            flush=True,
                        )

                    elif mtype in ("provisional", "2pass-provisional"):
                        text = data.get("text", "")
                        sent_id = data.get("sentence_id", 1)
                        start_ms = data.get("start_ms", data.get("begin_time", 0))
                        end_ms = data.get("end_ms", data.get("end_time", 0))
                        print(
                            f"\n{C_YELLOW} ↳ [Provisional #{sent_id} {start_ms/1000:.1f}s~{end_ms/1000:.1f}s]{C_RESET} {text}"
                        )

                    elif mtype in ("final", "2pass-offline") and (data.get("is_final") or mtype == "final"):
                        text = data.get("text", "")
                        sent_id = data.get("sentence_id", 1)
                        spk = data.get("speaker", "SPK1")
                        src = data.get("final_source", "qwen3-asr")
                        dist = data.get("revision_distance", 0.0)
                        start_ms = data.get("start_ms", data.get("begin_time", 0))
                        end_ms = data.get("end_ms", data.get("end_time", 0))

                        src_tag = f"{C_BLUE}[{src}]{C_RESET}"
                        review_tag = f" {C_RED}(Review Needed){C_RESET}" if data.get("needs_review") else ""
                        print(
                            f"{C_BOLD}{C_GREEN} ✔ [Final #{sent_id}] [{spk}] ({start_ms/1000:.1f}s~{end_ms/1000:.1f}s):{C_RESET} "
                            f"{C_BOLD}{text}{C_RESET} {src_tag} dist={dist:.2f}{review_tag}"
                        )

                    elif mtype == "session_finished" or data.get("event") == "stopped":
                        print(f"\n\n{C_BOLD}{C_BLUE}================== FINAL SESSION TRANSCRIPT =================={C_RESET}")
                        transcript = data.get("transcript") or data.get("sentences") or []
                        for row in transcript:
                            spk = row.get("speaker", "SPK1")
                            s_start = row.get("start_ms", row.get("start", 0)) / 1000.0
                            s_end = row.get("end_ms", row.get("end", 0)) / 1000.0
                            txt = row.get("text", "")
                            print(f"  {C_MAGENTA}[{spk}]{C_RESET} ({s_start:.2f}s - {s_end:.2f}s): {txt}")
                        print(f"{C_BOLD}{C_BLUE}=============================================================={C_RESET}\n")
                        break

                except Exception as e:
                    logger.warning(f"Failed to parse server message: {e}")

        recv_task = asyncio.create_task(response_listener())

        # 2. Push Audio Stream in Realtime chunks
        offset = 0
        sleep_interval = (chunk_ms / 1000.0) / max(0.1, speed_rate)

        while offset < len(audio_data):
            chunk = audio_data[offset : offset + bytes_per_chunk]
            offset += len(chunk)
            await ws.send(chunk)
            await asyncio.sleep(sleep_interval)

        # 3. Send STOP to trigger session-end finalization
        logger.info("Audio transmission completed. Sending STOP...")
        await ws.send(json.dumps({"type": "stop"}))

        # Wait for all final results
        await asyncio.wait_for(recv_task, timeout=10.0)


def main():
    parser = argparse.ArgumentParser(description="ASRFlow Client Demo")
    parser.add_argument("--uri", type=str, default="ws://127.0.0.1:10095", help="WebSocket URI")
    parser.add_argument("--wav", type=str, default=None, help="Path to input WAV file")
    parser.add_argument("--hotwords", nargs="+", default=["星巴克", "电影院", "周末爬山"], help="Hotwords list")
    parser.add_argument("--postprocess", type=str, default="星巴客=>星巴克,电应院=>电影院", help="Postprocess hotwords")
    parser.add_argument("--speakers", type=int, default=2, help="Expected speakers count")
    parser.add_argument("--chunk_ms", type=int, default=60, help="Frame duration in ms")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time)")
    args = parser.parse_args()

    asyncio.run(
        stream_audio_client(
            uri=args.uri,
            wav_path=args.wav,
            hotwords=args.hotwords,
            postprocess_hotwords=args.postprocess,
            expected_speakers=args.speakers,
            chunk_ms=args.chunk_ms,
            speed_rate=args.rate,
        )
    )


if __name__ == "__main__":
    main()
