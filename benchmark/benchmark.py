#!/usr/bin/env python3
"""
Multi-session concurrent load tester and benchmark evaluation suite
for ASRFlow heterogeneous 2-pass service.
"""

import os
import sys
import json
import time
import math
import struct
import asyncio
import argparse
from typing import List, Dict, Any, Optional
import websockets
import numpy as np
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_speech_turn_pcm16(
    speech_duration_sec: float = 3.0,
    silence_duration_sec: float = 1.5,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a single speech turn: speech segment + trailing silence segment."""
    samples = []
    # 1. Speech samples (multi-frequency tones simulating voice formant harmonics)
    speech_samples = int(speech_duration_sec * sample_rate)
    for i in range(speech_samples):
        t = i / sample_rate
        val = (
            0.5 * math.sin(2 * math.pi * 260 * t)
            + 0.3 * math.sin(2 * math.pi * 520 * t)
            + 0.2 * math.sin(2 * math.pi * 780 * t)
        )
        int_val = int(val * 16000)
        samples.append(max(-32767, min(32767, int_val)))

    # 2. Silence samples (zeros)
    silence_samples = int(silence_duration_sec * sample_rate)
    samples.extend([0] * silence_samples)

    return struct.pack(f"<{len(samples)}h", *samples)


class SingleSessionBenchmark:
    """Simulates a single real-time streaming WebSocket client session."""

    def __init__(
        self,
        session_idx: int,
        uri: str,
        turns: int = 3,
        speech_sec: float = 3.0,
        silence_sec: float = 1.5,
        chunk_ms: int = 60,
    ):
        self.session_idx = session_idx
        self.uri = uri
        self.turns = turns
        self.speech_sec = speech_sec
        self.silence_sec = silence_sec
        self.chunk_ms = chunk_ms
        self.session_id = f"bench-session-{session_idx}-{int(time.time()*1000)}"

        # Metrics collected
        self.first_partial_latencies: List[float] = []
        self.provisional_latencies: List[float] = []
        self.final_latencies: List[float] = []
        self.revision_distances: List[float] = []
        self.total_audio_sec: float = turns * (speech_sec + silence_sec)
        self.sentences_finalized: int = 0
        self.fallback_count: int = 0
        self.success: bool = False
        self.error_msg: Optional[str] = None

    async def run(self):
        try:
            async with websockets.connect(self.uri) as ws:
                # 1. Send START initialization
                start_cmd = {
                    "type": "start",
                    "session_id": self.session_id,
                    "language": "zh",
                    "enable_spk": True,
                    "expected_speakers": 2,
                    "hotwords": ["星巴克", "电影院", "周末爬山"],
                }
                await ws.send(json.dumps(start_cmd))

                # Wait for session_ready
                ready_raw = await ws.recv()
                ready_data = json.loads(ready_raw)
                if ready_data.get("type") != "session_ready" and ready_data.get("event") != "started":
                    raise RuntimeError(f"Unexpected response to start: {ready_raw}")

                # Receiver Task
                turn_starts: Dict[int, float] = {}
                turn_ends: Dict[int, float] = {}
                first_partial_recorded = set()

                async def receiver_loop():
                    async for message in ws:
                        recv_ts = time.time()
                        try:
                            data = json.loads(message)
                            mtype = data.get("type") or data.get("mode")

                            if mtype in ("partial", "2pass-online") and not data.get("is_final", False):
                                sent_approx = data.get("seq", 0) // 10
                                if sent_approx not in first_partial_recorded and sent_approx in turn_starts:
                                    lat_ms = (recv_ts - turn_starts[sent_approx]) * 1000.0
                                    self.first_partial_latencies.append(lat_ms)
                                    first_partial_recorded.add(sent_approx)

                            elif mtype in ("provisional", "2pass-provisional"):
                                sent_id = data.get("sentence_id", 1) - 1
                                if sent_id in turn_ends:
                                    lat_ms = (recv_ts - turn_ends[sent_id]) * 1000.0
                                    self.provisional_latencies.append(lat_ms)

                            elif mtype in ("final", "2pass-offline") and (data.get("is_final") or mtype == "final"):
                                sent_id = data.get("sentence_id", 1) - 1
                                if sent_id in turn_ends:
                                    lat_ms = (recv_ts - turn_ends[sent_id]) * 1000.0
                                    self.final_latencies.append(lat_ms)
                                self.sentences_finalized += 1
                                if data.get("final_source") == "paraformer-fallback":
                                    self.fallback_count += 1
                                self.revision_distances.append(data.get("revision_distance", 0.0))

                            elif mtype == "session_finished" or data.get("event") == "stopped":
                                break
                        except Exception as e:
                            logger.warning(f"[{self.session_id}] Error parsing message: {e}")

                recv_task = asyncio.create_task(receiver_loop())

                # Stream Speech Turns
                bytes_per_chunk = int(16000 * 2 * (self.chunk_ms / 1000.0))  # 1920 bytes
                chunk_interval_sec = self.chunk_ms / 1000.0

                for turn_idx in range(self.turns):
                    turn_audio = generate_speech_turn_pcm16(
                        speech_duration_sec=self.speech_sec,
                        silence_duration_sec=self.silence_sec,
                    )
                    speech_bytes_len = int(self.speech_sec * 32000)
                    turn_starts[turn_idx] = time.time()

                    offset = 0
                    while offset < len(turn_audio):
                        chunk = turn_audio[offset : offset + bytes_per_chunk]
                        offset += len(chunk)

                        if offset >= speech_bytes_len and turn_idx not in turn_ends:
                            turn_ends[turn_idx] = time.time()

                        await ws.send(chunk)
                        await asyncio.sleep(chunk_interval_sec)

                # Send STOP
                await ws.send(json.dumps({"type": "stop"}))
                await asyncio.wait_for(recv_task, timeout=10.0)
                self.success = True

        except Exception as e:
            self.success = False
            self.error_msg = str(e)
            logger.error(f"[{self.session_id}] Benchmark session failed: {e}")


async def run_benchmark(
    uri: str,
    concurrency: int = 10,
    turns: int = 3,
    speech_sec: float = 3.0,
    silence_sec: float = 1.5,
    chunk_ms: int = 60,
    output_json: Optional[str] = None,
):
    logger.info("=================================================================")
    logger.info("Starting ASRFlow Concurrent Benchmark")
    logger.info(f"Target WebSocket URI: {uri}")
    logger.info(f"Concurrent Sessions: {concurrency}")
    logger.info(f"Speech Turns per Session: {turns}")
    logger.info(f"Turn: {speech_sec}s speech + {silence_sec}s silence")
    logger.info("=================================================================")

    start_wall_clock = time.time()

    sessions = [
        SingleSessionBenchmark(
            session_idx=i,
            uri=uri,
            turns=turns,
            speech_sec=speech_sec,
            silence_sec=silence_sec,
            chunk_ms=chunk_ms,
        )
        for i in range(concurrency)
    ]

    # Run sessions concurrently
    tasks = [s.run() for s in sessions]
    await asyncio.gather(*tasks)

    elapsed_wall_clock = time.time() - start_wall_clock

    # Collect and calculate metrics
    total_audio_sec = sum(s.total_audio_sec for s in sessions)
    successful_sessions = sum(1 for s in sessions if s.success)
    total_sentences = sum(s.sentences_finalized for s in sessions)
    total_fallbacks = sum(s.fallback_count for s in sessions)

    all_first_partial = []
    all_provisional = []
    all_finals = []
    all_revisions = []

    for s in sessions:
        all_first_partial.extend(s.first_partial_latencies)
        all_provisional.extend(s.provisional_latencies)
        all_finals.extend(s.final_latencies)
        all_revisions.extend(s.revision_distances)

    def calc_percentiles(arr: List[float]):
        if not arr:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
        np_arr = np.array(arr)
        return {
            "p50": float(np.percentile(np_arr, 50)),
            "p90": float(np.percentile(np_arr, 90)),
            "p95": float(np.percentile(np_arr, 95)),
            "p99": float(np.percentile(np_arr, 99)),
            "avg": float(np.mean(np_arr)),
        }

    p_first_partial = calc_percentiles(all_first_partial)
    p_provisional = calc_percentiles(all_provisional)
    p_finals = calc_percentiles(all_finals)
    avg_revision = float(np.mean(all_revisions)) if all_revisions else 0.0
    rtf = total_audio_sec / max(0.001, elapsed_wall_clock)
    fallback_rate = (total_fallbacks / max(1, total_sentences)) * 100.0

    print("\n" + "=" * 65)
    print("                BENCHMARK RESULTS REPORT")
    print("=" * 65)
    print(f"Total Sessions Simulated : {concurrency}")
    print(f"Successful Sessions       : {successful_sessions} / {concurrency} ({successful_sessions/concurrency*100:.1f}%)")
    print(f"Total Audio Processed    : {total_audio_sec:.1f} seconds")
    print(f"Wall-clock Elapsed Time  : {elapsed_wall_clock:.2f} seconds")
    print(f"Real-Time Factor (RTF)   : {rtf:.3f}")
    print(f"Total Sentences Finalized: {total_sentences}")
    print(f"Fallback Count / Rate    : {total_fallbacks} ({fallback_rate:.2f}%)")
    print("-" * 65)
    print("LATENCY DISTRIBUTION (ms):")
    print(f"  First Partial P50/P90/P95 : {p_first_partial['p50']:.1f} / {p_first_partial['p90']:.1f} / {p_first_partial['p95']:.1f} ms")
    print(f"  Provisional   P50/P90/P95 : {p_provisional['p50']:.1f} / {p_provisional['p90']:.1f} / {p_provisional['p95']:.1f} ms")
    print(f"  Final ASR     P50/P90/P95 : {p_finals['p50']:.1f} / {p_finals['p90']:.1f} / {p_finals['p95']:.1f} ms")
    print("-" * 65)
    print(f"Avg Revision Distance    : {avg_revision:.4f}")
    print("=" * 65 + "\n")

    report_data = {
        "concurrency": concurrency,
        "successful_sessions": successful_sessions,
        "total_audio_sec": total_audio_sec,
        "elapsed_wall_clock_sec": elapsed_wall_clock,
        "rtf": rtf,
        "total_sentences": total_sentences,
        "fallback_rate": fallback_rate,
        "latency_ms": {
            "first_partial": p_first_partial,
            "provisional": p_provisional,
            "final_asr": p_finals,
        },
        "avg_revision_distance": avg_revision,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Benchmark report exported to: {output_json}")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="ASRFlow Concurrent Benchmark")
    parser.add_argument("--uri", type=str, default="ws://127.0.0.1:10095", help="Target WebSocket server URI")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent sessions")
    parser.add_argument("--turns", type=int, default=3, help="Speech turns per session")
    parser.add_argument("--speech_sec", type=float, default=3.0, help="Duration of speech per turn (seconds)")
    parser.add_argument("--silence_sec", type=float, default=1.5, help="Duration of silence per turn (seconds)")
    parser.add_argument("--chunk_ms", type=int, default=60, help="Audio frame size in ms")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save JSON benchmark report")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            uri=args.uri,
            concurrency=args.concurrency,
            turns=args.turns,
            speech_sec=args.speech_sec,
            silence_sec=args.silence_sec,
            chunk_ms=args.chunk_ms,
            output_json=args.output_json,
        )
    )


if __name__ == "__main__":
    main()
