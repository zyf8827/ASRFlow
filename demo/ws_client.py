#!/usr/bin/env python3
"""
Simple WebSocket Streaming Client for ASRFlow Service.
"""

import os
import sys
import json
import asyncio
import argparse
import websockets
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_client(uri: str, wav_path: str, chunk_ms: int = 60):
    logger.info(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        # Start session
        session_id = "test-session-cli"
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "session_id": session_id,
                    "language": "zh",
                    "enable_spk": True,
                    "expected_speakers": 2,
                    "hotwords": ["星巴克", "电影院"],
                }
            )
        )

        async def receiver():
            async for msg in ws:
                data = json.loads(msg)
                mtype = data.get("type") or data.get("mode")
                if mtype in ("partial", "2pass-online") and not data.get("is_final", False):
                    print(f"[Partial] {data.get('text')}")
                elif mtype in ("provisional", "2pass-provisional"):
                    print(f"[Provisional] #{data.get('sentence_id')}: {data.get('text')}")
                elif mtype in ("final", "2pass-offline") and (data.get("is_final") or mtype == "final"):
                    print(
                        f"[Final] #{data.get('sentence_id')} [{data.get('speaker')}]: {data.get('text')} (src={data.get('final_source')})"
                    )
                elif mtype == "session_finished":
                    print("[Session Finished]")
                    break

        recv_task = asyncio.create_task(receiver())

        # Send Audio
        if wav_path and os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                header = f.read(44)  # Skip wav header
                data = f.read()
        else:
            # Synthetic 4s PCM16
            data = b"\x10\x00" * (16000 * 4)

        chunk_size = int(16000 * 2 * (chunk_ms / 1000.0))
        offset = 0
        while offset < len(data):
            chunk = data[offset : offset + chunk_size]
            offset += len(chunk)
            await ws.send(chunk)
            await asyncio.sleep(chunk_ms / 1000.0)

        # Stop session
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(recv_task, timeout=10.0)


def main():
    parser = argparse.ArgumentParser(description="Simple Streaming Client")
    parser.add_argument("--uri", type=str, default="ws://127.0.0.1:10095", help="WebSocket URI")
    parser.add_argument("--wav", type=str, default=None, help="WAV file path")
    parser.add_argument("--chunk_ms", type=int, default=60, help="Chunk duration in ms")
    args = parser.parse_args()

    asyncio.run(run_client(args.uri, args.wav, args.chunk_ms))


if __name__ == "__main__":
    main()
