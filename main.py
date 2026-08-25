#!/usr/bin/env python3
"""ASRFlow entrypoint (early skeleton)."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ASRFlow gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10095)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"ASRFlow websocket skeleton on {args.host}:{args.port}")


if __name__ == "__main__":
    main()
