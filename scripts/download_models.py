#!/usr/bin/env python3
"""
Model download helper script for ASRFlow Service.
Supports ModelScope and Hugging Face Hub download.
"""

import os
import sys
import argparse
from typing import List, Dict


MODELS = {
    "paraformer_streaming": {
        "model_id": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        "revision": "v2.0.4",
        "description": "Paraformer (small) Streaming 16k Online ASR Model (Pass-1, ONNX export source)",
    },
    "fsmn_vad": {
        "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "v2.0.4",
        "description": "FSMN-VAD 16k Voice Activity Detection Model (Unified Timeline)",
    },
    "eres2net_speaker": {
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "revision": "master",
        "description": "Speaker Embedding & Verification Model (Diarization)",
    },
    "ct_punc": {
        "model_id": "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        "revision": "v2.0.4",
        "description": "CT-Transformer Real-time Punctuation Restoration Model",
    },
    "qwen3_asr": {
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "revision": "main",
        "description": "Qwen3-ASR-1.7B High Precision Final ASR Model (Pass-2 Final)",
    },
    "qwen3_asr_small": {
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "revision": "main",
        "description": "Qwen3-ASR-0.6B Lite Final ASR Model (Pass-2 Final, lower compute)",
    },
}


def download_from_modelscope(target_dir: str, include_qwen: bool = True):
    print("=" * 65)
    print("Downloading Models from ModelScope Hub...")
    print(f"Target Directory: {target_dir}")
    print("=" * 65)

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("[ERROR] 'modelscope' package is not installed. Please run: pip install modelscope")
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    for key, info in MODELS.items():
        if key.startswith("qwen") and not include_qwen:
            continue
        model_id = info["model_id"]
        revision = info.get("revision", "master")
        print(f"\n--> Downloading [{key}] {info['description']}")
        print(f"    Model ID: {model_id} (revision: {revision})")
        try:
            path = snapshot_download(
                model_id=model_id,
                revision=revision,
                cache_dir=target_dir,
            )
            print(f"    [OK] Downloaded to: {path}")
        except Exception as e:
            print(f"    [WARNING] Failed to download {model_id}: {e}")

    print("\n" + "=" * 65)
    print("Model download process completed!")
    print("=" * 65)


def download_from_huggingface(target_dir: str, include_qwen: bool = True):
    print("=" * 65)
    print("Downloading Models from Hugging Face Hub...")
    print(f"Target Directory: {target_dir}")
    print("=" * 65)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 'huggingface_hub' package is not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    for key, info in MODELS.items():
        if key.startswith("qwen") and not include_qwen:
            continue
        model_id = info["model_id"]
        print(f"\n--> Downloading [{key}] {info['description']}")
        print(f"    Model ID: {model_id}")
        try:
            path = snapshot_download(
                repo_id=model_id,
                local_dir=os.path.join(target_dir, model_id.replace("/", "_")),
            )
            print(f"    [OK] Downloaded to: {path}")
        except Exception as e:
            print(f"    [WARNING] Failed to download {model_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="ASRFlow Model Downloader")
    parser.add_argument(
        "--hub",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Model hub source (default: modelscope)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=os.path.expanduser("~/.cache/modelscope/hub"),
        help="Target local cache directory for models",
    )
    parser.add_argument(
        "--include_qwen",
        action="store_true",
        default=True,
        help="Include Qwen3-ASR-1.7B in download",
    )
    parser.add_argument(
        "--only_funasr",
        action="store_true",
        help="Download only FunASR CPU models (Paraformer, VAD, Speaker, Punc)",
    )
    args = parser.parse_args()

    include_qwen = args.include_qwen and not args.only_funasr

    if args.hub == "modelscope":
        download_from_modelscope(args.target_dir, include_qwen=include_qwen)
    else:
        download_from_huggingface(args.target_dir, include_qwen=include_qwen)


if __name__ == "__main__":
    main()
