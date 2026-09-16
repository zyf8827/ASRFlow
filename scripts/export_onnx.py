#!/usr/bin/env python3
"""
导出流式 Paraformer 为 ONNX (encoder/decoder/predictor 分图) + int8 量化。

funasr 1.4.3 的 export 调用 torch.onnx.export(dynamic_axes=...), 与 torch 2.13
的新导出器不兼容 (要求 dynamic_shapes); 此处以 legacy exporter
(torch.onnx.utils.export) monkeypatch 规避, 导出语义与旧版 torch 一致
(官方 funasr-runtime 的 ONNX 产物即来自该路径)。

用法:
  python scripts/export_onnx.py [--out models/onnx] [--quantize]
  # 默认导出小参数 Paraformer; --model 仅在指向本地快照目录时覆盖
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        help="Export source: small Paraformer (fixed). Override only for a local snapshot dir.",
    )
    ap.add_argument("--model_revision", default="v2.0.4")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "onnx"))
    ap.add_argument("--quantize", action="store_true", default=True)
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    args = ap.parse_args()

    import torch
    import torch.onnx

    # torch>=2.6: funasr 的 dynamic_axes 调用风格走 legacy exporter
    _legacy = torch.onnx.utils.export

    def _patched_export(*a, **kw):
        kw.pop("dynamo", None)
        return _legacy(*a, **kw)

    torch.onnx.export = _patched_export

    from funasr import AutoModel

    os.makedirs(args.out, exist_ok=True)
    m = AutoModel(
        model=args.model,
        model_revision=args.model_revision,
        device="cpu",
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )
    out = m.export(
        input=None,
        type="onnx",
        quantize=args.quantize,
        opset_version=14,
        output_dir=args.out,
    )
    # 补齐推理侧配套文件 (与官方 onnx 发布物对齐: config.yaml / am.mvn / tokens.json)
    import shutil
    import glob

    snapshot = None
    for p in glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "modelscope",
                                    "models", "*--*" + args.model.split("/")[-1], "snapshots", "*")):
        snapshot = p
        break
    if snapshot:
        for fname in ("config.yaml", "am.mvn", "tokens.json"):
            src = os.path.join(snapshot, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, fname))
                print(f"[export_onnx] copied {fname}")
    else:
        print("[export_onnx] WARN: modelscope snapshot not found; copy config.yaml/am.mvn/tokens.json manually")
    print(f"[export_onnx] done -> {out}")
    for f in sorted(os.listdir(out)):
        print(f"  {f}  ({os.path.getsize(os.path.join(out, f)) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
