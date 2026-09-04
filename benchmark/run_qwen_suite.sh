#!/usr/bin/env bash
# =============================================================================
# 二遍 Qwen3-ASR (远程 vLLM, 单卡 4090) 并发压测套件 —— 编排脚本
#
# 基于 config/config.default.yaml 的 final_asr 配置 (默认 127.0.0.1:8899,
# qwen3-asr-1.7b), 直连 vLLM HTTP 接口 (复用生产 Qwen3ASREngine 的请求行为),
# 并发从 1 逐步加压, 绘制: 延迟-并发曲线 / 吞吐-并发曲线 / 超时率曲线。
#
# 用法:
#   bash benchmark/run_qwen_suite.sh
#
# 可调参数 (环境变量覆盖):
#   CONFIG / AUDIO / CONCURRENCY / SEG_SEC_LIST / REQUESTS_PER_LEVEL
#   TIMEOUT (默认 30: 测真实容量曲线; 设 8 则复现生产硬超时口径的超时率)
#   COOLDOWN
#
# 产物: benchmark/results/<时间戳>_qwen_suite/
#   ├── qwen/ qwen_vllm.csv + qwen_vllm_details.json + bench_qwen_vllm.log
#   ├── manifest.json / suite.log
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
[[ -n "$PY" ]] || { echo "[ERROR] 未找到 Python 解释器"; exit 2; }

CONFIG="${CONFIG:-$ROOT/config/config.default.yaml}"
AUDIO="${AUDIO:-$ROOT/fixtures/sample_16k.wav}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,24,32}"     # 并发级别 (单 4090 从 1 加到 32)
SEG_SEC_LIST="${SEG_SEC_LIST:-3,10}"               # 段长: 短句/长句两种 VAD 断句形态
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-24}"     # 每级别总请求数 (采样量)
TIMEOUT="${TIMEOUT:-30}"                           # 单请求超时(s); 生产口径改 8
COOLDOWN="${COOLDOWN:-8}"

TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/benchmark/results/${TAG}_qwen_suite"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/suite.log") 2>&1

echo "=============================================================="
echo " 二遍 Qwen3-ASR 远程 vLLM 并发压测套件"
echo " 配置: $CONFIG | 音频: $AUDIO"
echo " 并发: $CONCURRENCY | 段长: $SEG_SEC_LIST | 每级别 $REQUESTS_PER_LEVEL 个请求"
echo " 超时: ${TIMEOUT}s | 输出: $RUN_ROOT"
echo "=============================================================="
"$PY" - <<EOF
import json, sys
sys.path.insert(0, "$ROOT")
from benchmark.bench_common import host_info
info = {"suite": "qwen_vllm", "started": "$TAG", "host": host_info(),
        "concurrency": "$CONCURRENCY".split(","), "seg_sec_list": "$SEG_SEC_LIST".split(","),
        "requests_per_level": $REQUESTS_PER_LEVEL, "timeout_sec": $TIMEOUT}
with open("$RUN_ROOT/manifest.json", "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
EOF

"$PY" benchmark/bench_qwen_vllm.py \
    --config "$CONFIG" \
    --audio "$AUDIO" \
    --concurrency "$CONCURRENCY" \
    --seg_sec_list "$SEG_SEC_LIST" \
    --requests_per_level "$REQUESTS_PER_LEVEL" \
    --timeout "$TIMEOUT" \
    --cooldown "$COOLDOWN" \
    --out_dir "$RUN_ROOT/qwen"

echo ""
echo "=============================================================="
echo " 套件完成. 产物:"
find "$RUN_ROOT" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.log" \) | sort
echo "=============================================================="
