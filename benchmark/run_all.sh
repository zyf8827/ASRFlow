#!/usr/bin/env bash
# =============================================================================
# 全量压测总控: 顺序执行 首遍 Paraformer 套件 + 二遍 Qwen vLLM 套件。
#
# 设计要点:
#   - 两个套件顺序执行 (并行会互相干扰: 首遍测 CPU 容量, 二遍测远程 GPU);
#   - 每个套件带看门狗超时 (timeout), 卡死也能保证后续套件执行并最终留痕;
#   - 进度/退出码/产物目录追加记录到 benchmark/results/run_all.status;
#   - 每个套件结束后清扫可能遗留的网关测试进程 (基准端口 210xx)。
#
# 用法 (建议脱离会话运行, 关掉终端不影响):
#   setsid nohup bash benchmark/run_all.sh > benchmark/results/run_all_console.log 2>&1 &
#
# 可调: PARA_TIMEOUT / QWEN_TIMEOUT (看门狗秒数)
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

STATUS="$ROOT/benchmark/results/run_all.status"
PARA_TIMEOUT="${PARA_TIMEOUT:-7200}"   # 首遍套件看门狗(s)
QWEN_TIMEOUT="${QWEN_TIMEOUT:-3600}"   # 二遍套件看门狗(s)

mkdir -p "$(dirname "$STATUS")"
echo "[run_all] START $(date '+%F %T') pid=$$ para_timeout=${PARA_TIMEOUT}s qwen_timeout=${QWEN_TIMEOUT}s" >> "$STATUS"

sweep_orphans() {
    # 清扫网关套件可能遗留的测试实例 (默认基准端口 210xx; 生产 10095 不受影响)
    pkill -f "main\.py --device cpu --port 210" 2>/dev/null || true
    sleep 1
}

echo "[run_all] [1/2] 首遍 Paraformer 套件 ..."
timeout --kill-after=60 "${PARA_TIMEOUT}" bash benchmark/run_paraformer_suite.sh
RC1=$?
sweep_orphans
LATEST_PARA="$(ls -dt "$ROOT"/benchmark/results/*_paraformer_suite 2>/dev/null | head -1 || true)"
echo "[run_all] paraformer_suite EXIT=$RC1 $(date '+%F %T') results=$LATEST_PARA" >> "$STATUS"

echo "[run_all] [2/2] 二遍 Qwen vLLM 套件 ..."
timeout --kill-after=60 "${QWEN_TIMEOUT}" bash benchmark/run_qwen_suite.sh
RC2=$?
LATEST_QWEN="$(ls -dt "$ROOT"/benchmark/results/*_qwen_suite 2>/dev/null | head -1 || true)"
echo "[run_all] qwen_suite EXIT=$RC2 $(date '+%F %T') results=$LATEST_QWEN" >> "$STATUS"

echo "[run_all] ALL_DONE $(date '+%F %T') rc_paraformer=$RC1 rc_qwen=$RC2 (0=正常, 124=看门狗超时)" >> "$STATUS"
