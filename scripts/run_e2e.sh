#!/usr/bin/env bash
# ASRFlow 端到端真实链路一键测试:
#   启动网关(CPU + ONNX 首遍 + FunASR VAD/说话人) -> 等待就绪 -> 推流 sample_16k.wav -> 校验二遍 Qwen3-ASR 结果
#
# 用法:
#   bash scripts/run_e2e.sh                 # 2 倍速推流 fixtures/sample_16k.wav 前 60 秒
#   RATE=1.0 bash scripts/run_e2e.sh        # 真实语速推流
#   MAX_SEC=0 bash scripts/run_e2e.sh       # 完整音频
#   AUDIO=/path/to/x.wav bash scripts/run_e2e.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ASRFLOW_PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="/tmp/asrflow_e2e"
mkdir -p "$LOG_DIR"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8899/v1/audio/transcriptions}"
MODEL_NAME="${MODEL_NAME:-qwen3-asr-1.7b}"
RATE="${RATE:-2.0}"
MAX_SEC="${MAX_SEC:-60}"
AUDIO="${AUDIO:-$ROOT/fixtures/sample_16k.wav}"
WS_PORT="${WS_PORT:-10095}"
HTTP_PORT="${HTTP_PORT:-10096}"
# 测试音频句间停顿设置，设为 480ms 可平滑切句
VAD_MAX_SILENCE="${VAD_MAX_SILENCE:-480}"

if [[ ! -x "$PY" ]]; then
    if command -v python3 > /dev/null 2>&1; then
        PY="$(command -v python3)"
    else
        echo "[ERROR] 未找到 Python 解释器" >&2
        exit 2
    fi
fi

echo "==> 配置: vllm_url=$VLLM_URL model=$MODEL_NAME rate=${RATE}x audio=$AUDIO vad_silence=${VAD_MAX_SILENCE}ms"

# 0. 生成本次测试的临时配置 (指定二遍模型名与超时; ASR_CONFIG_PATH 为整体替换, 其余段落沿用代码默认值)
cat > "$LOG_DIR/e2e_config.yaml" <<EOF
final_asr:
  vllm_url: "$VLLM_URL"
  model_name: "$MODEL_NAME"
  hard_timeout_sec: 8.0
vad:
  max_end_silence_time: $VAD_MAX_SILENCE
EOF

# 1. 后台启动 ASRFlow 网关 (CPU + ONNX 首遍 + FunASR VAD/说话人 + 远程 Qwen3-ASR 二遍)
cd "$ROOT"
ASR_CONFIG_PATH="$LOG_DIR/e2e_config.yaml" VLLM_URL="$VLLM_URL" "$PY" main.py \
    --device cpu \
    --port "$WS_PORT" \
    --http_port "$HTTP_PORT" \
    --streaming_backend onnx \
    --vad_backend funasr \
    --speaker_backend funasr \
    --final_backend vllm_http \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

cleanup() {
    echo "==> 停止服务进程..."
}
trap 'cleanup; kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM

# 2. 轮询就绪 (/ready 在模型全部加载完成后才可用)
echo "==> 等待服务就绪 (首次需加载 ONNX 首遍与 FunASR VAD/说话人, 最长 10 分钟)..."
READY=0
for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$HTTP_PORT/ready" > "$LOG_DIR/ready.json" 2>/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] 服务进程已退出, 日志: $LOG_DIR/server.log" >&2
        tail -30 "$LOG_DIR/server.log" >&2
        exit 3
    fi
    sleep 2
done
if [[ "$READY" != "1" ]]; then
    echo "[ERROR] 服务 300 次轮询未就绪, 日志: $LOG_DIR/server.log" >&2
    exit 3
fi
echo "==> 服务就绪: $(cat "$LOG_DIR/ready.json")"

# 3. 运行端到端测试客户端
"$PY" scripts/e2e_test.py \
    --uri "ws://127.0.0.1:$WS_PORT" \
    --audio "$AUDIO" \
    --rate "$RATE" \
    --max-sec "$MAX_SEC"
E2E_RC=$?

echo "==> 服务日志: $LOG_DIR/server.log"
exit "$E2E_RC"
