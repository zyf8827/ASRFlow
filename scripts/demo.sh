#!/usr/bin/env bash
# ASRFlow Demo 客户端启动脚本（默认连接本机服务 ws://127.0.0.1:10095）
#
# 用法:
#   bash scripts/demo.sh /path/to/audio.wav     # 位置参数直接指定音频文件
#   bash scripts/demo.sh                        # 推流 sample_16k.wav 实时语速
#   bash scripts/demo.sh -f                     # 其余参数透传给 demo/console_client.py
#   bash scripts/demo.sh /path/x.wav --full     # 音频 + 透传参数混用
#
# 参数修改 (环境变量, 未设置的沿用 console_client 默认值):
#   DEMO_AUDIO=/path/to/a.wav             # 音频文件 (mp3/wav/flac), 也可用位置参数传入
#   DEMO_PORT=10095                       # 本机服务 WS 端口 (生成默认 uri)
#   DEMO_URI=ws://127.0.0.1:10095         # 直接指定服务地址 (覆盖 DEMO_PORT)
#   DEMO_HTTP_PORT=10096                  # 本机就绪检查端口 (仅 uri 为本机时使用)
#   DEMO_RATE=2.0                         # 推流倍速 (1.0=实时; >1.1 才能测 RTF 上界)
#   DEMO_MAX_SEC=60                       # 只取前 N 秒 (0=完整)
#   DEMO_HOTWORDS="星巴克,电影院"          # 逗号分隔热词
#   DEMO_SPEAKERS=2                       # 期望说话人数
#   DEMO_NO_COLOR=1                       # 禁用彩色/光标控制
#   DEMO_FULL=1                           # 结束时打印完整转写
#   DEMO_SKIP_CHECK=1                     # 跳过本机服务就绪检查
#
# 示例:
#   DEMO_RATE=2.0 DEMO_MAX_SEC=30 bash scripts/demo.sh
#   DEMO_URI=ws://127.0.0.1:10095 DEMO_AUDIO=fixtures/sample_16k.wav bash scripts/demo.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ASRFLOW_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    if command -v python3 > /dev/null 2>&1; then
        PY="$(command -v python3)"
    else
        echo "[ERROR] 未找到 Python 解释器 (先执行: uv venv --python 3.10 .venv && uv pip install -r requirements.txt)" >&2
        exit 2
    fi
fi

# 未显式指定地址/端口时, 优先读取服务运行环境文件记录的实际端口
ENV_FILE="$ROOT/logs/asrflow.env"
if [[ -f "$ENV_FILE" ]]; then
    if [[ -z "${DEMO_URI:-}" && -z "${DEMO_PORT:-}" ]]; then
        DEMO_PORT="$(sed -n 's/^PORT=//p' "$ENV_FILE" | head -1)"
    fi
    if [[ -z "${DEMO_HTTP_PORT:-}" ]]; then
        DEMO_HTTP_PORT="$(sed -n 's/^HTTP_PORT=//p' "$ENV_FILE" | head -1)"
    fi
fi
DEMO_PORT="${DEMO_PORT:-10095}"
DEMO_HTTP_PORT="${DEMO_HTTP_PORT:-10096}"
URI="${DEMO_URI:-ws://127.0.0.1:$DEMO_PORT}"
AUDIO="${DEMO_AUDIO:-$ROOT/fixtures/sample_16k.wav}"
# 位置参数指定音频文件 (不以 - 开头), 优先级: 位置参数 > DEMO_AUDIO > 默认 sample_16k.wav
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    AUDIO="$1"
    shift
fi

if [[ ! -f "$AUDIO" ]]; then
    echo "[ERROR] 音频文件不存在: $AUDIO (位置参数或 DEMO_AUDIO 指定)" >&2
    exit 2
fi

# 连接本机服务时先做就绪检查, 避免客户端长时间挂等
if [[ "${DEMO_SKIP_CHECK:-0}" != "1" ]]; then
    case "$URI" in
        ws://127.0.0.1:*|ws://localhost:*|ws://\[::1\]:*)
            if ! curl -sf --max-time 2 "http://127.0.0.1:$DEMO_HTTP_PORT/healthz" > /dev/null 2>&1; then
                echo "[ERROR] 本机服务未运行 (http://127.0.0.1:$DEMO_HTTP_PORT/healthz 不可达)" >&2
                echo "    先启动服务: python3 main.py" >&2
                echo "    连接其他地址: DEMO_URI=ws://<host>:<port> bash scripts/demo.sh" >&2
                exit 2
            fi
            ;;
    esac
fi

ARGS=(--uri "$URI" --audio "$AUDIO")
[[ -n "${DEMO_RATE:-}" ]] && ARGS+=(--rate "$DEMO_RATE")
[[ -n "${DEMO_MAX_SEC:-}" ]] && ARGS+=(--max-sec "$DEMO_MAX_SEC")
[[ -n "${DEMO_HOTWORDS:-}" ]] && ARGS+=(--hotwords "$DEMO_HOTWORDS")
[[ -n "${DEMO_SPEAKERS:-}" ]] && ARGS+=(--expected_speakers "$DEMO_SPEAKERS")
[[ "${DEMO_NO_COLOR:-0}" == "1" ]] && ARGS+=(--no-color)
[[ "${DEMO_FULL:-0}" == "1" ]] && ARGS+=(--full)

cd "$ROOT"
exec "$PY" demo/console_client.py "${ARGS[@]}" "$@"
