#!/usr/bin/env bash
# =============================================================================
# 首遍 Paraformer (CPU) 性能压测套件 —— 编排脚本
#
# 基于 config/config.default.yaml 的配置, 分两层单独量化首遍模型:
#   1. 引擎级 (bench_paraformer.py): 绕过网关, 直测 Paraformer CPU 推理;
#      多进程 (每进程独立加载模型) 从 1 个逐步加到多个, 验证吞吐是否线性扩展。
#   2. 网关级 (bench_gateway.py): 真实 WS 链路 (VAD+首遍+网关), N 个网关进程
#      客户端轮询负载均衡, 从单进程到多进程验证端到端承载线性度。
#      (二遍用 mock 立即返回, 排除远程 GPU 干扰, 实现首遍隔离)
#
# 用法:
#   bash benchmark/run_paraformer_suite.sh                    # 全量 (引擎+网关)
#   SKIP_GATEWAY=1 bash benchmark/run_paraformer_suite.sh     # 只跑引擎级
#   SKIP_ENGINE=1  bash benchmark/run_paraformer_suite.sh     # 只跑网关级
#
# 可调参数 (环境变量覆盖, 见下文默认值):
#   CONFIG / AUDIO / THREADS / PROCS_LIST / STREAMS_LIST / PACING_LIST
#   GATEWAY_PROCS / GATEWAY_SESSIONS / GW_THREADS / MAX_END_SILENCE
#   STREAM_SEC / CHUNK_MS / BASE_WS_PORT / BASE_HTTP_PORT
#
# 产物: benchmark/results/<时间戳>_paraformer_suite/
#   ├── engine/    paraformer_engine.csv + paraformer_scaling.csv + 日志
#   ├── gateway_N/ gateway_ws.csv + 日志 + 各实例服务日志 (gw_i.log)
#   ├── gateway_all.csv  (跨 N 拼接, 便于画对比图)
#   └── suite.log
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
[[ -n "$PY" ]] || { echo "[ERROR] 未找到 Python 解释器"; exit 2; }

# ----------------------------- 参数 (环境变量覆盖) -----------------------------
CONFIG="${CONFIG:-$ROOT/config/config.default.yaml}"
AUDIO="${AUDIO:-$ROOT/fixtures/sample_16k.wav}"

# 引擎级矩阵
PACING_LIST="${PACING_LIST:-offline,realtime}"   # offline=纯吞吐, realtime=实时节奏
PROCS_LIST="${PROCS_LIST:-1,2,4,6,12}"           # 进程数 (每进程独立加载模型)
STREAMS_LIST="${STREAMS_LIST:-1,4}"              # 每进程并发流数
THREADS="${THREADS:-2}"                          # 每进程 torch 线程数 (固定值保证线性对比公平;
                                                 # 0=auto 按核数均分, 会引入线程数差异的干扰)

# 网关级矩阵
GATEWAY_PROCS="${GATEWAY_PROCS:-1,2,4}"          # 网关实例数
GATEWAY_SESSIONS="${GATEWAY_SESSIONS:-1,4,8,16}" # 并发会话数 (跨实例轮询分发)
GW_THREADS="${GW_THREADS:-0}"                    # 每实例 OMP 线程 (0=auto: 核数/实例数)
MAX_END_SILENCE="${MAX_END_SILENCE:-480}"        # 句间停顿切句阈值设置
STREAM_SEC="${STREAM_SEC:-30}"                   # 每路流/会话音频时长(s)
CHUNK_MS="${CHUNK_MS:-60}"
BASE_WS_PORT="${BASE_WS_PORT:-21000}"
BASE_HTTP_PORT="${BASE_HTTP_PORT:-21100}"

# ----------------------------- 运行目录 -----------------------------
TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/benchmark/results/${TAG}_paraformer_suite"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/suite.log") 2>&1

GW_PIDS=()
cleanup() {
    if [[ ${#GW_PIDS[@]} -gt 0 ]]; then
        echo "==> 停止 ${#GW_PIDS[@]} 个网关进程 ..."
        kill "${GW_PIDS[@]}" 2>/dev/null || true
        wait "${GW_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

concat_csv() { # concat_csv <输出> <输入1> [输入2 ...]
    local out="$1"; shift
    : > "$out"
    local f first=1
    for f in "$@"; do
        [[ -f "$f" ]] || continue
        if [[ $first -eq 1 ]]; then cat "$f" >> "$out"; first=0
        else tail -n +2 "$f" >> "$out"; fi
    done
}

echo "=============================================================="
echo " 首遍 Paraformer CPU 压测套件"
echo " 配置: $CONFIG | 音频: $AUDIO"
echo " 引擎级: pacing=$PACING_LIST procs=$PROCS_LIST streams=$STREAMS_LIST threads=$THREADS"
echo " 网关级: 实例=$GATEWAY_PROCS 会话=$GATEWAY_SESSIONS 每实例线程=$GW_THREADS"
echo " 输出:   $RUN_ROOT"
echo "=============================================================="
"$PY" - <<EOF
import json, sys
sys.path.insert(0, "$ROOT")
from benchmark.bench_common import host_info
info = {"suite": "paraformer", "started": "$TAG", "host": host_info(),
        "engine_matrix": {"pacing": "$PACING_LIST".split(","), "procs": "$PROCS_LIST".split(","),
                          "streams": "$STREAMS_LIST".split(","), "threads": $THREADS},
        "gateway_matrix": {"procs": "$GATEWAY_PROCS".split(","), "sessions": "$GATEWAY_SESSIONS".split(","),
                           "gw_threads": $GW_THREADS, "max_end_silence": $MAX_END_SILENCE}}
with open("$RUN_ROOT/manifest.json", "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
EOF

# =============================================================================
# 1) 引擎级: 纯 Paraformer CPU 推理, 多进程扩展性
# =============================================================================
if [[ "${SKIP_ENGINE:-0}" != "1" ]]; then
    echo ""
    echo "==> [1/2] 引擎级压测 (离线吞吐 + 实时节奏 x 多进程扩展) ..."
    "$PY" benchmark/bench_paraformer.py \
        --config "$CONFIG" \
        --audio "$AUDIO" \
        --pacing "$PACING_LIST" \
        --procs "$PROCS_LIST" \
        --streams "$STREAMS_LIST" \
        --threads "$THREADS" \
        --chunk_ms "$CHUNK_MS" \
        --stream_sec "$STREAM_SEC" \
        --out_dir "$RUN_ROOT/engine" \
        --save_details
fi

# =============================================================================
# 2) 网关级: N 个网关实例 (final=mock 隔离首遍), 客户端轮询负载均衡
# =============================================================================
if [[ "${SKIP_GATEWAY:-0}" != "1" ]]; then
    echo ""
    echo "==> [2/2] 网关级压测 (final_asr=mock 隔离首遍; VAD/说话人 funasr, 首遍 onnx) ..."

    # 2.1 派生网关配置: 基于 config.default.yaml, 仅调 VAD 静音阈值适配压测音频
    GW_CONFIG="$RUN_ROOT/gateway_config.yaml"
    sed -e "s/max_end_silence_time: .*/max_end_silence_time: $MAX_END_SILENCE/" "$CONFIG" > "$GW_CONFIG"
    echo "    派生配置: $GW_CONFIG (max_end_silence_time=$MAX_END_SILENCE)"

    NPROC_TOTAL="$(nproc)"
    GW_CSV_LIST=()
    IFS=',' read -ra N_LIST <<< "$GATEWAY_PROCS"
    for N in "${N_LIST[@]}"; do
        SUB_DIR="$RUN_ROOT/gateway_${N}inst"
        mkdir -p "$SUB_DIR"
        URIS=""

        # 2.2 每实例线程配额: 0=按核均分 (1 实例吃满整机, N 实例各得 1/N)
        if [[ "$GW_THREADS" -gt 0 ]]; then
            TPT="$GW_THREADS"
        else
            TPT=$(( NPROC_TOTAL / N )); [[ $TPT -lt 1 ]] && TPT=1
        fi

        # 2.3 启动 N 个网关实例 (每进程独立加载模型; 关闭 Nacos 注册; final=mock)
        echo "---- 启动 $N 个网关实例 (每实例 OMP 线程=$TPT, 模型加载需数十秒) ..."
        for i in $(seq 1 "$N"); do
            WS_PORT=$((BASE_WS_PORT + i))
            HTTP_PORT=$((BASE_HTTP_PORT + i))
            if ss -ltn "sport = :$WS_PORT" 2>/dev/null | grep -q LISTEN; then
                echo "[ERROR] 端口 $WS_PORT 已被占用, 请换 BASE_WS_PORT"; exit 3
            fi
            OMP_NUM_THREADS=$TPT MKL_NUM_THREADS=$TPT \
            ASR_CONFIG_PATH="$GW_CONFIG" \
            FINAL_ASR_BACKEND=mock \
            NACOS_ENABLE=false \
            LOG_FILE="$SUB_DIR/gw_${i}.log" \
            "$PY" main.py \
                --device cpu \
                --port "$WS_PORT" --http_port "$HTTP_PORT" \
                --streaming_backend onnx --vad_backend funasr \
                --speaker_backend funasr --final_backend mock \
                > "$SUB_DIR/gw_${i}.stdout.log" 2>&1 &
            GW_PIDS+=($!)
            URIS="${URIS}ws://127.0.0.1:${WS_PORT},"
            echo "    实例 $i: ws=$WS_PORT http=$HTTP_PORT pid=${GW_PIDS[-1]}"
        done
        URIS="${URIS%,}"

        # 2.4 等待全部实例就绪 (/ready 在模型加载完成后才可用)
        echo "---- 等待 $N 个实例就绪 ..."
        for i in $(seq 1 "$N"); do
            HTTP_PORT=$((BASE_HTTP_PORT + i))
            READY=0
            for _ in $(seq 1 300); do
                if curl -sf "http://127.0.0.1:$HTTP_PORT/ready" >/dev/null 2>&1; then READY=1; break; fi
                sleep 2
            done
            if [[ "$READY" != "1" ]]; then
                echo "[ERROR] 实例 $i 未就绪, 日志: $SUB_DIR/gw_${i}.stdout.log"; exit 3
            fi
        done
        echo "    全部就绪."

        # 2.5 压测: 并发会话轮询分发到 N 个实例
        "$PY" benchmark/bench_gateway.py \
            --uris "$URIS" \
            --sessions "$GATEWAY_SESSIONS" \
            --audio "$AUDIO" \
            --stream_sec "$STREAM_SEC" \
            --chunk_ms "$CHUNK_MS" \
            --rate 1.0 \
            --out_dir "$SUB_DIR"
        # bench_gateway 会在 out_dir 下再建时间戳子目录, 取实际 CSV 路径
        GW_CSV_LIST+=("$(find "$SUB_DIR" -name gateway_ws.csv | head -1)")

        # 2.6 停掉本组实例, 释放 CPU 后再测下一组
        echo "---- 停止 $N 个实例, 冷却 10s ..."
        kill "${GW_PIDS[@]}" 2>/dev/null || true
        wait "${GW_PIDS[@]}" 2>/dev/null || true
        GW_PIDS=()
        sleep 10
    done

    # 2.7 跨实例数拼接汇总 (同一表头)
    if [[ ${#GW_CSV_LIST[@]} -gt 0 ]]; then
        concat_csv "$RUN_ROOT/gateway_all.csv" "${GW_CSV_LIST[@]}"
        echo "==> 网关级汇总: $RUN_ROOT/gateway_all.csv"
    fi
fi

echo ""
echo "=============================================================="
echo " 套件完成. 产物目录:"
find "$RUN_ROOT" -maxdepth 2 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.log" \) | sort
echo "=============================================================="
