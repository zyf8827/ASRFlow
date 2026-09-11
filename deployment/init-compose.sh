#!/usr/bin/env bash
# 交互式生成 deployment/docker-compose.yaml + .env
# 变量说明见 deployment/.env.example
#
# 用法:
#   bash deployment/init-compose.sh
#   cd deployment && docker compose up -d
#
# 回车 = 使用括号内默认值. 生成文件不入库 (见 .gitignore).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/deployment"
ENV_FILE="$DEST/.env"
COMPOSE_FILE="$DEST/docker-compose.yaml"
SINGLE_EXAMPLE="$DEST/docker-compose.single.example.yaml"
MULTI_EXAMPLE="$DEST/docker-compose.multi.example.yaml"

if [[ ! -t 0 ]]; then
    echo "[ERROR] 请在终端交互运行: bash deployment/init-compose.sh" >&2
    exit 2
fi
[[ -f "$SINGLE_EXAMPLE" && -f "$MULTI_EXAMPLE" ]] || {
    echo "[ERROR] 找不到 example 模板: $SINGLE_EXAMPLE" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# 本机 CPU
# ---------------------------------------------------------------------------
LOGICAL="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
PHYSICAL=""
if command -v lscpu >/dev/null 2>&1; then
    PHYSICAL="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
fi
CPU_MODEL="$(awk -F: '/model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"

# ---------------------------------------------------------------------------
# 读已有 .env 作默认 (再次初始化时沿用)
# ---------------------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a
    # 去掉 export, 允许 KEY='val'
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

ask() {  # ask <varname> <prompt> <default>  自由输入, 回车=默认
    local _n="$1" _p="$2" _d="$3" _v
    if [[ -n "$_d" ]]; then
        read -r -p "$_p [$_d]: " _v || true
    else
        read -r -p "$_p: " _v || true
    fi
    if [[ -z "$_v" ]]; then
        printf -v "$_n" '%s' "$_d"
    else
        printf -v "$_n" '%s' "$_v"
    fi
}

# ask_choice <varname> <prompt> <default_index> <opt1> <opt2> ...
# 选项从 1 编号, 回车=默认. 写入选项原文.
ask_choice() {
    local __n="$1" __p="$2" __d="$3"
    shift 3
    local __opts=("$@")
    local __nopts=${#__opts[@]}
    local __v __i __k
    while :; do
        echo "$__p"
        for __i in "${!__opts[@]}"; do
            __k=$((__i + 1))
            if (( __k == __d )); then
                echo "  $__k) ${__opts[$__i]}  (默认)"
            else
                echo "  $__k) ${__opts[$__i]}"
            fi
        done
        read -r -p "请输入 1-$__nopts [默认 $__d]: " __v || true
        __v="${__v:-$__d}"
        if [[ "$__v" =~ ^[1-9][0-9]*$ ]] && (( __v >= 1 && __v <= __nopts )); then
            printf -v "$__n" '%s' "${__opts[$((__v - 1))]}"
            return 0
        fi
        echo "  无效, 请输入 1-$__nopts"
    done
}

ask_yn() {  # ask_yn <varname> <prompt> <default y|n>  大小写不敏感, 回车=默认
    local _n="$1" _p="$2" _d="$3" _v _hint
    if [[ "$_d" == "y" ]]; then _hint="Y/n"; else _hint="y/N"; fi
    while :; do
        read -r -p "$_p [$_hint]: " _v || true
        _v="$(echo "${_v:-$_d}" | tr '[:upper:]' '[:lower:]')"
        case "$_v" in
            y|yes) printf -v "$_n" 'y'; return 0 ;;
            n|no)  printf -v "$_n" 'n'; return 0 ;;
        esac
        echo "  请输入 y 或 n"
    done
}

detect_host_ip() {
    local ip
    ip="$(ip -4 route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
    [[ -n "$ip" ]] || ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "$ip" ]] || ip="127.0.0.1"
    printf '%s' "$ip"
}

suggest_layout() {
    local logical="$1"
    local reserve=0 cores=4 n=1
    if (( logical <= 4 )); then
        reserve=0; cores=$logical; n=1
    elif (( logical <= 8 )); then
        reserve=0; cores=4; n=1
    else
        reserve=$(( logical > 16 ? 4 : 2 ))
        cores=4
        n=$(( (logical - reserve) / cores ))
        (( n < 1 )) && n=1
        (( n > 8 )) && n=8
    fi
    printf '%s %s %s' "$n" "$cores" "$reserve"
}

env_quote() {
    local s="$1"
    s="${s//\'/\'\\\'}"
    printf "'%s'" "$s"
}

port_in_use() {  # <port>  返回 0=有监听; ss > netstat > /dev/tcp 逐级兜底
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}\$"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}\$"
    else
        (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
    fi
}

# ---------------------------------------------------------------------------
echo "======================================================================"
echo " ASRFlow Compose 初始化"
echo " 仓库: $ROOT"
echo " CPU:  ${LOGICAL} 逻辑核${PHYSICAL:+ / ${PHYSICAL} 物理核}${CPU_MODEL:+ / $CPU_MODEL}"
echo "======================================================================"
echo "选择题输入序号, 其它项回车即默认. 生成: $COMPOSE_FILE 与 $ENV_FILE"
echo

# ---- 1. 模式 ----
read -r SUG_N SUG_CORES SUG_RESERVE <<< "$(suggest_layout "$LOGICAL")"
DEF_MODE="${ASRFLOW_MODE:-}"
if [[ -z "$DEF_MODE" ]]; then
    if (( SUG_N > 1 )); then DEF_MODE="multi"; else DEF_MODE="single"; fi
fi
DEF_MODE_IDX=1
[[ "$DEF_MODE" == "multi" ]] && DEF_MODE_IDX=2
ask_choice MODE_LABEL "部署模式" "$DEF_MODE_IDX" "单实例" "多实例绑核"
if [[ "$MODE_LABEL" == "单实例" ]]; then MODE="single"; else MODE="multi"; fi

N_INST=1
CORES_PER=$LOGICAL
RESERVE=0
if [[ "$MODE" == "multi" ]]; then
    ask N_INST "实例数" "${ASRFLOW_N_INST:-$SUG_N}"
    ask CORES_PER "每实例核数 (同时作为 ORT/OMP 线程数)" "${ASRFLOW_CORES_PER:-$SUG_CORES}"
    ask RESERVE "预留核数 (从高位核留给 OS/同机 vLLM, 不分配给 ASR)" "${ASRFLOW_RESERVE:-$SUG_RESERVE}"
    if ! [[ "$N_INST" =~ ^[1-9][0-9]*$ && "$CORES_PER" =~ ^[1-9][0-9]*$ && "$RESERVE" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 实例数/核数必须是正整数, 预留核数必须是非负整数" >&2
        exit 2
    fi
    NEED=$(( N_INST * CORES_PER ))
    AVAIL=$(( LOGICAL - RESERVE ))
    if (( AVAIL < 1 )); then
        echo "[ERROR] 预留 $RESERVE 核后没有剩余 CPU (本机 $LOGICAL)" >&2
        exit 2
    fi
    if (( NEED > AVAIL )); then
        echo "[ERROR] 需要 ${N_INST}×${CORES_PER}=${NEED} 核, 可分配只有 ${AVAIL} (本机 $LOGICAL, 预留 $RESERVE)" >&2
        exit 2
    fi
    echo
    echo "  cpuset 分配 (核 0 起连续切, 高位 ${RESERVE} 核预留):"
    for i in $(seq 1 "$N_INST"); do
        s=$(( (i - 1) * CORES_PER ))
        e=$(( s + CORES_PER - 1 ))
        ws=$(( 10095 + (i - 1) * 10 ))
        echo "    asrflow-$i  cpuset=${s}-${e}  threads=$CORES_PER  ws=$ws  http=$((ws + 1))"
    done
    LEFT=$(( AVAIL - NEED ))
    echo "    未分配: $(( LEFT + RESERVE )) 核 (含预留 $RESERVE)"
    echo
else
    DEF_THREADS="${STREAMING_ONNX_THREADS:-0}"
    ask CORES_PER "单实例 ORT 线程数 (0=容器内全部核)" "$DEF_THREADS"
    if ! [[ "$CORES_PER" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 线程数必须是非负整数" >&2
        exit 2
    fi
    N_INST=1
    RESERVE=0
fi

# ---- 2. Nacos ----
if [[ "$MODE" == "multi" ]]; then DEF_NACOS="y"; else DEF_NACOS="n"; fi
[[ "${NACOS_ENABLE:-}" == "true" ]] && DEF_NACOS="y"
[[ "${NACOS_ENABLE:-}" == "false" ]] && DEF_NACOS="n"
ask_yn NACOS_ON "启用 Nacos 服务注册?" "$DEF_NACOS"
if [[ "$NACOS_ON" == "y" ]]; then
    ask NACOS_SERVER_ENDPOINT "  Nacos 地址 host:port" "${NACOS_SERVER_ENDPOINT:-127.0.0.1:8848}"
    ask NACOS_NAMESPACE "  命名空间" "${NACOS_NAMESPACE:-public}"
    ask NACOS_GROUP_NAME "  分组" "${NACOS_GROUP_NAME:-DEFAULT_GROUP}"
    ask NACOS_SERVICE_NAME "  服务名" "${NACOS_SERVICE_NAME:-asrflow}"
    ask NACOS_SERVICE_HOST "  注册 IP (不能是 0.0.0.0)" "${NACOS_SERVICE_HOST:-$(detect_host_ip)}"
    ask NACOS_USERNAME "  用户名" "${NACOS_USERNAME:-nacos}"
    ask NACOS_PASSWORD "  密码" "${NACOS_PASSWORD:-CHANGE_ME}"
    NACOS_ENABLE="true"
else
    NACOS_ENABLE="false"
    NACOS_SERVER_ENDPOINT="${NACOS_SERVER_ENDPOINT:-127.0.0.1:8848}"
    NACOS_NAMESPACE="${NACOS_NAMESPACE:-public}"
    NACOS_GROUP_NAME="${NACOS_GROUP_NAME:-DEFAULT_GROUP}"
    NACOS_SERVICE_NAME="${NACOS_SERVICE_NAME:-asrflow}"
    NACOS_SERVICE_HOST="${NACOS_SERVICE_HOST:-127.0.0.1}"
    NACOS_USERNAME="${NACOS_USERNAME:-nacos}"
    NACOS_PASSWORD="${NACOS_PASSWORD:-CHANGE_ME}"
fi

# ---- 3. vLLM ----
# 首遍固定 ONNX CPU; 镜像不分 cuda/ascend, amd64/arm64 各一份 Dockerfile
# (下方按宿主架构把 build 段指向 Dockerfile.arm64)
ASR_DEVICE="cpu"
ASRFLOW_VERSION="${ASRFLOW_VERSION:-1.0.0}"
if [[ "$MODE" == "multi" ]]; then
    # multi = host 网络, 容器内 127.0.0.1 即宿主
    DEF_VLLM="${VLLM_URL:-http://127.0.0.1:8899/v1/audio/transcriptions}"
else
    # single = bridge 网络, 容器内 127.0.0.1 指向容器自身, 经 host-gateway 别名访问宿主
    DEF_VLLM="${VLLM_URL:-http://host.docker.internal:8899/v1/audio/transcriptions}"
fi
ask VLLM_URL "二遍 vLLM 地址" "$DEF_VLLM"
ask FINAL_ASR_MODEL "vLLM 模型名 (须与 --served-model-name 一致)" "${FINAL_ASR_MODEL:-qwen3-asr-1.7b}"
DEF_SPEECH_SEC="30"
if [[ -n "${VAD_MAX_SPEECH_MS:-}" ]]; then
    DEF_SPEECH_SEC=$(( VAD_MAX_SPEECH_MS / 1000 ))
fi
ask SPEECH_SEC "最长单句时长秒 (VAD 强制切分)" "$DEF_SPEECH_SEC"
if ! [[ "$SPEECH_SEC" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 最长单句时长必须是非负整数秒" >&2
    exit 2
fi
VAD_MAX_SPEECH_MS=$(( SPEECH_SEC * 1000 ))
# config.default.yaml ring_buffer=32s, pre_roll=800ms
if (( VAD_MAX_SPEECH_MS + 800 > 32000 )); then
    echo "[WARN] ${SPEECH_SEC}s 超过默认环形缓冲 32s, 启动校验会失败; 请同步加大 audio.ring_buffer_duration_sec" >&2
fi

# ---- 4. 可选 ----
# 宿主日志基目录: compose 里以 ${ASRFLOW_LOG_BASE:-默认} 插值, 改 .env 即换路径,
# 不必重新生成 compose. 多实例按 基目录/instance-N 子目录区分, 互不冲突.
ask LOG_BASE "宿主日志基目录 (单实例=基目录本身, 多实例=基目录/instance-N)" "${ASRFLOW_LOG_BASE:-/var/log/asrflow}"
if [[ ! "$LOG_BASE" == /* ]]; then
    echo "[ERROR] 日志基目录必须是绝对路径: $LOG_BASE" >&2
    exit 2
fi
# 源码热替换: -v <仓库根>:/app. 日志挂载 /app/logs 是嵌套挂载, 优先级高于 ..:/app,
# 多实例仍各自写 基目录/instance-N, 不冲突. 离线目标机器没有源码目录, 默认 n.
DEF_MOUNT="n"
[[ "${ASRFLOW_MOUNT_SOURCE:-}" == "true" ]] && DEF_MOUNT="y"
ask_yn MOUNT_ON "源码热替换 (把仓库根目录挂载到容器 /app, 改代码 restart 即生效, 不需重建镜像)?" "$DEF_MOUNT"
if [[ "$MOUNT_ON" == "y" ]]; then MOUNT_SOURCE="true"; else MOUNT_SOURCE="false"; fi
# 容器内存上限: single/multi compose 均以 mem_limit: ${ASRFLOW_MEM_LIMIT:-2g} 插值,
# 之后改 .env 即可, 不必重新生成
ask MEM "容器内存上限 (docker mem_limit)" "${ASRFLOW_MEM_LIMIT:-2g}"
ask_yn MORE "配置更多项 (端口/合批/量化/日志)?" "n"
WS_PORT="${ASR_SERVER_PORT:-10095}"
HTTP_PORT="${ASR_HTTP_PORT:-10096}"
BATCH="${STREAMING_BATCH_SIZE:-8}"
WINDOW="${STREAMING_BATCH_WINDOW_MS:-15}"
QUANT="${STREAMING_ONNX_QUANT:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
if [[ "$MORE" == "y" ]]; then
    ask WS_PORT "  起始 WS 端口 (多实例每实例 +10)" "$WS_PORT"
    ask HTTP_PORT "  起始 HTTP 端口 (单实例; 多实例=WS+1)" "$HTTP_PORT"
    ask BATCH "  合批最大流数 STREAMING_BATCH_SIZE" "$BATCH"
    ask WINDOW "  攒批窗口 ms" "$WINDOW"
    DEF_Q_IDX=1
    [[ "$QUANT" == "false" ]] && DEF_Q_IDX=2
    ask_choice QUANT_LABEL "  ONNX 精度" "$DEF_Q_IDX" "int8 量化" "fp32"
    if [[ "$QUANT_LABEL" == "fp32" ]]; then QUANT="false"; else QUANT="true"; fi
    DEF_L_IDX=2
    case "$LOG_LEVEL" in
        DEBUG) DEF_L_IDX=1 ;;
        WARNING) DEF_L_IDX=3 ;;
        ERROR) DEF_L_IDX=4 ;;
        *) DEF_L_IDX=2 ;;
    esac
    ask_choice LOG_LEVEL "  日志级别" "$DEF_L_IDX" "DEBUG" "INFO" "WARNING" "ERROR"
fi

STREAMING_ASR_BACKEND="${STREAMING_ASR_BACKEND:-onnx}"
VAD_BACKEND="${VAD_BACKEND:-funasr}"
SPEAKER_BACKEND="${SPEAKER_BACKEND:-funasr}"
FINAL_ASR_BACKEND="${FINAL_ASR_BACKEND:-vllm_http}"

# ---- 确认 ----
echo
echo "----------------------------------------------------------------------"
echo " 模式:     $MODE"
if [[ "$MODE" == "multi" ]]; then
    echo " 实例:     $N_INST × ${CORES_PER} 核  (预留 $RESERVE, 本机 $LOGICAL)"
else
    echo " ORT 线程: $CORES_PER  (0=全部核)"
fi
echo " Nacos:    $NACOS_ENABLE${NACOS_ENABLE:+  $NACOS_SERVER_ENDPOINT  user=${NACOS_USERNAME:-<空>}}"
echo " vLLM:     $VLLM_URL  model=$FINAL_ASR_MODEL"
if [[ "$MODE" == "multi" ]]; then
    echo " 日志:     $LOG_BASE/instance-1..instance-$N_INST"
else
    echo " 日志:     $LOG_BASE"
fi
echo " 最长单句: ${SPEECH_SEC}s"
echo " 首遍:     ONNX CPU  asrflow:${ASRFLOW_VERSION}"
if [[ "$MOUNT_ON" == "y" ]]; then
    echo " 热替换:   ..:/app  (日志仍按实例分开: /app/logs 嵌套挂载优先)"
else
    echo " 热替换:   关闭"
fi
echo " 合批:     batch=$BATCH window=${WINDOW}ms quant=$QUANT  mem=$MEM"
echo "----------------------------------------------------------------------"

# ---- 端口占用检查: 仅对被占用的端口提示 (未占用不输出), multi 按实例逐个判断 ----
port_conflict=0
port_warn() {  # <实例> <协议> <端口>
    if port_in_use "$3"; then
        port_conflict=1
        echo " [WARN] $1 $2 端口 $3 已被占用 (运行中的旧实例或其它服务?)"
    fi
}
if [[ "$MODE" == "multi" ]]; then
    for i in $(seq 1 "$N_INST"); do
        ws=$(( WS_PORT + (i - 1) * 10 ))
        port_warn "asrflow-$i" ws "$ws"
        port_warn "asrflow-$i" http "$(( ws + 1 ))"
    done
else
    port_warn asrflow ws "$WS_PORT"
    port_warn asrflow http "$HTTP_PORT"
fi
if [[ "$port_conflict" == "1" ]]; then
    echo " [WARN] 以上实例 up 时会端口绑定失败, 请先停掉占用方或重新 init 换端口"
fi

ask_yn OK "写入 $COMPOSE_FILE 与 $ENV_FILE ?" "y"
[[ "$OK" == "y" ]] || { echo "已取消"; exit 0; }

if [[ -f "$COMPOSE_FILE" ]]; then
    ask_yn OVER "已存在 docker-compose.yaml, 覆盖?" "y"
    [[ "$OVER" == "y" ]] || { echo "已取消"; exit 0; }
fi

# 预创建宿主日志目录, 归当前用户所有 (失败仅告警; 缺失时 dockerd 会以 root 自动创建)
LOG_DIRS=("$LOG_BASE")
if [[ "$MODE" == "multi" ]]; then
    for i in $(seq 1 "$N_INST"); do LOG_DIRS+=("$LOG_BASE/instance-$i"); done
fi
for d in "${LOG_DIRS[@]}"; do
    mkdir -p "$d" 2>/dev/null || echo "[WARN] 无法创建日志目录 $d, up 时由 dockerd 以 root 创建" >&2
done

# ---------------------------------------------------------------------------
# 写 .env
# ---------------------------------------------------------------------------
{
    echo "# generated by deployment/init-compose.sh $(date '+%F %T')"
    echo "ASRFLOW_MODE=$(env_quote "$MODE")"
    echo "ASRFLOW_N_INST=$(env_quote "$N_INST")"
    echo "ASRFLOW_CORES_PER=$(env_quote "$CORES_PER")"
    echo "ASRFLOW_RESERVE=$(env_quote "$RESERVE")"
    echo "ASRFLOW_VERSION=$(env_quote "$ASRFLOW_VERSION")"
    echo "ASRFLOW_MEM_LIMIT=$(env_quote "$MEM")"
    echo "ASRFLOW_MOUNT_SOURCE=$(env_quote "$MOUNT_SOURCE")"
    echo "ASRFLOW_LOG_BASE=$(env_quote "$LOG_BASE")"
    echo "ASR_DEVICE=$(env_quote "$ASR_DEVICE")"
    echo "ASR_SERVER_PORT=$(env_quote "$WS_PORT")"
    echo "ASR_HTTP_PORT=$(env_quote "$HTTP_PORT")"
    echo "STREAMING_ASR_BACKEND=$(env_quote "$STREAMING_ASR_BACKEND")"
    echo "VAD_BACKEND=$(env_quote "$VAD_BACKEND")"
    echo "SPEAKER_BACKEND=$(env_quote "$SPEAKER_BACKEND")"
    echo "FINAL_ASR_BACKEND=$(env_quote "$FINAL_ASR_BACKEND")"
    echo "STREAMING_BATCH_SIZE=$(env_quote "$BATCH")"
    echo "STREAMING_BATCH_WINDOW_MS=$(env_quote "$WINDOW")"
    echo "STREAMING_ONNX_QUANT=$(env_quote "$QUANT")"
    if [[ "$CORES_PER" == "0" ]]; then
        echo "STREAMING_ONNX_THREADS=''"
    else
        echo "STREAMING_ONNX_THREADS=$(env_quote "$CORES_PER")"
        echo "OMP_NUM_THREADS=$(env_quote "$CORES_PER")"
        echo "MKL_NUM_THREADS=$(env_quote "$CORES_PER")"
    fi
    echo "VLLM_URL=$(env_quote "$VLLM_URL")"
    echo "FINAL_ASR_MODEL=$(env_quote "$FINAL_ASR_MODEL")"
    echo "VAD_MAX_SPEECH_MS=$(env_quote "$VAD_MAX_SPEECH_MS")"
    echo "LOG_LEVEL=$(env_quote "$LOG_LEVEL")"
    echo "NACOS_ENABLE=$(env_quote "$NACOS_ENABLE")"
    echo "NACOS_SERVER_ENDPOINT=$(env_quote "$NACOS_SERVER_ENDPOINT")"
    echo "NACOS_SERVICE_NAME=$(env_quote "$NACOS_SERVICE_NAME")"
    echo "NACOS_SERVICE_HOST=$(env_quote "$NACOS_SERVICE_HOST")"
    echo "NACOS_NAMESPACE=$(env_quote "$NACOS_NAMESPACE")"
    echo "NACOS_GROUP_NAME=$(env_quote "$NACOS_GROUP_NAME")"
    echo "NACOS_USERNAME=$(env_quote "$NACOS_USERNAME")"
    echo "NACOS_PASSWORD=$(env_quote "$NACOS_PASSWORD")"
    echo "NACOS_HEARTBEAT_SEC='5'"
    if [[ "$MODE" == "multi" ]]; then
        for i in $(seq 1 "$N_INST"); do
            s=$(( (i - 1) * CORES_PER ))
            e=$(( s + CORES_PER - 1 ))
            echo "ASRFLOW_${i}_CPUSET=$(env_quote "${s}-${e}")"
            echo "ASRFLOW_${i}_ONNX_THREADS=$(env_quote "$CORES_PER")"
        done
    fi
} > "$ENV_FILE"

# ---------------------------------------------------------------------------
# 写 compose
# ---------------------------------------------------------------------------
GEN_HDR="# Generated by deployment/init-compose.sh $(date '+%F %T')
# mode=$MODE n=$N_INST cores=$CORES_PER reserve=$RESERVE host_cpus=$LOGICAL source_mount=$MOUNT_SOURCE log_base=$LOG_BASE
# 重新运行 init-compose.sh 会覆盖本文件. 模板见 docker-compose.${MODE}.example.yaml
"

if [[ "$MODE" == "single" ]]; then
    {
        echo "$GEN_HDR"
        # 丢掉模板第一行标题, 保留正文; 热替换启用时取消注释模板里的 - ..:/app 行
        # (并去掉"保持注释状态"那句, 已不再成立)
        tail -n +2 "$SINGLE_EXAMPLE" | sed 's|^\([[:space:]]*\)# - \.\.:/app$|\1- ..:/app|'
        if [[ "$MOUNT_ON" == "y" ]]; then
            sed -i '/^[[:space:]]*# 离线目标机器没有源码目录/d' "$COMPOSE_FILE"
        fi
    } > "$COMPOSE_FILE"
else
    {
        echo "$GEN_HDR"
        cat <<'YAML'
x-asrflow-common: &asrflow-base
  build:
    context: ..
    dockerfile: deployment/Dockerfile
  image: asrflow:${ASRFLOW_VERSION:-1.0.0}
  restart: always
  # 健康检查: /ready 聚合引擎自愈状态(重载循环中 503 摘流)。镜像必带 python3。
  # 注意: 原生 docker compose 只标记 unhealthy 不自动重启。
  healthcheck:
    test: ["CMD", "python3", "-c", "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('ASR_HTTP_PORT','10096')+'/ready', timeout=3).status==200 else 1)"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 180s
  network_mode: host
  mem_limit: ${ASRFLOW_MEM_LIMIT:-2g}
  # 容器 stdout/stderr (json-file) 轮转: 不配置时 docker 默认不封顶, 会打爆宿主磁盘。
  # json-file 只能按大小轮转 (无按天模式): 默认 100m×15, 常规流量下保留 ≥15 天
  logging:
    driver: json-file
    options:
      max-size: "${DOCKER_LOG_MAX_SIZE:-100m}"
      max-file: "${DOCKER_LOG_MAX_FILE:-15}"
  environment: &asrflow-env
    TZ: Asia/Shanghai                        # 容器时区: 东八区, 日志时间戳与宿主一致
    ASR_SERVER_HOST: 0.0.0.0
    ASR_DEVICE: ${ASR_DEVICE:-cpu}
    RESUME_TTL_SEC: ${RESUME_TTL_SEC:-30}
    IDLE_TIMEOUT_SEC: ${IDLE_TIMEOUT_SEC:-300}
    ASR_MAX_FRAME_MS: ${ASR_MAX_FRAME_MS:-2000}
    MAX_CONNECTIONS: ${MAX_CONNECTIONS:-100}
    ASR_WORKER_THREADS: ${ASR_WORKER_THREADS:-0}
    ADMISSION_ENABLE: ${ADMISSION_ENABLE:-true}
    ADMISSION_STREAMING_WAIT_MS: ${ADMISSION_STREAMING_WAIT_MS:-200}
    ADMISSION_VAD_WAIT_MS: ${ADMISSION_VAD_WAIT_MS:-150}
    ADMISSION_FINAL_FALLBACK_RATE: ${ADMISSION_FINAL_FALLBACK_RATE:-0.5}
    STREAMING_ASR_BACKEND: ${STREAMING_ASR_BACKEND:-onnx}
    VAD_BACKEND: ${VAD_BACKEND:-funasr}
    SPEAKER_BACKEND: ${SPEAKER_BACKEND:-funasr}
    FINAL_ASR_BACKEND: ${FINAL_ASR_BACKEND:-vllm_http}
    VAD_MAX_SPEECH_MS: ${VAD_MAX_SPEECH_MS:-30000}
    STREAMING_BATCH_SIZE: ${STREAMING_BATCH_SIZE:-8}
    STREAMING_BATCH_WINDOW_MS: ${STREAMING_BATCH_WINDOW_MS:-15}
    STREAMING_ONNX_QUANT: ${STREAMING_ONNX_QUANT:-}
    VLLM_URL: ${VLLM_URL}
    FINAL_ASR_MODEL: ${FINAL_ASR_MODEL:-qwen3-asr-1.7b}
    FINAL_TIMEOUT_SEC: ${FINAL_TIMEOUT_SEC:-8}
    FINAL_MAX_CONCURRENCY: ${FINAL_MAX_CONCURRENCY:-32}
    FINAL_MAX_TOKENS: ${FINAL_MAX_TOKENS:-512}
    FINAL_CONTEXT_HISTORY_CHARS: ${FINAL_CONTEXT_HISTORY_CHARS:-200}
    LOG_LEVEL: ${LOG_LEVEL:-INFO}
    LOG_DIR: logs
    LOG_ROTATION: ${LOG_ROTATION:-00:00}
    LOG_RETENTION: ${LOG_RETENTION:-30 days}
    NACOS_ENABLE: ${NACOS_ENABLE:-false}
    NACOS_SERVER_ENDPOINT: ${NACOS_SERVER_ENDPOINT:-127.0.0.1:8848}
    NACOS_SERVICE_NAME: ${NACOS_SERVICE_NAME:-asrflow}
    NACOS_SERVICE_HOST: ${NACOS_SERVICE_HOST:-127.0.0.1}
    NACOS_NAMESPACE: ${NACOS_NAMESPACE:-public}
    NACOS_GROUP_NAME: ${NACOS_GROUP_NAME:-DEFAULT_GROUP}
    NACOS_USERNAME: ${NACOS_USERNAME:-nacos}
    NACOS_PASSWORD: ${NACOS_PASSWORD:-CHANGE_ME}
    NACOS_HEARTBEAT_SEC: ${NACOS_HEARTBEAT_SEC:-5}

services:
YAML
        for i in $(seq 1 "$N_INST"); do
            s=$(( (i - 1) * CORES_PER ))
            e=$(( s + CORES_PER - 1 ))
            ws=$(( WS_PORT + (i - 1) * 10 ))
            http=$(( ws + 1 ))
            # 热替换挂载行: 空 = 不挂; 启用时放在日志挂载之后, /app/logs 嵌套挂载
            # 优先于 ..:/app, 各实例日志仍落到宿主 $LOG_BASE/instance-N, 互不冲突
            SRC_MOUNT_LINE=""
            if [[ "$MOUNT_ON" == "y" ]]; then
                SRC_MOUNT_LINE=$'\n      # 源码热替换: /app/logs 为嵌套挂载, 优先于 ..:/app, 日志仍按实例分开'
                SRC_MOUNT_LINE+=$'\n      - ..:/app'
            fi
            cat <<YAML
  asrflow-$i:
    <<: *asrflow-base
    container_name: asrflow-$i
    cpuset: "${s}-${e}"
    environment:
      <<: *asrflow-env
      ASR_SERVER_PORT: $ws
      ASR_HTTP_PORT: $http
      STREAMING_ONNX_THREADS: "$CORES_PER"
      OMP_NUM_THREADS: "$CORES_PER"
      MKL_NUM_THREADS: "$CORES_PER"
    volumes:
      # \$ 转义: 保留给 compose 在 up 时按 .env 插值, 改 ASRFLOW_LOG_BASE 不必重新生成
      - \${ASRFLOW_LOG_BASE:-/var/log/asrflow}/instance-$i:/app/logs$SRC_MOUNT_LINE

YAML
        done
    } > "$COMPOSE_FILE"
fi

# arm64 宿主: compose build 段换用 arm64 专用 Dockerfile (amd64 版钉 +cpu 轮,
# aarch64 上 torchaudio 无 +cpu 轮可装, 直接 build 会失败); amd64 宿主保持默认.
if [[ "$(uname -m)" == "aarch64" ]]; then
    sed -i 's|dockerfile: deployment/Dockerfile$|dockerfile: deployment/Dockerfile.arm64|' "$COMPOSE_FILE"
fi

echo
echo "已写入:"
echo "  $ENV_FILE"
echo "  $COMPOSE_FILE"
echo
echo "下一步:"
echo "  bash deployment/build.sh ${ASRFLOW_VERSION}   # 若尚未构建镜像"
echo "  cd deployment && docker compose up -d"
if [[ "$MOUNT_ON" == "y" ]]; then
    echo "  热替换: 改宿主机代码后 docker compose restart 即生效, 无需重建镜像"
fi
echo "  docker compose -f $COMPOSE_FILE ps"
if [[ "$MODE" == "single" ]]; then
    echo "  curl -sf http://127.0.0.1:${HTTP_PORT}/ready"
else
    echo "  curl -sf http://127.0.0.1:$((WS_PORT + 1))/ready"
fi
