#!/bin/bash
set -e

# ==============================================================================
# Build and Package Script for ASRFlow Service (offline-ready image)
#
# 构建含全部模型 (除 vLLM/Qwen3-ASR 二遍外) 的离线镜像: 首遍 ONNX 合批导出物、
# FSMN-VAD、说话人 CampPlus、可选实时标点。
#
# 模型目录 models/ 不入 git, 构建前需**人工**将各模型目录完整拷贝进去
# (或事先手动运行 scripts/prepare_models.sh / scripts/export_onnx.py 生成,
# 两者效果等同人工拷贝)。本脚本只做两件事: 逐文件校验完整性 -> docker build,
# 校验不通过即中止并列出缺失文件与补齐方法。
#
# models/ 期望布局 (除 punc 可选外全部必需):
#   models/onnx/                                            首遍 ONNX 合批导出物
#     model_quant.onnx decoder_quant.onnx                   int8 量化图 (默认运行)
#     model.onnx decoder.onnx                               fp32 图 (STREAMING_ONNX_QUANT=false 时)
#     config.yaml am.mvn tokens.json
#   models/speech_fsmn_vad_zh-cn-16k-common-pytorch/
#     model.pt config.yaml                                  VAD
#   models/speech_campplus_sv_zh-cn_16k-common/
#     campplus_cn_common.bin                                说话人
#   models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/
#     model.pt                                              实时标点 (可选, 默认不启用)
#
# 注意: 流式 Paraformer 的 torch 原始模型 (speech_paraformer*...-online) 只是
# ONNX 导出的**输入**, 在构建机上供 scripts/export_onnx.py 使用, 服务运行与
# 镜像打包均不需要 (已被 .dockerignore 排除出构建上下文)。
#
# 生成辅助脚本 (手动执行, 代替人工拷贝; 产物同样是上面的布局):
#   bash scripts/prepare_models.sh            # VAD + 说话人 (并导出 paraformer 供 onnx 导出用)
#   .venv/bin/python scripts/export_onnx.py   # 首遍 ONNX 导出物 -> models/onnx/
#
# 镜像命名: asrflow:<版本号>, 如 asrflow:1.0.0 (amd64/arm64 同名).
# 两份 Dockerfile 按 CPU 架构分开维护 (差异原因见各自文件头注释), 本脚本按
# 构建机架构自动选择; PLATFORM 交叉构建时以 PLATFORM 为准:
#   linux/amd64 : deployment/Dockerfile        (torch 钉 +cpu 轮)
#   linux/arm64 : deployment/Dockerfile.arm64  (官方 aarch64 轮即 CPU 版)
#
# 二遍 Qwen3-ASR 的 vLLM 镜像是另一份独立镜像, 见 build-vllm.sh。
#
# Usage:
#   bash deployment/build.sh                     # asrflow:1.0.0 (模型须已拷入 models/)
#   bash deployment/build.sh 1.1.0               # asrflow:1.1.0
#   PLATFORM=linux/arm64 bash deployment/build.sh  # 交叉构建 arm64 (自动选用 Dockerfile.arm64)
#   CHECK_ONLY=1 bash deployment/build.sh        # 只做模型校验, 不构建
#
# Args / Env:
#   $1 / IMAGE_TAG     版本号 (默认 1.0.0)
#   PLATFORM           可选, 传给 docker build --platform (如 linux/amd64)
#   CHECK_ONLY=1       只校验模型完整性, 不执行 docker build
#   MODELS_DIR=models  模型暂存目录 (一般不改)
# ==============================================================================

IMAGE_NAME="asrflow"
IMAGE_TAG="${1:-${IMAGE_TAG:-1.0.0}}"
MODELS_DIR="${MODELS_DIR:-models}"

# Dockerfile 按目标架构选择: PLATFORM 交叉构建时以 PLATFORM 为准, 否则随构建机
if [[ -n "${PLATFORM:-}" ]]; then
    BUILD_ARCH="${PLATFORM#linux/}"   # linux/arm64 -> arm64
    BUILD_ARCH="${BUILD_ARCH%%/*}"    # linux/arm64/v8 -> arm64
else
    case "$(uname -m)" in
        aarch64|arm64) BUILD_ARCH="arm64" ;;
        *)             BUILD_ARCH="amd64" ;;
    esac
fi
case "$BUILD_ARCH" in
    amd64) DOCKERFILE="deployment/Dockerfile" ;;
    arm64) DOCKERFILE="deployment/Dockerfile.arm64" ;;
    *)
        echo "[ERROR] 不支持的目标架构: ${PLATFORM:-$(uname -m)} (目前支持 linux/amd64 | linux/arm64)" >&2
        exit 2
        ;;
esac
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "======================================================================"
echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "  Dockerfile: ${DOCKERFILE} | 模型目录: $MODELS_DIR/${PLATFORM:+ | platform: $PLATFORM}"
echo "======================================================================"

cd "$(dirname "$0")/.."

# ---- 构建前模型完整性校验 (模型须已人工拷入 $MODELS_DIR/; Dockerfile 内还有兜底校验) ----
echo "======================================================================"
echo "[1/2] 校验 $MODELS_DIR/ 模型完整性"
echo "======================================================================"
MISSING=0

req() {  # req <显示名> <路径>
    if [[ -e "$2" ]]; then
        echo "  OK    $1"
    else
        echo "  MISS  $1   <-- $2"
        MISSING=$((MISSING + 1))
    fi
}

req "FSMN-VAD model.pt"    "$MODELS_DIR/speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt"
req "FSMN-VAD config.yaml" "$MODELS_DIR/speech_fsmn_vad_zh-cn-16k-common-pytorch/config.yaml"
req "说话人 CampPlus (campplus_cn_common.bin)" \
    "$MODELS_DIR/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

# 可选: 实时标点 (punc.enable_realtime 默认关闭, 目录存在才校验)
PUNC_DIR="$MODELS_DIR/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
if [[ -d "$PUNC_DIR" ]]; then
    req "实时标点 model.pt (可选)" "$PUNC_DIR/model.pt"
fi

# 首遍 ONNX 导出物 (唯一真实首遍引擎, 必须完整):
# int8 量化图为默认运行路径, fp32 图供 STREAMING_ONNX_QUANT=false
for f in model_quant.onnx decoder_quant.onnx model.onnx decoder.onnx config.yaml am.mvn tokens.json; do
    req "ONNX 合批 $f" "$MODELS_DIR/onnx/$f"
done
# torch>=2.6 导出时大图可能落成外部数据 model.onnx.data, 与 model.onnx 同目录一起拷入即可
if [[ -f "$MODELS_DIR/onnx/model.onnx.data" ]]; then
    echo "  OK    ONNX 合批 model.onnx.data (fp32 外部数据, 随目录一起打包)"
fi

if [[ "$MISSING" -gt 0 ]]; then
    echo
    echo "[ERROR] 模型文件不完整: 缺 $MISSING 项, 已中止构建。" >&2
    echo "请把各模型目录完整拷贝到 $MODELS_DIR/ 后重试 (期望布局见本脚本头部注释);" >&2
    echo "或手动运行生成脚本代替拷贝:" >&2
    echo "  bash scripts/prepare_models.sh                          # VAD + 说话人 (+ 导出源模型)" >&2
    echo "  .venv/bin/python scripts/export_onnx.py                # ONNX 导出物" >&2
    exit 1
fi
echo "模型校验通过。"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "CHECK_ONLY=1, 跳过构建。"
    exit 0
fi

# ---- 构建 ----
echo "======================================================================"
echo "[2/2] docker build: ${FULL_IMAGE_NAME}"
echo "======================================================================"

BUILD_ARGS=(-f "$DOCKERFILE" -t "${FULL_IMAGE_NAME}")
[[ -n "${PLATFORM:-}" ]] && BUILD_ARGS+=(--platform "$PLATFORM")
docker build "${BUILD_ARGS[@]}" .

echo "======================================================================"
echo "Successfully built: ${FULL_IMAGE_NAME}"
echo ""
echo "镜像内模型: /opt/asrflow/models (onnx/ 合批导出物 + VAD + 说话人)。"
echo ""
echo "离线交付 (目标机器无公网):"
echo "  docker save ${FULL_IMAGE_NAME} | gzip > asrflow-image.tar.gz"
echo "  # 目标机器: docker load < asrflow-image.tar.gz"
echo "  #           bash deployment/init-compose.sh && cd deployment && docker compose up -d"
echo ""
echo "本地运行: bash deployment/init-compose.sh && cd deployment && docker compose up -d"
echo "======================================================================"
