#!/bin/bash
set -e

# ==============================================================================
# Build Script for Qwen3-ASR vLLM Image (二遍高精 ASR 独立镜像)
#
# 在官方 vLLM 基础镜像上重装 vllm[audio]==<同版本>: 基础镜像只带纯文本 vLLM,
# 缺音频依赖 (soundfile/librosa 等), Qwen3-ASR 在线推理必需; 版本与基础镜像
# 内置 vllm 一致, 已满足的依赖 pip 默认跳过, 不会动内置 vllm/torch。
# pip 源: 统一阿里云 (清华源在昇腾机实测拉不到 numpy 等候选), 可 PIP_INDEX_URL 覆盖。
#
# 与 asrflow 网关镜像 (build.sh) 相互独立: 本镜像不含本仓库代码与模型,
# 也不需要 models/ 校验; context 只取 deployment/ 目录。
# 运行编排参考: deployment/docker-compose.qwen3-asr.gpu.example.yaml (NVIDIA)
#               deployment/docker-compose.qwen3-asr.ascend.example.yaml (昇腾)
#
# Dockerfile 按部署硬件分开维护 (差异原因见各自文件头注释):
#   gpu          (NVIDIA)      : Dockerfile.vllm         <- vllm/vllm-openai:v0.28.0                -> vllm-qwen3-asr:0.28.0
#   ascend       (昇腾 910B)   : Dockerfile.vllm.ascend  <- quay.io/ascend/vllm-ascend:v0.23.0      -> vllm-qwen3-asr:0.23.0-ascend
#   ascend-310p  (昇腾 310P)   : Dockerfile.vllm.ascend  <- quay.io/ascend/vllm-ascend:v0.23.0-310p -> vllm-qwen3-asr:0.23.0-310p
#                     (后缀避免各 target tag 冲突; 310P 与 910B 基础镜像同仓库同包环境, 共用 Dockerfile)
#
# Usage:
#   bash deployment/build-vllm.sh               # gpu (默认)
#   bash deployment/build-vllm.sh ascend        # 昇腾基础镜像 (910B)
#   bash deployment/build-vllm.sh ascend-310p   # 昇腾 310P 基础镜像
#   ARCH=ascend bash deployment/build-vllm.sh   # 同上 (环境变量形式)
#
# Args / Env:
#   $1 / ARCH          gpu (默认) | ascend | ascend-310p
#   VLLM_BASE_IMAGE    覆盖基础镜像 (换私有仓库镜像时)
#   VLLM_VERSION       覆盖 vllm[audio] 版本 (默认与基础镜像同版本: gpu 0.28.0 / ascend、ascend-310p 0.23.0)
#   PIP_INDEX_URL      pip 源 (默认阿里云; 清华源在昇腾机实测拉不到 numpy 等候选)
#   PLATFORM           可选, 传给 docker build --platform (如 linux/amd64)
# ==============================================================================

cd "$(dirname "$0")/.."

ARCH="${1:-${ARCH:-gpu}}"   # gpu (NVIDIA) | ascend (昇腾 910B) | ascend-310p (昇腾 310P)
case "$ARCH" in
    gpu)
        BASE_IMAGE="vllm/vllm-openai:v0.28.0"
        PIN_VERSION="0.28.0"
        TAG_SUFFIX=""
        DOCKERFILE="deployment/Dockerfile.vllm"
        COMPOSE_REF="gpu"
        ;;
    ascend)
        BASE_IMAGE="quay.io/ascend/vllm-ascend:v0.23.0"
        PIN_VERSION="0.23.0"
        TAG_SUFFIX="-ascend"
        DOCKERFILE="deployment/Dockerfile.vllm.ascend"
        COMPOSE_REF="ascend"
        ;;
    ascend-310p)
        # 310P 与 910B 同仓库同包环境, 共用 Dockerfile.vllm.ascend, 仅基础镜像与 tag 后缀不同
        BASE_IMAGE="quay.io/ascend/vllm-ascend:v0.23.0-310p"
        PIN_VERSION="0.23.0"
        TAG_SUFFIX="-310p"
        DOCKERFILE="deployment/Dockerfile.vllm.ascend"
        COMPOSE_REF="ascend"
        ;;
    *)
        echo "[ERROR] 架构必须是 gpu、ascend 或 ascend-310p: $ARCH (用法: build-vllm.sh [gpu|ascend|ascend-310p])" >&2
        exit 2
        ;;
esac
# 环境变量可覆盖 (换私有仓库镜像 / 调整版本对)
BASE_IMAGE="${VLLM_BASE_IMAGE:-$BASE_IMAGE}"
PIN_VERSION="${VLLM_VERSION:-$PIN_VERSION}"
IMAGE_NAME="vllm-qwen3-asr"
FULL_IMAGE_NAME="${IMAGE_NAME}:${PIN_VERSION}${TAG_SUFFIX}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

echo "======================================================================"
echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "  Dockerfile: ${DOCKERFILE} | 基础镜像: ${BASE_IMAGE}"
echo "  vllm[audio]==${PIN_VERSION} | pip 源: ${PIP_INDEX_URL}${PLATFORM:+ | platform: $PLATFORM}"
echo "======================================================================"

BUILD_ARGS=(
    -f "${DOCKERFILE}" deployment
    --build-arg BASE_IMAGE="$BASE_IMAGE"
    --build-arg VLLM_VERSION="$PIN_VERSION"
    --build-arg PIP_INDEX_URL="$PIP_INDEX_URL"
    -t "${FULL_IMAGE_NAME}"
)
[[ -n "${PLATFORM:-}" ]] && BUILD_ARGS+=(--platform "$PLATFORM")
docker build "${BUILD_ARGS[@]}"

echo "======================================================================"
echo "Successfully built: ${FULL_IMAGE_NAME}"
echo ""
    echo "运行参考: deployment/docker-compose.qwen3-asr.${COMPOSE_REF}.example.yaml"
if [[ "$COMPOSE_REF" != "$ARCH" ]]; then
    echo "  (310P 无独立编排模板, 沿用 ascend 模板, 把其中镜像 tag 换为 ${FULL_IMAGE_NAME} 即可)"
fi
echo "  (asrflow 侧对齐: VLLM_URL=http://<宿主>:8899/v1/audio/transcriptions,"
echo "   FINAL_ASR_MODEL 须与 compose 的 --served-model-name 一致)"
echo ""
echo "离线交付 (目标机器无公网):"
echo "  docker save ${FULL_IMAGE_NAME} | gzip > ${IMAGE_NAME}-image.tar.gz"
echo "  # 目标机器: docker load < ${IMAGE_NAME}-image.tar.gz"
echo "======================================================================"
