FROM python:3.10-slim-bookworm

WORKDIR /app

# 网关镜像 (linux/amd64 专用; arm64 见 Dockerfile.arm64). 镜像名均为
# asrflow:<版本>, 由 deployment/build.sh 按构建机架构 (或 PLATFORM) 自动选择.
# 国内镜像源, 构建时可覆盖:
#   --build-arg APT_MIRROR=mirrors.aliyun.com
#   --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
#   --build-arg TORCH_FIND_LINKS=https://download.pytorch.org/whl/cpu
ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
ARG TORCH_VERSION=2.6.0

# Install system audio and build tools
RUN set -e; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# amd64 钉 +cpu 轮子, 避免 pip 选到捆绑 CUDA 的更高版本.
# (arm64 用 Dockerfile.arm64: 官方 aarch64 轮本身即 CPU 版, 且 +cpu 本地版本号
#  在 pytorch-wheels/cpu 镜像上并无 torchaudio aarch64 轮可钉.)
RUN set -e; \
    pip3 install --no-cache-dir --upgrade pip -i ${PIP_INDEX_URL} && \
    pip3 install --no-cache-dir -i ${PIP_INDEX_URL} \
        -f ${TORCH_FIND_LINKS} \
        "torch==${TORCH_VERSION}+cpu" "torchaudio==${TORCH_VERSION}+cpu" && \
    pip3 install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt

# Offline model bundle: staged under models/ BEFORE building (ONNX export via
# scripts/export_onnx.py, VAD/speaker via scripts/prepare_models.sh or manual
# copy). Kept OUTSIDE /app so bind-mounting host source over /app never hides
# the packaged models. The torch Paraformer source model is export-time only
# and excluded by .dockerignore.
COPY models/ /opt/asrflow/models/

RUN set -e; \
    if [ ! -d /opt/asrflow/models/speech_fsmn_vad_zh-cn-16k-common-pytorch ]; then \
        echo "ERROR: models/ 下缺少 FSMN-VAD 模型, 先执行: bash scripts/prepare_models.sh" >&2; exit 1; \
    fi; \
    if [ ! -d /opt/asrflow/models/speech_campplus_sv_zh-cn_16k-common ]; then \
        echo "ERROR: models/ 下缺少说话人模型, 先执行: bash scripts/prepare_models.sh" >&2; exit 1; \
    fi; \
    for f in model_quant.onnx decoder_quant.onnx model.onnx decoder.onnx config.yaml am.mvn tokens.json; do \
        if [ ! -f "/opt/asrflow/models/onnx/$f" ]; then \
            echo "ERROR: models/onnx/ 缺少 $f (首遍 ONNX 合批引擎需要; 重新生成: scripts/export_onnx.py)" >&2; exit 1; \
        fi; \
    done; \
    echo "model bundle OK: onnx (int8+fp32) + vad + speaker"

COPY main.py ./
COPY core/ core/
COPY config/ config/
COPY pipeline/ pipeline/
COPY server/ server/
COPY scripts/ scripts/

COPY deployment/docker-entrypoint.sh /usr/local/bin/asrflow-entrypoint.sh

EXPOSE 10095 10096

ENV PYTHONUNBUFFERED=1
ENV ASR_DEVICE=cpu
ENV ASR_SERVER_HOST=0.0.0.0
ENV ASR_SERVER_PORT=10095
ENV ASR_HTTP_PORT=10096

ENTRYPOINT ["bash", "/usr/local/bin/asrflow-entrypoint.sh"]
