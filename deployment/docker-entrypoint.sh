#!/usr/bin/env bash
# ASRFlow 容器入口: 把未显式设置的模型环境变量解析到镜像内 /opt/asrflow/models
# 下打包好的模型目录, 保证离线(无公网)启动时直接从本地路径加载, 不触发下载。
#
# 解析规则: 环境变量已设置且非空则完全尊重; 否则按固定目录名 / 通配符在
# ASRFLOW_MODEL_ROOT (默认 /opt/asrflow/models, 独立于 /app 源码目录) 下查找。
# 最后 exec main.py 透传所有参数。

set -euo pipefail

MODEL_ROOT="${ASRFLOW_MODEL_ROOT:-/opt/asrflow/models}"

# resolve <ENV_NAME> <glob...>  : 第一个匹配目录赋给 ENV_NAME
resolve() {
    local env_name="$1"
    shift
    if [[ -n "${!env_name:-}" ]]; then
        echo "[entrypoint] $env_name 已设置, 使用: ${!env_name}"
        return 0
    fi
    local match
    for match in "$@"; do
        if [[ -d "$match" ]]; then
            export "$env_name=$match"
            echo "[entrypoint] $env_name 自动解析 -> $match (镜像内置模型)"
            return 0
        fi
    done
    echo "[entrypoint] 警告: $MODEL_ROOT 下未找到 $env_name 对应的模型目录 ($*)" >&2
}

resolve VAD_MODEL \
    "$MODEL_ROOT"/speech_fsmn_vad_zh-cn-16k-common-pytorch
resolve SPEAKER_MODEL \
    "$MODEL_ROOT"/speech_campplus_sv_zh-cn_16k-common
# 标点模型默认不打包; 只有目录存在时才解析 (punc.enable_realtime 默认关闭)
resolve PUNC_MODEL \
    "$MODEL_ROOT"/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727

# 首遍 ONNX 合批引擎 (唯一真实引擎) 的导出物目录; mock 后端 (测试) 之外必须有
resolve STREAMING_ONNX_DIR "$MODEL_ROOT"/onnx
if [[ "${STREAMING_ASR_BACKEND:-auto}" != "mock" && ! -d "${STREAMING_ONNX_DIR:-}" ]]; then
    echo "[entrypoint] 错误: 首遍 ONNX 导出物目录不存在 (STREAMING_ONNX_DIR=${STREAMING_ONNX_DIR:-未设置})" >&2
    echo "[entrypoint]        镜像需打包 models/onnx (见 deployment/build.sh), 或挂载目录并显式设置 STREAMING_ONNX_DIR" >&2
    exit 1
fi

exec python3 main.py "$@"
