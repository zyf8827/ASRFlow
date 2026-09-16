#!/usr/bin/env bash
# 离线部署模型导出脚本: 把本机 ModelScope 缓存中的 FunASR 模型拷贝到 ./models/,
# 供 deployment/Dockerfile 构建时打包进镜像 (VAD/说话人) 以及 scripts/export_onnx.py
# 导出首遍 ONNX 图 (流式 Paraformer 原始模型, 仅导出输入, 不入镜像)。
# models/ 已在 .gitignore 中, 仅作构建上下文使用, 不入库。
#
# 首遍流式模型固定为小参数 Paraformer (iic/speech_paraformer_asr_nat-zh-cn-16k-
# common-vocab8404-online), 不提供 large 变体。
#
# 用法:
#   bash scripts/prepare_models.sh
#   WITH_PUNC=1 bash scripts/prepare_models.sh         # 附带实时标点模型 (默认不打包)
#   MODELSCOPE_CACHE=/data/ms_cache bash scripts/prepare_models.sh   # 非默认缓存根目录
#
# 可选环境变量:
#   WITH_PUNC=0|1           是否打包 CT-Transformer 实时标点模型 (默认 0;
#                           仅 config punc.enable_realtime=true 时需要)
#   MODELSCOPE_CACHE=...    ModelScope 缓存根目录 (默认 ~/.cache/modelscope)
#
# 缓存中缺模型时, 先在有网机器上补齐:
#   python3 scripts/download_models.py --only_funasr

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models"
CACHE_ROOT="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

WITH_PUNC="${WITH_PUNC:-0}"

# 待导出的模型列表: "<org>/<name> <revision>" (revision 仅用于缓存内定位, 运行时走本地路径)
MODELS=(
    "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online v2.0.4"
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch v2.0.4"
    "iic/speech_campplus_sv_zh-cn_16k-common master"
)
if [[ "$WITH_PUNC" == "1" ]]; then
    MODELS+=("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 v2.0.4")
fi

# 在给定基目录下定位快照: <base>/<org>--<name>/snapshots/<revision>/,
# 缺指定 revision 时回退到该模型唯一(首个)快照; 找不到返回 1
find_in_base() {
    local base="$1" org="$2" name="$3" rev="$4" only
    [[ -d "$base/$org--$name/snapshots" ]] || return 1
    if [[ -d "$base/$org--$name/snapshots/$rev" ]]; then
        echo "$base/$org--$name/snapshots/$rev"
        return 0
    fi
    only="$(ls -1 "$base/$org--$name/snapshots" 2>/dev/null | head -1 || true)"
    if [[ -n "$only" ]]; then
        echo "[WARN] $org/$name 未找到 revision $rev, 回退使用快照: $only" >&2
        echo "$base/$org--$name/snapshots/$only"
        return 0
    fi
    return 1
}

# 在 ModelScope 缓存中定位 <org>/<name> 的快照目录 (兼容多代目录布局)
find_snapshot() {
    local id="$1" rev="$2" org name
    org="${id%%/*}"
    name="${id##*/}"
    # 新版布局 (无 hub/ 段) 与近期布局 (hub/models/), 优先精确 revision
    if find_in_base "$CACHE_ROOT/models" "$org" "$name" "$rev"; then return 0; fi
    if find_in_base "$CACHE_ROOT/hub/models" "$org" "$name" "$rev"; then return 0; fi
    # 旧版平铺布局: hub/models/<org>/<name> 或 hub/<org>/<name>
    if [[ -d "$CACHE_ROOT/hub/models/$org/$name" ]]; then
        echo "$CACHE_ROOT/hub/models/$org/$name"
        return 0
    fi
    if [[ -d "$CACHE_ROOT/hub/$org/$name" ]]; then
        echo "$CACHE_ROOT/hub/$org/$name"
        return 0
    fi
    return 1
}

# 校验导出目录含配置文件与模型权重
validate_dir() {
    local d="$1" name="$2" f found=0
    if [[ ! -f "$d/configuration.json" && ! -f "$d/config.yaml" ]]; then
        echo "[ERROR] $name: $d 下缺少 configuration.json / config.yaml" >&2
        return 1
    fi
    for f in "$d"/*.pt "$d"/*.bin "$d"/*.safetensors "$d"/*.onnx; do
        if [[ -f "$f" ]]; then
            found=1
            break
        fi
    done
    if [[ "$found" != "1" ]]; then
        echo "[ERROR] $name: $d 下缺少权重文件 (*.pt/*.bin/*.safetensors/*.onnx)" >&2
        return 1
    fi
}

copy_model() {
    local src="$1" dst_name="$2" dst
    dst="$DEST/$dst_name"
    rm -rf "$dst"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        # 排除示例/图片等与推理无关的文件, 减小镜像体积
        rsync -a \
            --exclude example/ --exclude examples/ --exclude fig/ \
            --exclude '*.png' --exclude '*.jpg' --exclude '*.jpeg' --exclude '*.gif' \
            --exclude .DS_Store --exclude .gitattributes \
            "$src"/ "$dst"/
    else
        cp -a "$src"/. "$dst"/
        find "$dst" \( -name example -o -name examples -o -name fig \
            -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.gif' \
            -o -name .DS_Store -o -name .gitattributes \) -exec rm -rf {} +
    fi
}

echo "======================================================================"
echo "导出 FunASR 模型用于离线镜像构建"
echo "  源缓存: $CACHE_ROOT"
echo "  目标:   $DEST"
echo "  Paraformer: small (固定)   标点模型: $WITH_PUNC"
echo "======================================================================"

mkdir -p "$DEST"

# 第一遍: 先定位全部模型的缓存快照, 任何一个缺失都立即退出 (不动已有导出内容)
SNAPSHOTS=()
for entry in "${MODELS[@]}"; do
    id="${entry%% *}"
    rev="${entry##* }"
    if ! src="$(find_snapshot "$id" "$rev")"; then
        echo "[ERROR] 缓存中找不到模型 $id (revision $rev)。请在有网机器执行:" >&2
        echo "         python3 scripts/download_models.py --only_funasr" >&2
        exit 1
    fi
    SNAPSHOTS+=("$id|$src")
done

# 清掉之前导出的其它 Paraformer 变体, 保证镜像内只有一个流式模型目录
rm -rf "$DEST"/speech_paraformer*online

failed=0
for entry in "${SNAPSHOTS[@]}"; do
    id="${entry%%|*}"
    src="${entry#*|}"
    name="${id##*/}"
    echo "--> $id"
    echo "    源: $src"
    copy_model "$src" "$name"
    if ! validate_dir "$DEST/$name" "$id"; then
        failed=1
    fi
done

if [[ "$failed" != "0" ]]; then
    echo "[ERROR] 模型导出未完成, 请按上方提示补齐缓存后重试" >&2
    exit 1
fi

# 记录本次导出的 Paraformer 目录名 (供 build.sh / 排障参考)
echo "speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online" > "$DEST/.paraformer"

echo
echo "======================================================================"
echo "模型导出完成, 打包内容:"
du -sh "$DEST"/*/ 2>/dev/null | sed 's/^/  /'
echo "  选择标记: $DEST/.paraformer -> $(cat "$DEST/.paraformer")"
echo "下一步: bash deployment/build.sh   (构建含模型的离线镜像)"
echo "======================================================================"
