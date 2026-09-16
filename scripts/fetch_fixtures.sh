#!/usr/bin/env bash
# Fetch public sample audio into fixtures/ and convert to 16 kHz mono PCM WAV.
# Verified URLs only — do not invent alternate hosts without updating checksums.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/fixtures"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

JFK_URL_PRIMARY="https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
JFK_URL_FALLBACK="https://cdn.jsdelivr.net/gh/openai/whisper@main/tests/jfk.flac"
JFK_FLAC_SHA256="63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
WAV_SHA256="fa9c009558aa3f214e4e8d442e16a3153c138065364871d24b6e54d55c70cd68"
WAV_OUT="${OUT_DIR}/sample_16k.wav"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd ffmpeg
need_cmd sha256sum

download() {
  local url="$1" dest="$2"
  echo "Downloading: $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

mkdir -p "$OUT_DIR"
FLAC_PATH="${TMP_DIR}/jfk.flac"

if ! download "$JFK_URL_PRIMARY" "$FLAC_PATH"; then
  echo "Primary URL failed; trying CDN fallback..."
  download "$JFK_URL_FALLBACK" "$FLAC_PATH"
fi

got="$(sha256sum "$FLAC_PATH" | awk '{print $1}')"
if [[ "$got" != "$JFK_FLAC_SHA256" ]]; then
  echo "error: FLAC checksum mismatch" >&2
  echo "  expected: $JFK_FLAC_SHA256" >&2
  echo "  got:      $got" >&2
  exit 1
fi
echo "FLAC checksum OK"

ffmpeg -y -i "$FLAC_PATH" -ar 16000 -ac 1 -c:a pcm_s16le "$WAV_OUT" </dev/null

got_wav="$(sha256sum "$WAV_OUT" | awk '{print $1}')"
if [[ "$got_wav" != "$WAV_SHA256" ]]; then
  echo "warning: WAV checksum differs from committed fixture (ffmpeg version skew?)" >&2
  echo "  expected: $WAV_SHA256" >&2
  echo "  got:      $got_wav" >&2
  echo "  file written anyway: $WAV_OUT" >&2
else
  echo "WAV checksum OK"
fi

echo "Wrote $WAV_OUT"
