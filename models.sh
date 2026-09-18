#!/usr/bin/env bash
# Download yuey.cpp GGUFs with curl. Use models.cmd on Windows without Git Bash.
# Usage: ./models.sh [--encoding bf16|q8_0|q4_k_m] [--profile core|transcribe|full]
#                    [--namespace HF_USER] [--out DIR] [--dry-run]
set -eu

ENCODING="q4_k_m"
PROFILE="full"
NAMESPACE="thepatch"
OUT="models"
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --encoding) ENCODING="$2"; shift ;;
    --profile) PROFILE="$2"; shift ;;
    --namespace) NAMESPACE="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$ENCODING" in
  bf16|BF16) ENC="BF16" ;;
  q8_0|Q8_0) ENC="Q8_0" ;;
  q4_k_m|Q4_K_M) ENC="Q4_K_M" ;;
  *) echo "unpublished encoding: $ENCODING (available: bf16|q8_0|q4_k_m)" >&2; exit 2 ;;
esac
case "$PROFILE" in core|transcribe|full) ;; *) echo "unknown profile: $PROFILE (core|transcribe|full)" >&2; exit 2 ;; esac

case "$OUT" in
  [A-Za-z]:/*) if command -v wslpath >/dev/null 2>&1; then OUT=$(wslpath -u "$OUT"); fi ;;
esac

REPO="$NAMESPACE/YuE2-3B-GGUF"
mkdir -p "$OUT"

dl() {
  local file="$1" dst="$OUT/$1" part="$OUT/$1.part"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[plan] https://huggingface.co/$REPO/resolve/main/$file -> $dst"
    return
  fi
  if [ -f "$dst" ]; then echo "[skip] $file"; return; fi
  if [ -f "$part" ]; then echo "[resume] $file"; else echo "[download] $REPO/$file"; fi
  curl -fL --retry 3 --continue-at - -o "$part" "https://huggingface.co/$REPO/resolve/main/$file"
  mv "$part" "$dst"
}

dl "yue2-3.6B-v1.0-$ENC.gguf"
dl "yue2-vae-v1.0-F16.gguf"
dl "yue2-qwen.tiktoken"
if [ "$PROFILE" != "core" ]; then dl "sheetsage2-mert2-0.7B-v1.0-F16.gguf"; fi
if [ "$PROFILE" = "full" ]; then
  dl "yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf"
  dl "yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf"
  dl "yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf"
fi
echo "[done] Yuey $PROFILE ($ENC) -> $OUT/"
