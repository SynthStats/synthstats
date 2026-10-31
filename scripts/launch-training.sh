#!/usr/bin/env bash
# launch-training.sh - thin wrapper for the Harbor + SkyRL training path.
#
# Usage:
#   ./scripts/launch-training.sh
#   BACKEND=vllm POLICY=hf_qwen3_5_0_8b ./scripts/launch-training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# defaults
BACKEND="${BACKEND:-vllm}"
ENV="${ENV:-dugongs}"
POLICY="${POLICY:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_DIR}/runs/checkpoints}"

case "$BACKEND" in
  vllm)
    if [ -z "$POLICY" ]; then
      POLICY="hf_qwen3_5_0_8b"
    fi
    ;;
  *)
    echo "Unknown backend: ${BACKEND}" >&2
    echo "Options: vllm" >&2
    exit 1
    ;;
esac

echo "Backend: ${BACKEND}"
echo "Env: ${ENV}"
echo "Policy: ${POLICY}"
echo "Epochs: ${EPOCHS}"

cd "$PROJECT_DIR"
uv run synthstats-train \
  env="${ENV}" \
  policy="${POLICY}" \
  backend="${BACKEND}" \
  runner.trainer.epochs="${EPOCHS}" \
  runner.trainer.train_batch_size="${BATCH_SIZE}" \
  checkpoint.save_dir="${CKPT_PATH}" \
  runner.trainer.ckpt_path="${CKPT_PATH}" \
  "$@"
