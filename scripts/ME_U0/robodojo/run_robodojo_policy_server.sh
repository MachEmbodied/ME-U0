#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIFIED_ROOT="${ROBODOJO_UNIFIED_ROOT:-/path/to/unified_eval}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${UNIFIED_ROOT}/source/RoboDojo-25691aa78fb3}"
PYTHON_BIN="${LEAP_POLICY_PYTHON:-python3}"
CONFIG="${ROBODOJO_LEAP_CONFIG:-${REPO_ROOT}/leap/configs/experiments/robodojo_sim_posttraining.yaml}"
CHECKPOINT="${ROBODOJO_LEAP_CHECKPOINT:-}"
STATS="${ROBODOJO_STATS_PATH:-}"
PORT="${ROBODOJO_POLICY_PORT:-19000}"
POLICY_GPU="${ROBODOJO_POLICY_GPU:-0}"
ACTION_CHUNK_SIZE="${ROBODOJO_ACTION_CHUNK_SIZE:-8}"
NUM_INFERENCE_STEPS="${ROBODOJO_NUM_INFERENCE_STEPS:-}"
PER_VIEW_SIZE="${ROBODOJO_IMAGE_SIZE:-}"
MOSAIC_LAYOUT="${ROBODOJO_MOSAIC_LAYOUT:-auto}"
DATASET_NAME="${ROBODOJO_DATASET_NAME:-}"
POLICY_SEED="${ROBODOJO_POLICY_SEED:-42}"
TASK_NAME="${ROBODOJO_TASK_NAME:-}"

export CUDA_VISIBLE_DEVICES="${POLICY_GPU}"
export PYTHONPATH="${REPO_ROOT}:${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# RTX5880 exposes 99 KiB shared memory per block.  The compiled flex-attention
# autotuner selected a 102 KiB Triton kernel and failed before the first action
# (`Required: 104448 Hardware limit: 101376`).  Eager flex attention is correct
# for this batch-1 evaluation path and passed the real step_20000 WS inference.
export TORCH_COMPILE_DISABLE="${ROBODOJO_TORCH_COMPILE_DISABLE:-1}"

optional_args=()
if [[ -n "${STATS}" ]]; then
  if [[ ! -f "${STATS}" ]]; then
    echo "[RoboDojo] stats file is missing: ${STATS}" >&2
    exit 2
  fi
  optional_args+=(--stats "${STATS}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  optional_args+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${PER_VIEW_SIZE}" ]]; then
  optional_args+=(--per-view-size "${PER_VIEW_SIZE}")
fi
if [[ -n "${DATASET_NAME}" ]]; then
  optional_args+=(--dataset-name "${DATASET_NAME}")
fi
if [[ -n "${TASK_NAME}" ]]; then
  optional_args+=(--eval-task "${TASK_NAME}")
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/ME_U0/robodojo/serve_robodojo_xpolicy.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device cuda \
  --domain-id 13 \
  --action-chunk-size "${ACTION_CHUNK_SIZE}" \
  --mosaic-layout "${MOSAIC_LAYOUT}" \
  --seed "${POLICY_SEED}" \
  "${optional_args[@]}" \
  "$@"
