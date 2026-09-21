#!/usr/bin/env bash
# MachEmbodiedUnifiedModel policy server launcher.
#
# The model-agnostic sim clients (sim_eval / eval_libero_plus.sh client side)
# connect to this over WebSocket. Run in the non-conda leap env.
#
#   bash scripts/ME_U0/run_policy_server.sh \
#       --config leap/configs/experiments/libero_posttraining.yaml \
#       --checkpoint <RUN>/checkpoints/step_N/pytorch_model/mp_rank_00_model_states.pt \
#       --port 8765 --gpu 3 --num-inference-steps 24 --seed 42
set -euo pipefail

CONFIG=""; CHECKPOINT=""; PORT="8765"; HOST="0.0.0.0"; GPU="0"; DEVICE="cuda"
NUM_INFERENCE_STEPS=""; MAX_BATCH_SIZE=""; MAX_WAIT_MS=""; RAW_ACTION_DIM="7"; VIDEO_OUT_DIR=""; PER_VIEW_SIZE=""; EXTRA=()
SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2";     shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-wait-ms)  MAX_WAIT_MS="$2"; shift 2 ;;
        --raw-action-dim) RAW_ACTION_DIM="$2"; shift 2 ;;
        --video-out-dir) VIDEO_OUT_DIR="$2"; shift 2 ;;
        --per-view-size) PER_VIEW_SIZE="$2"; shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        *=*) EXTRA+=("$1"); shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then echo "ERROR: --config is required" >&2; exit 2; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=(python scripts/ME_U0/serve.py --config "${CONFIG}" --port "${PORT}" --host "${HOST}" --device "${DEVICE}" --raw-action-dim "${RAW_ACTION_DIM}")
[[ -n "${CHECKPOINT}" ]] && CMD+=(--checkpoint "${CHECKPOINT}")
[[ -n "${NUM_INFERENCE_STEPS}" ]] && CMD+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
[[ -n "${MAX_BATCH_SIZE}" ]] && CMD+=(--max-batch-size "${MAX_BATCH_SIZE}")
[[ -n "${MAX_WAIT_MS}" ]] && CMD+=(--max-wait-ms "${MAX_WAIT_MS}")
[[ -n "${VIDEO_OUT_DIR}" ]] && CMD+=(--video-out-dir "${VIDEO_OUT_DIR}")
[[ -n "${PER_VIEW_SIZE}" ]] && CMD+=(--per-view-size "${PER_VIEW_SIZE}")
[[ -n "${SEED}" ]] && CMD+=(--seed "${SEED}")
CMD+=("${EXTRA[@]}")

echo "==> config:     ${CONFIG}"
[[ -n "${CHECKPOINT}" ]] && echo "==> checkpoint: ${CHECKPOINT}"
[[ -n "${SEED}" ]] && echo "==> seed:       ${SEED}"
echo "==> serving on: ${HOST}:${PORT}  (CUDA_VISIBLE_DEVICES=${GPU})"
exec "${CMD[@]}"
