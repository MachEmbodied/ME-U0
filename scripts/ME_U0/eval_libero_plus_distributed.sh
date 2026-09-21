#!/usr/bin/env bash
# Run this entrypoint on every node; reuse the LIBERO-plus shard evaluator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_NODES=1
NODE_RANK=auto
OUTPUT_ROOT=""
CHECKPOINT=""
SERVER_GPUS="0,1,2,3"
SIM_GPUS="0,1,2,3"
TASK_SUITE=all
SHARDS_PER_SUITE=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-nodes) NUM_NODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --output-root|--output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --server-gpus|--server-gpu) SERVER_GPUS="$2"; shift 2 ;;
        --sim-gpus) SIM_GPUS="$2"; shift 2 ;;
        --task-suite) TASK_SUITE="$2"; shift 2 ;;
        --libero-plus-shards-per-suite|--shards-per-suite) SHARDS_PER_SUITE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/ME_U0/eval_libero_plus_distributed.sh --config YAML --checkpoint PATH --num-nodes N --node-rank auto|R [--output-root DIR] [--server-gpus 0,1,2,3] [--sim-gpus 0,1,2,3] [existing LIBERO-plus options]"
            exit 0 ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
    step_dir="${CHECKPOINT%/}"
    [[ -f "${step_dir}" ]] && step_dir="$(dirname "${step_dir}")"
    [[ "$(basename "${step_dir}")" == pytorch_model ]] && step_dir="$(dirname "${step_dir}")"
    OUTPUT_ROOT="$(dirname "$(dirname "${step_dir}")")/eval_libero_plus/$(basename "${step_dir}")"
fi

if [[ "${NODE_RANK}" == auto ]]; then
    claim_root="${OUTPUT_ROOT}/node_rank_claims"
    mkdir -p "${claim_root}"
    node_owner="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
    for ((rank=0; rank<NUM_NODES; rank++)); do
        claim="${claim_root}/rank_${rank}"
        if [[ -f "${claim}/owner" && "$(<"${claim}/owner")" == "${node_owner}" ]]; then
            NODE_RANK="${rank}"
            break
        fi
        if mkdir "${claim}" 2>/dev/null; then
            printf '%s\n' "${node_owner}" > "${claim}/owner"
            NODE_RANK="${rank}"
            break
        fi
    done
    if [[ "${NODE_RANK}" == auto ]]; then
        echo "No free node rank in ${OUTPUT_ROOT}" >&2
        exit 1
    fi
fi

# With four GPUs per node, all four suites each have one shard on every node.
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-${NUM_NODES}}"
export ME_U0_LIBERO_PLUS_NUM_NODES="${NUM_NODES}"
export ME_U0_LIBERO_PLUS_NODE_RANK="${NODE_RANK}"
export ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE="${ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE:-libero}"
node_dir="${OUTPUT_ROOT}/nodes/node_${NODE_RANK}"
mkdir -p "${node_dir}"
echo "LIBERO-plus node ${NODE_RANK}/${NUM_NODES}; output=${OUTPUT_ROOT}"
bash "${SCRIPT_DIR}/eval_libero_plus.sh" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --task-suite "${TASK_SUITE}" \
    --libero-plus-shards-per-suite "${SHARDS_PER_SUITE}" \
    --server-gpu "${SERVER_GPUS}" \
    --sim-gpus "${SIM_GPUS}" \
    "${FORWARD_ARGS[@]}" 2>&1 | tee "${node_dir}/node.log"
