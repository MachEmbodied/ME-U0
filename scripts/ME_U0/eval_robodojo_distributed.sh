#!/usr/bin/env bash
# eval_robodojo_distributed.sh
#   ├─ robodojo/install_robodojo_xpolicy_client.sh
#   └─ robodojo/run_robodojo_eval_lane.sh
#        └─ eval_robodojo.sh
#             └─ run_robodojo_policy_server.sh
#                  └─ serve_robodojo_xpolicy.py
#                       └─ robodojo_xpolicy_policy.py
set -euo pipefail

# Public one-command RoboDojo evaluator.  Every GPU pair is an independent
# stateful ME_U0 policy + Isaac-Sim lane.  Across machines, task/seed
# invocations are assigned deterministically by (node_rank, local_lane).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/ME_U0/eval_robodojo_distributed.sh"
UNIFIED_ROOT="${ROBODOJO_UNIFIED_ROOT:-/path/to/unified_eval}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${UNIFIED_ROOT}/source/RoboDojo-25691aa78fb3}"
DEFAULT_CONFIG="${REPO_ROOT}/leap/configs/experiments/robodojo_sim_posttraining.yaml"
CHECKPOINT="${ROBODOJO_LEAP_CHECKPOINT:-}"
CONFIG="${ROBODOJO_LEAP_CONFIG:-${DEFAULT_CONFIG}}"
STATS="${ROBODOJO_STATS_PATH:-}"
OUTPUT_ROOT=""
RUN_ID="${ROBODOJO_FULL_RUN_ID:-}"
TASK_SUITE="all"
TASK_SPEC=""
SEEDS="0,1,2"
EVAL_NUM="native"
NUM_NODES=1
NODE_RANK="${ROBODOJO_NODE_RANK:-auto}"
SERVER_GPUS="0,1,2,3"
SIM_GPUS="0,1,2,3"
PORT_BASE=19000
ACTION_CHUNK_SIZE=16
NUM_INFERENCE_STEPS=""
IMAGE_SIZE=""
MOSAIC_LAYOUT="auto"
ENV_CFG_TYPE="arx_x5"
FAIL_FAST=0
DRY_RUN=0
DETACH=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: bash scripts/ME_U0/eval_robodojo_distributed.sh [options]

Core options:
  --config PATH                  ME_U0 experiment config
  --checkpoint PATH              step_N directory or mp_rank_00_model_states.pt
  --stats PATH                   override config's RoboDojo stats JSON
  --output-root PATH             auto-derived from checkpoint experiment/step/config
  --run-id ID                    optional override; auto-derived identically on all nodes
  --task-suite NAME              all/generalization/precision/long-horizon/memory/open
  --tasks CSV                    explicit runnable task configs; overrides task-suite
  --seeds CSV                    default: 0,1,2
  --eval-num native|N            native formal counts, or a capped smoke-test count

Parallel options:
  --num-nodes N                  total evaluation machines
  --node-rank R|auto             default auto; atomically claim a rank in shared output
  --server-gpus CSV              one policy GPU per local lane
  --sim-gpus CSV                 one Isaac-Sim GPU per local lane
  --port-base PORT               local policy ports are PORT + lane_id

Policy options:
  --action-chunk-size N          execute N actions, up to the checkpoint's trained horizon
  --num-inference-steps N        diffusion inference steps; config default when omitted
  --image-size N                 per-camera diffusion input size; config default when omitted
  --mosaic-layout NAME           auto (default, from config), horizontal or pyramid

Runtime options:
  --detach                       run this node in a tmux session
  --dry-run                      resolve and shard tasks without loading model/Isaac Sim
  --fail-fast                    stop a lane after its first failed invocation
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|--checkpoint|--stats|--output-root|--run-id|--task-suite|--tasks|--seeds|--eval-num|--num-nodes|--node-rank|--server-gpus|--sim-gpus|--port-base|--action-chunk-size|--num-inference-steps|--image-size|--mosaic-layout|--env-cfg-type)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[robodojo] missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --config) CONFIG="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --stats) STATS="$2" ;;
        --output-root) OUTPUT_ROOT="$2" ;;
        --run-id) RUN_ID="$2" ;;
        --task-suite) TASK_SUITE="$2" ;;
        --tasks) TASK_SPEC="$2" ;;
        --seeds) SEEDS="$2" ;;
        --eval-num) EVAL_NUM="$2" ;;
        --num-nodes) NUM_NODES="$2" ;;
        --node-rank) NODE_RANK="$2" ;;
        --server-gpus) SERVER_GPUS="$2" ;;
        --sim-gpus) SIM_GPUS="$2" ;;
        --port-base) PORT_BASE="$2" ;;
        --action-chunk-size) ACTION_CHUNK_SIZE="$2" ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2" ;;
        --image-size) IMAGE_SIZE="$2" ;;
        --mosaic-layout) MOSAIC_LAYOUT="$2" ;;
        --env-cfg-type) ENV_CFG_TYPE="$2" ;;
      esac
      shift 2
      ;;
    --detach) DETACH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[robodojo] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -d "${CHECKPOINT}" ]]; then
  if [[ -f "${CHECKPOINT}/pytorch_model/mp_rank_00_model_states.pt" ]]; then
    CHECKPOINT="${CHECKPOINT}/pytorch_model/mp_rank_00_model_states.pt"
  elif [[ -f "${CHECKPOINT}/mp_rank_00_model_states.pt" ]]; then
    CHECKPOINT="${CHECKPOINT}/mp_rank_00_model_states.pt"
  else
    echo "[robodojo] checkpoint directory has no mp_rank_00_model_states.pt: ${CHECKPOINT}" >&2
    exit 2
  fi
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[robodojo] required file is missing: ${CONFIG}" >&2
  exit 2
fi
readarray -t config_contract < <(
  cd "${REPO_ROOT}"
  python3 - "${CONFIG}" <<'PY'
from pathlib import Path
import sys

from leap.core.config import load_config

cfg = load_config(sys.argv[1])
datasets = [
    dataset
    for dataset in cfg.data.train.get("datasets", [])
    if "robodojo" in str(dataset.get("dataset_name", "")).lower()
]
if len(datasets) != 1:
    raise SystemExit(f"expected one RoboDojo dataset, found {len(datasets)}")
print(str(cfg.get("robodojo_action_type", "joint")))
stats_path = Path(str(datasets[0].pretrained_norm_stats))
if not stats_path.is_absolute():
    stats_path = Path.cwd() / stats_path
print(str(stats_path.resolve()))
PY
)
if [[ ${#config_contract[@]} -ne 2 ]]; then
  echo "[robodojo] failed to resolve action type/stats from config: ${CONFIG}" >&2
  exit 2
fi
ACTION_TYPE="${ROBODOJO_ACTION_TYPE:-${config_contract[0]}}"
if [[ "${ACTION_TYPE}" != "joint" ]]; then
  echo "[robodojo] ME_U0 RoboDojo evaluation only supports joint actions, got: ${ACTION_TYPE}" >&2
  exit 2
fi
if [[ -z "${STATS}" ]]; then
  STATS="${config_contract[1]}"
elif [[ "${STATS}" != /* ]]; then
  STATS="${REPO_ROOT}/${STATS}"
fi

for file in "${CHECKPOINT}" "${CONFIG}" "${STATS}" \
    "${ROBODOJO_ROOT}/scripts/internal/task_inventory.py" \
    "${ROBODOJO_ROOT}/scripts/internal/summarize_result.py" \
    "${REPO_ROOT}/scripts/ME_U0/robodojo/run_robodojo_eval_lane.sh"; do
  if [[ ! -f "${file}" ]]; then
    echo "[robodojo] required file is missing: ${file}" >&2
    exit 2
  fi
done

CHECKPOINT_STEP="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
CHECKPOINTS_DIR="$(dirname "$(dirname "$(dirname "${CHECKPOINT}")")")"
EXPERIMENT_ROOT="$(dirname "${CHECKPOINTS_DIR}")"
config_name="$(basename "${CONFIG}" .yaml)"
case "${config_name}" in
  *delta_joint*source_q01q99*) output_tag="delta_joint_source_q01q99" ;;
  *source_minmax*) output_tag="source_minmax" ;;
  *source_q01q99*) output_tag="source_q01q99" ;;
  *) output_tag="custom" ;;
esac
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${EXPERIMENT_ROOT}/robodojo_eval_${CHECKPOINT_STEP}_ME_U0_${output_tag}_full"
fi
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${CHECKPOINT_STEP}_${output_tag}_full"
fi

for file in "${ROBODOJO_ROOT}/src/eval_client/main.py" \
    "${ROBODOJO_ROOT}/src/eval_client/eval_env.py"; do
  if ! grep -q 'ROBODOJO_EVAL_ROOT' "${file}"; then
    echo "[robodojo] ${file} lacks direct-output support (ROBODOJO_EVAL_ROOT)" >&2
    exit 2
  fi
done
if [[ ! "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[robodojo] num-nodes must be a positive integer" >&2
  exit 2
fi
if [[ "${NODE_RANK}" == "auto" ]]; then
  mkdir -p "${OUTPUT_ROOT}/node_rank_claims"
  node_owner="${ROBODOJO_NODE_ID:-}"
  if [[ -z "${node_owner}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    node_owner="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
  fi
  if [[ -z "${node_owner}" ]]; then
    node_owner="$(hostname -s)"
  fi
  claimed_rank=""
  for ((candidate=0; candidate<NUM_NODES; candidate++)); do
    claim_dir="${OUTPUT_ROOT}/node_rank_claims/rank_${candidate}"
    if [[ -f "${claim_dir}/owner_id" ]] && [[ "$(<"${claim_dir}/owner_id")" == "${node_owner}" ]]; then
      claimed_rank="${candidate}"
      break
    fi
  done
  if [[ -z "${claimed_rank}" ]]; then
    for ((candidate=0; candidate<NUM_NODES; candidate++)); do
      claim_dir="${OUTPUT_ROOT}/node_rank_claims/rank_${candidate}"
      if mkdir "${claim_dir}" 2>/dev/null; then
        printf '%s\n' "${node_owner}" >"${claim_dir}/owner_id"
        printf 'claimed_at=%s\nhost=%s\n' "$(date --iso-8601=seconds)" "$(hostname)" \
          >"${claim_dir}/claim.env"
        claimed_rank="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${claimed_rank}" ]]; then
    echo "[robodojo] no free node rank in [0, ${NUM_NODES}); use an explicit --node-rank to recover a stale claim" >&2
    exit 4
  fi
  NODE_RANK="${claimed_rank}"
  echo "[robodojo] auto node rank=${NODE_RANK}/${NUM_NODES} owner=${node_owner}"
fi
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ || "${NODE_RANK}" -ge "${NUM_NODES}" ]]; then
  echo "[robodojo] invalid node rank ${NODE_RANK}/${NUM_NODES}" >&2
  exit 2
fi
if [[ ! "${PORT_BASE}" =~ ^[1-9][0-9]*$ || "${PORT_BASE}" -gt 65535 ]]; then
  echo "[robodojo] invalid port base: ${PORT_BASE}" >&2
  exit 2
fi
if [[ ! "${ACTION_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[robodojo] action-chunk-size must be a positive integer" >&2
  exit 2
fi
if [[ -n "${NUM_INFERENCE_STEPS}" && ! "${NUM_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[robodojo] num-inference-steps must be positive" >&2
  exit 2
fi
if [[ -n "${IMAGE_SIZE}" && ! "${IMAGE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[robodojo] image-size must be positive" >&2
  exit 2
fi
if [[ "${MOSAIC_LAYOUT}" != "auto" && "${MOSAIC_LAYOUT}" != "horizontal" && "${MOSAIC_LAYOUT}" != "pyramid" ]]; then
  echo "[robodojo] mosaic-layout must be auto, horizontal or pyramid" >&2
  exit 2
fi
if [[ "${MOSAIC_LAYOUT}" == "pyramid" && -n "${IMAGE_SIZE}" ]]; then
  echo "[robodojo] omit --image-size with pyramid; [height,width] comes from config" >&2
  exit 2
fi
if [[ "${EVAL_NUM}" != "native" && ! "${EVAL_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[robodojo] eval-num must be native or positive" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "[robodojo] unsafe run id: ${RUN_ID}" >&2
  exit 2
fi

IFS=',' read -r -a SERVER_GPU_ARRAY <<<"${SERVER_GPUS}"
IFS=',' read -r -a SIM_GPU_ARRAY <<<"${SIM_GPUS}"
if [[ ${#SERVER_GPU_ARRAY[@]} -eq 0 || ${#SERVER_GPU_ARRAY[@]} -ne ${#SIM_GPU_ARRAY[@]} ]]; then
  echo "[robodojo] server-gpus and sim-gpus must contain the same non-zero number of entries" >&2
  exit 2
fi
for gpu in "${SERVER_GPU_ARRAY[@]}" "${SIM_GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "[robodojo] invalid GPU index: ${gpu}" >&2
    exit 2
  fi
done
LOCAL_LANES=${#SERVER_GPU_ARRAY[@]}
if (( PORT_BASE + LOCAL_LANES - 1 > 65535 )); then
  echo "[robodojo] policy port range exceeds 65535" >&2
  exit 2
fi

INVENTORY="${ROBODOJO_ROOT}/scripts/internal/task_inventory.py"
if [[ -z "${TASK_SPEC}" && "${TASK_SUITE}" != "all" ]]; then
  mapfile -t SUITE_TASKS < <(cd "${ROBODOJO_ROOT}" && python3 "${INVENTORY}" \
    --only-runnable --dimension "${TASK_SUITE}")
  if [[ ${#SUITE_TASKS[@]} -eq 0 ]]; then
    echo "[robodojo] task-suite resolved to no runnable tasks: ${TASK_SUITE}" >&2
    exit 3
  fi
  printf -v TASK_SPEC '%s,' "${SUITE_TASKS[@]}"
  TASK_SPEC="${TASK_SPEC%,}"
fi

EVAL_ROOT="${OUTPUT_ROOT}/results/RoboDojo"
NODE_ROOT="${OUTPUT_ROOT}/nodes/node_${NODE_RANK}"
NODE_LOCAL_CACHE_BASE="${ROBODOJO_NODE_LOCAL_CACHE_BASE:-/tmp/ME_U0_robodojo_cache/${RUN_ID}/node_${NODE_RANK}}"
mkdir -p "${NODE_ROOT}" "${EVAL_ROOT}"

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'node_rank=%s\n' "${NODE_RANK}"
  printf 'num_nodes=%s\n' "${NUM_NODES}"
  printf 'server_gpus=%s\n' "${SERVER_GPUS}"
  printf 'sim_gpus=%s\n' "${SIM_GPUS}"
  printf 'node_local_cache_base=%s\n' "${NODE_LOCAL_CACHE_BASE}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'eval_root=%s\n' "${EVAL_ROOT}"
} >"${NODE_ROOT}/node.env"

if [[ "${DETACH}" == "1" && "${ROBODOJO_DISTRIBUTED_IN_TMUX:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[robodojo] tmux is required by --detach" >&2
    exit 2
  fi
  SESSION="robodojo_${RUN_ID}_node${NODE_RANK}"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[robodojo] tmux session already exists: ${SESSION}" >&2
    exit 4
  fi
  command_parts=(env ROBODOJO_DISTRIBUTED_IN_TMUX=1 bash "${SCRIPT_PATH}")
  command_parts+=("${ORIGINAL_ARGS[@]}")
  printf -v command_q '%q ' "${command_parts[@]}"
  printf -v repo_q '%q' "${REPO_ROOT}"
  printf -v log_q '%q' "${NODE_ROOT}/node.log"
  tmux new-session -d -s "${SESSION}" \
    "set -o pipefail; cd ${repo_q} && ${command_q}2>&1 | tee -a ${log_q}"
  {
    printf 'session=%s\n' "${SESSION}"
    printf 'log=%s\n' "${NODE_ROOT}/node.log"
    printf 'launched_at=%s\n' "$(date --iso-8601=seconds)"
  } >"${NODE_ROOT}/tmux.env"
  echo "[robodojo] detached node evaluator started"
  echo "[robodojo] session=${SESSION}"
  echo "[robodojo] output_root=${OUTPUT_ROOT}"
  echo "[robodojo] log=${NODE_ROOT}/node.log"
  exit 0
fi

ROBODOJO_ACTION_TYPE="${ACTION_TYPE}" \
  "${REPO_ROOT}/scripts/ME_U0/robodojo/install_robodojo_xpolicy_client.sh"

TOTAL_LANES=$((NUM_NODES * LOCAL_LANES))
lane_pids=()
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${lane_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${lane_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  printf 'stopped_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${rc}" >>"${NODE_ROOT}/node.env"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

echo "[robodojo] node=${NODE_RANK}/${NUM_NODES} local_lanes=${LOCAL_LANES} total_lanes=${TOTAL_LANES}"
for ((lane_id=0; lane_id<LOCAL_LANES; lane_id++)); do
  shard_index=$((NODE_RANK * LOCAL_LANES + lane_id))
  port=$((PORT_BASE + lane_id))
  lane_root="${NODE_ROOT}/lanes/lane_${lane_id}"
  mkdir -p "${lane_root}"
  (
    ROBODOJO_UNIFIED_ROOT="${UNIFIED_ROOT}" \
    ROBODOJO_ROOT="${ROBODOJO_ROOT}" \
    ROBODOJO_LEAP_CHECKPOINT="${CHECKPOINT}" \
    ROBODOJO_LEAP_CONFIG="${CONFIG}" \
    ROBODOJO_STATS_PATH="${STATS}" \
    ROBODOJO_FULL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    ROBODOJO_EVAL_ROOT="${EVAL_ROOT}" \
    ROBODOJO_NODE_ROOT="${NODE_ROOT}" \
    ROBODOJO_FULL_RUN_ID="${RUN_ID}" \
    ROBODOJO_NODE_RANK="${NODE_RANK}" \
    ROBODOJO_LANE_ID="${lane_id}" \
    ROBODOJO_LANE_CACHE_ROOT="${NODE_LOCAL_CACHE_BASE}/lane_${lane_id}" \
    ROBODOJO_SHARD_COUNT="${TOTAL_LANES}" \
    ROBODOJO_SHARD_INDEX="${shard_index}" \
    ROBODOJO_POLICY_GPU="${SERVER_GPU_ARRAY[$lane_id]}" \
    ROBODOJO_SIM_GPU="${SIM_GPU_ARRAY[$lane_id]}" \
    ROBODOJO_POLICY_PORT="${port}" \
    ROBODOJO_ENV_CFG_TYPE="${ENV_CFG_TYPE}" \
    ROBODOJO_ACTION_TYPE="${ACTION_TYPE}" \
    ROBODOJO_FULL_EVAL_NUM="${EVAL_NUM}" \
    ROBODOJO_FULL_SEEDS="${SEEDS}" \
    ROBODOJO_FULL_TASKS="${TASK_SPEC}" \
    ROBODOJO_FULL_FAIL_FAST="${FAIL_FAST}" \
    ROBODOJO_FULL_DRY_RUN="${DRY_RUN}" \
    ROBODOJO_ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE}" \
    ROBODOJO_NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    ROBODOJO_IMAGE_SIZE="${IMAGE_SIZE}" \
    ROBODOJO_MOSAIC_LAYOUT="${MOSAIC_LAYOUT}" \
      bash "${REPO_ROOT}/scripts/ME_U0/robodojo/run_robodojo_eval_lane.sh"
  ) >>"${lane_root}/lane.log" 2>&1 &
  lane_pids+=("$!")
  echo "[robodojo] lane=${lane_id} shard=${shard_index}/${TOTAL_LANES} policy_gpu=${SERVER_GPU_ARRAY[$lane_id]} sim_gpu=${SIM_GPU_ARRAY[$lane_id]} port=${port} pid=$!"
done

node_rc=0
for pid in "${lane_pids[@]}"; do
  if ! wait "${pid}"; then
    node_rc=7
  fi
done
lane_pids=()

if [[ "${DRY_RUN}" != "1" ]]; then
  (
    if command -v flock >/dev/null 2>&1; then
      flock -x 9
    fi
    ROBODOJO_EVAL_ROOT="${EVAL_ROOT}" \
      python3 "${ROBODOJO_ROOT}/scripts/internal/summarize_result.py" \
      >"${NODE_ROOT}/summary.log" 2>&1 || true
    if [[ -s "${EVAL_ROOT}/_summary.md" ]]; then
      cp -f "${EVAL_ROOT}/_summary.md" "${OUTPUT_ROOT}/summary.md"
    fi
  ) 9>"${OUTPUT_ROOT}/.summary.lock"
fi

printf 'completed_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${node_rc}" >"${NODE_ROOT}/COMPLETED"
echo "[robodojo] node ${NODE_RANK} complete rc=${node_rc} output_root=${OUTPUT_ROOT}"
exit "${node_rc}"
