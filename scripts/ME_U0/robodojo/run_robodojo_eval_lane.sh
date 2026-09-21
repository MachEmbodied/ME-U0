#!/usr/bin/env bash
set -euo pipefail

# Internal worker for one independent policy-server + Isaac-Sim lane.
# The public entry point is eval_robodojo_distributed.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIFIED_ROOT="${ROBODOJO_UNIFIED_ROOT:-/path/to/unified_eval}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${UNIFIED_ROOT}/source/RoboDojo-25691aa78fb3}"
CHECKPOINT="${ROBODOJO_LEAP_CHECKPOINT:?ROBODOJO_LEAP_CHECKPOINT is required}"
CONFIG="${ROBODOJO_LEAP_CONFIG:?ROBODOJO_LEAP_CONFIG is required}"
# Empty means "use dataset.pretrained_norm_stats from CONFIG".  Only an
# explicit --stats/ROBODOJO_STATS_PATH is allowed to override that contract.
STATS="${ROBODOJO_STATS_PATH:-}"
OUTPUT_ROOT="${ROBODOJO_FULL_OUTPUT_ROOT:?ROBODOJO_FULL_OUTPUT_ROOT is required}"
EVAL_ROOT="${ROBODOJO_EVAL_ROOT:-${OUTPUT_ROOT}/results/RoboDojo}"
NODE_ROOT="${ROBODOJO_NODE_ROOT:?ROBODOJO_NODE_ROOT is required}"
RUN_ID="${ROBODOJO_FULL_RUN_ID:?ROBODOJO_FULL_RUN_ID is required}"
NODE_RANK="${ROBODOJO_NODE_RANK:-0}"
LANE_ID="${ROBODOJO_LANE_ID:-0}"
SHARD_COUNT="${ROBODOJO_SHARD_COUNT:-1}"
SHARD_INDEX="${ROBODOJO_SHARD_INDEX:-0}"
POLICY_GPU="${ROBODOJO_POLICY_GPU:-0}"
SIM_GPU="${ROBODOJO_SIM_GPU:-0}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-19000}"
ENV_CFG_TYPE="${ROBODOJO_ENV_CFG_TYPE:-arx_x5}"
ACTION_TYPE="${ROBODOJO_ACTION_TYPE:-joint}"
EVAL_NUM="${ROBODOJO_FULL_EVAL_NUM:-native}"
SEED_SPEC="${ROBODOJO_FULL_SEEDS:-0,1,2}"
TASK_SPEC="${ROBODOJO_FULL_TASKS:-}"
FAIL_FAST="${ROBODOJO_FULL_FAIL_FAST:-0}"
DRY_RUN="${ROBODOJO_FULL_DRY_RUN:-0}"
ACTION_CHUNK_SIZE="${ROBODOJO_ACTION_CHUNK_SIZE:-16}"
NUM_INFERENCE_STEPS="${ROBODOJO_NUM_INFERENCE_STEPS:-}"
IMAGE_SIZE="${ROBODOJO_IMAGE_SIZE:-}"
MOSAIC_LAYOUT="${ROBODOJO_MOSAIC_LAYOUT:-auto}"
INVENTORY="${ROBODOJO_ROOT}/scripts/internal/task_inventory.py"
CHECKPOINT_STEP="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
ADDITIONAL_INFO="${ROBODOJO_ADDITIONAL_INFO:-ckpt_name=${CHECKPOINT_STEP},action_type=${ACTION_TYPE}}"
LANE_ROOT="${NODE_ROOT}/lanes/lane_${LANE_ID}"
LANE_CACHE_ROOT="${ROBODOJO_LANE_CACHE_ROOT:-/tmp/ME_U0_robodojo_cache/${RUN_ID}/node_${NODE_RANK}/lane_${LANE_ID}}"
MANIFEST="${LANE_ROOT}/manifest.tsv"
ASSIGNMENTS="${LANE_ROOT}/assignments.tsv"

for required_file in "${CHECKPOINT}" "${CONFIG}" "${STATS}" "${INVENTORY}" \
    "${REPO_ROOT}/scripts/ME_U0/robodojo/eval_robodojo.sh" \
    "${REPO_ROOT}/scripts/ME_U0/robodojo/run_robodojo_policy_server.sh"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[lane ${LANE_ID}] required file is missing: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "[lane ${LANE_ID}] unsafe run id: ${RUN_ID}" >&2
  exit 2
fi
if [[ ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ || ! "${SHARD_INDEX}" =~ ^[0-9]+$ \
      || "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]]; then
  echo "[lane ${LANE_ID}] invalid shard ${SHARD_INDEX}/${SHARD_COUNT}" >&2
  exit 2
fi
if [[ "${EVAL_NUM}" != "native" && ! "${EVAL_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[lane ${LANE_ID}] eval-num must be native or a positive integer" >&2
  exit 2
fi
if [[ ! "${ACTION_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[lane ${LANE_ID}] action-chunk-size must be a positive integer" >&2
  exit 2
fi
if [[ "${MOSAIC_LAYOUT}" != "auto" && "${MOSAIC_LAYOUT}" != "horizontal" && "${MOSAIC_LAYOUT}" != "pyramid" ]]; then
  echo "[lane ${LANE_ID}] mosaic layout must be auto, horizontal or pyramid" >&2
  exit 2
fi
mkdir -p "${LANE_ROOT}/tasks" "${EVAL_ROOT}"
if [[ ! -e "${MANIFEST}" ]]; then
  printf 'timestamp\tseed\ttask\texpected_episodes\tstatus\trc\trun_dir\tresult_json\n' >"${MANIFEST}"
fi

mapfile -t ALL_TASKS < <(cd "${ROBODOJO_ROOT}" && python3 "${INVENTORY}" --only-runnable)
declare -A RUNNABLE=()
for task in "${ALL_TASKS[@]}"; do
  RUNNABLE["${task}"]=1
done
if [[ -z "${TASK_SPEC}" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  task_words="${TASK_SPEC//,/ }"
  read -r -a TASKS <<<"${task_words}"
fi
for task in "${TASKS[@]}"; do
  if [[ -z "${RUNNABLE[${task}]+x}" ]]; then
    echo "[lane ${LANE_ID}] unknown or non-runnable task: ${task}" >&2
    exit 3
  fi
done
seed_words="${SEED_SPEC//,/ }"
read -r -a SEEDS <<<"${seed_words}"
for seed in "${SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[lane ${LANE_ID}] invalid seed: ${seed}" >&2
    exit 3
  fi
done

native_episode_count() {
  case "$1" in
    stack_bowls|stack_bowls_random|push_T|push_T_random|pack_objects_into_box|pack_objects_into_box_random|fold_clothes|fold_clothes_random|hang_mugs|hang_mugs_random|sweep_blocks|sweep_blocks_random|pour_liquid_into_cup|pour_liquid_into_cup_random|make_toast|make_toast_random|arrange_largest_number|arrange_largest_number_random|sort_nesting_dolls_by_size|sort_nesting_dolls_by_size_random|store_laptop_and_headphones|store_laptop_and_headphones_random|stack_blocks|stack_blocks_random)
      echo 25
      ;;
    *)
      echo 50
      ;;
  esac
}

expected_episode_count() {
  local native
  native="$(native_episode_count "$1")"
  if [[ "${EVAL_NUM}" == "native" || "${EVAL_NUM}" -ge "${native}" ]]; then
    echo "${native}"
  else
    echo "${EVAL_NUM}"
  fi
}

result_complete() {
  python3 - "$1" "$2" <<'PY'
import json
import os
import sys

path, expected = sys.argv[1], int(sys.argv[2])
if not os.path.isfile(path):
    raise SystemExit(1)
try:
    result = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
details = result.get("details") or {}
raise SystemExit(0 if int(result.get("eval_time") or 0) >= expected and len(details) >= expected else 1)
PY
}

ASSIGNED_SEEDS=()
ASSIGNED_TASKS=()
ASSIGNED_EXPECTED=()
global_index=0
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    if (( global_index % SHARD_COUNT == SHARD_INDEX )); then
      ASSIGNED_SEEDS+=("${seed}")
      ASSIGNED_TASKS+=("${task}")
      ASSIGNED_EXPECTED+=("$(expected_episode_count "${task}")")
    fi
    global_index=$((global_index + 1))
  done
done

printf 'seed\ttask\texpected_episodes\n' >"${ASSIGNMENTS}"
for idx in "${!ASSIGNED_TASKS[@]}"; do
  printf '%s\t%s\t%s\n' "${ASSIGNED_SEEDS[$idx]}" "${ASSIGNED_TASKS[$idx]}" \
    "${ASSIGNED_EXPECTED[$idx]}" >>"${ASSIGNMENTS}"
done
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'node_rank=%s\n' "${NODE_RANK}"
  printf 'lane_id=%s\n' "${LANE_ID}"
  printf 'shard=%s/%s\n' "${SHARD_INDEX}" "${SHARD_COUNT}"
  printf 'policy_gpu=%s\n' "${POLICY_GPU}"
  printf 'sim_gpu=%s\n' "${SIM_GPU}"
  printf 'policy_port=%s\n' "${POLICY_PORT}"
  printf 'action_chunk_size=%s\n' "${ACTION_CHUNK_SIZE}"
  printf 'num_inference_steps=%s\n' "${NUM_INFERENCE_STEPS}"
  printf 'image_size=%s\n' "${IMAGE_SIZE}"
  printf 'mosaic_layout=%s\n' "${MOSAIC_LAYOUT}"
  printf 'lane_cache_root=%s\n' "${LANE_CACHE_ROOT}"
  printf 'assigned_invocations=%s\n' "${#ASSIGNED_TASKS[@]}"
  printf 'eval_root=%s\n' "${EVAL_ROOT}"
} >"${LANE_ROOT}/lane.env"

echo "[lane ${LANE_ID}] shard=${SHARD_INDEX}/${SHARD_COUNT} assigned=${#ASSIGNED_TASKS[@]} policy_gpu=${POLICY_GPU} sim_gpu=${SIM_GPU}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'dry_run_at=%s\n' "$(date --iso-8601=seconds)" >"${LANE_ROOT}/DRY_RUN"
  exit 0
fi

PENDING_INDICES=()
for idx in "${!ASSIGNED_TASKS[@]}"; do
  seed="${ASSIGNED_SEEDS[$idx]}"
  task="${ASSIGNED_TASKS[$idx]}"
  expected="${ASSIGNED_EXPECTED[$idx]}"
  stamp="${RUN_ID}_seed${seed}_${task}"
  result_json="${EVAL_ROOT}/${task}/ME_U0/${ENV_CFG_TYPE}/${seed}_${ADDITIONAL_INFO}/${stamp}/_result.json"
  if result_complete "${result_json}" "${expected}"; then
    task_run_dir="${LANE_ROOT}/tasks/seed_${seed}/${task}"
    mkdir -p "${task_run_dir}"
    printf '%s\t%s\t%s\t%s\tSKIP_COMPLETE\t0\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${seed}" "${task}" "${expected}" \
      "${task_run_dir}" "${result_json}" >>"${MANIFEST}"
  else
    PENDING_INDICES+=("${idx}")
  fi
done

if [[ ${#PENDING_INDICES[@]} -eq 0 ]]; then
  printf 'completed_at=%s passed=0 failed=0 skipped=%s\n' \
    "$(date --iso-8601=seconds)" "${#ASSIGNED_TASKS[@]}" >"${LANE_ROOT}/COMPLETED"
  echo "[lane ${LANE_ID}] all assigned invocations were already complete"
  exit 0
fi

if python3 - "${POLICY_PORT}" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "[lane ${LANE_ID}] policy port is occupied: ${POLICY_PORT}" >&2
  exit 5
fi

current_task_pid=""

collect_descendants() {
  local parent_pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    collect_descendants "${child_pid}"
    printf '%s\n' "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
}

terminate_task_tree() {
  local root_pid="$1"
  local descendants alive pid
  descendants="$(collect_descendants "${root_pid}")"
  kill -TERM "${root_pid}" 2>/dev/null || true
  if [[ -n "${descendants}" ]]; then
    # shellcheck disable=SC2086
    kill -TERM ${descendants} 2>/dev/null || true
  fi
  for _ in {1..15}; do
    alive=0
    for pid in "${root_pid}" ${descendants}; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [[ "${alive}" == "0" ]] && return
    sleep 1
  done
  kill -KILL "${root_pid}" 2>/dev/null || true
  if [[ -n "${descendants}" ]]; then
    # shellcheck disable=SC2086
    kill -KILL ${descendants} 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${current_task_pid}" ]] && kill -0 "${current_task_pid}" 2>/dev/null; then
    terminate_task_tree "${current_task_pid}"
    wait "${current_task_pid}" 2>/dev/null || true
  fi
  printf 'stopped_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${rc}" >>"${LANE_ROOT}/lane.env"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

passed=0
failed=0
for idx in "${PENDING_INDICES[@]}"; do
  seed="${ASSIGNED_SEEDS[$idx]}"
  task="${ASSIGNED_TASKS[$idx]}"
  expected="${ASSIGNED_EXPECTED[$idx]}"
  stamp="${RUN_ID}_seed${seed}_${task}"
  task_run_dir="${LANE_ROOT}/tasks/seed_${seed}/${task}"
  result_json="${EVAL_ROOT}/${task}/ME_U0/${ENV_CFG_TYPE}/${seed}_${ADDITIONAL_INFO}/${stamp}/_result.json"
  mkdir -p "$(dirname "${task_run_dir}")"
  echo "[lane ${LANE_ID}] START seed=${seed} task=${task} expected=${expected}"
  mkdir -p "${LANE_CACHE_ROOT}"
  printf 'updated_at=%s seed=%s task=%s expected=%s cache=%s\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${task}" "${expected}" \
    "${LANE_CACHE_ROOT}" >"${LANE_ROOT}/heartbeat.txt"

  ROBODOJO_START_POLICY_SERVER=1 \
  ROBODOJO_INSTALL_XPOLICY_CLIENT=0 \
  ROBODOJO_RESUME=1 \
  ROBODOJO_LEAP_CHECKPOINT="${CHECKPOINT}" \
  ROBODOJO_LEAP_CONFIG="${CONFIG}" \
  ROBODOJO_STATS_PATH="${STATS}" \
  ROBODOJO_TASK_NAME="${task}" \
  ROBODOJO_ENV_CFG_TYPE="${ENV_CFG_TYPE}" \
  ROBODOJO_POLICY_PORT="${POLICY_PORT}" \
  ROBODOJO_POLICY_GPU="${POLICY_GPU}" \
  ROBODOJO_SIM_GPU="${SIM_GPU}" \
  ROBODOJO_EVAL_SEED="${seed}" \
  ROBODOJO_POLICY_SEED="${seed}" \
  ROBODOJO_ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE}" \
  ROBODOJO_NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  ROBODOJO_IMAGE_SIZE="${IMAGE_SIZE}" \
  ROBODOJO_MOSAIC_LAYOUT="${MOSAIC_LAYOUT}" \
  ROBODOJO_ACTION_TYPE="${ACTION_TYPE}" \
  ROBODOJO_EVAL_STAMP="${stamp}" \
  ROBODOJO_RUN_DIR="${task_run_dir}" \
  ROBODOJO_ADDITIONAL_INFO="${ADDITIONAL_INFO}" \
  ROBODOJO_EVAL_ROOT="${EVAL_ROOT}" \
  ROBODOJO_LANE_CACHE_ROOT="${LANE_CACHE_ROOT}" \
  EVAL_NUM="${EVAL_NUM}" \
    "${REPO_ROOT}/scripts/ME_U0/robodojo/eval_robodojo.sh" &
  current_task_pid=$!

  set +e
  wait "${current_task_pid}"
  task_rc=$?
  set -e
  current_task_pid=""

  if [[ ${task_rc} -eq 0 ]] && result_complete "${result_json}" "${expected}"; then
    status=PASS
    passed=$((passed + 1))
  else
    status=FAIL
    failed=$((failed + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${task}" "${expected}" \
    "${status}" "${task_rc}" "${task_run_dir}" "${result_json}" >>"${MANIFEST}"
  echo "[lane ${LANE_ID}] ${status} seed=${seed} task=${task} rc=${task_rc}"
  if [[ "${status}" == "FAIL" && "${FAIL_FAST}" == "1" ]]; then
    break
  fi
done

printf 'completed_at=%s passed=%s failed=%s skipped=%s\n' \
  "$(date --iso-8601=seconds)" "${passed}" "${failed}" \
  "$((${#ASSIGNED_TASKS[@]} - ${#PENDING_INDICES[@]}))" >"${LANE_ROOT}/COMPLETED"
echo "[lane ${LANE_ID}] complete passed=${passed} failed=${failed}"
if [[ ${failed} -ne 0 ]]; then
  exit 7
fi
