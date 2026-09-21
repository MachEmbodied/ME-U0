#!/usr/bin/env bash
# ============================================================
# LIBERO-plus evaluation for MachEmbodiedUnifiedModel.
# Drives the ME_U0 policy server (scripts/ME_U0/run_policy_server.sh) and the
# MachEmbodiedUnifiedModel-aware
# scripts/ME_U0/eval_libero.sh in the SHARDS_PER_SUITE=1 fast path.
#
# Official LIBERO-plus uses the same LIBERO evaluation path with one trial per
# task. ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE selects the checkpoint normalizer family:
#   libero      -> original LIBERO-trained checkpoints, use per-suite keys
#   libero_plus -> LIBERO-plus-trained checkpoints, use the single libero_plus key
#
# pgrep -af 'scripts/ME_U0/serve.py'
# pkill -f 'scripts/ME_U0/serve.py'
#
# sharded usage (N shards/suite; one policy server per shard):
# cd /path/to/leap_posttrain_open
# bash scripts/install.sh && bash scripts/ME_U0/eval_libero_plus.sh \
#   --libero-plus-shards-per-suite 2 \
#   --config leap/configs/experiments/libero_posttraining.yaml \
#   --checkpoint /path/to/mp_rank_00_model_states.pt \
#   --task-suite all \
#   --server-gpu 0,1,2,3,4,5,6,7 \
#   --sim-gpus 0,1,2,3,4,5,6,7 \
#   --num-inference-steps 24 --action-chunk-size 8
#
# libero zero shot plus usage:
# cd /path/to/leap_posttrain_open
# bash scripts/install.sh && ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE=libero bash scripts/ME_U0/eval_libero_plus.sh \
#   --libero-plus-shards-per-suite 1 \
#   --config leap/configs/experiments/libero_posttraining.yaml \
#   --checkpoint /path/to/mp_rank_00_model_states.pt \
#   --task-suite all --server-gpu 0,1,2,3 --sim-gpus 4,5,6,7 --num-inference-steps 24 --action-chunk-size 16
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LIBERO_UNIFIED_ROOT="${LIBERO_UNIFIED_ROOT:-/path/to/unified_eval}"
export LIBERO_HOME="${LIBERO_HOME:-${LIBERO_UNIFIED_ROOT}/source/LIBERO-plus}"
export CONDA_LIBERO_PY="${CONDA_LIBERO_PY:-${LIBERO_UNIFIED_ROOT}/conda/libero_py38/bin/python}"
export EVAL_NAME="${EVAL_NAME:-eval_libero_plus}"

# Keep the legacy variable as a read-only fallback so existing evaluation jobs
# continue to work while all newly documented ME_U0 controls share one prefix.
NORMALIZER_MODE="${ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE:-${LIBERO_PLUS_NORMALIZER_SOURCE:-libero_plus}}"
NORMALIZER_ARGS=()
case "${NORMALIZER_MODE}" in
    libero)
        ;;
    libero_plus)
        NORMALIZER_ARGS=(--normalizer-source libero_plus)
        ;;
    *)
        echo "ERROR: ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE must be 'libero' or 'libero_plus', got '${NORMALIZER_MODE}'" >&2
        exit 2
        ;;
esac

SHARDS_PER_SUITE="${ME_U0_LIBERO_PLUS_SHARDS_PER_SUITE:-${LIBERO_PLUS_SHARDS_PER_SUITE:-1}}"
DRY_RUN="${ME_U0_LIBERO_PLUS_DRY_RUN:-${LIBERO_PLUS_DRY_RUN:-0}}"
NUM_NODES="${ME_U0_LIBERO_PLUS_NUM_NODES:-1}"
NODE_RANK="${ME_U0_LIBERO_PLUS_NODE_RANK:-0}"
USER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --libero-plus-shards-per-suite|--shards-per-suite)
            SHARDS_PER_SUITE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            USER_ARGS+=("$1")
            shift
            ;;
    esac
done

FORWARD_ARGS=(
    --episodes "${ME_U0_LIBERO_PLUS_EPISODES:-${LIBERO_PLUS_EPISODES:-1}}"
    --libero-home "${LIBERO_HOME}"
)
if [[ ${#NORMALIZER_ARGS[@]} -gt 0 ]]; then
    FORWARD_ARGS+=("${NORMALIZER_ARGS[@]}")
fi
FORWARD_ARGS+=("${USER_ARGS[@]}")

if ! [[ "${SHARDS_PER_SUITE}" =~ ^[0-9]+$ ]] || [[ "${SHARDS_PER_SUITE}" -lt 1 ]]; then
    echo "ERROR: --libero-plus-shards-per-suite must be a positive integer, got '${SHARDS_PER_SUITE}'" >&2
    exit 2
fi

if [[ "${SHARDS_PER_SUITE}" -eq 1 && "${NUM_NODES}" -eq 1 ]]; then
    exec bash "${SCRIPT_DIR}/eval_libero.sh" "${FORWARD_ARGS[@]}"
fi

CONFIG=""
CHECKPOINT=""
SERVER=""
TASK_SUITE="libero_goal"
SERVER_GPUS="0"
SIM_GPUS="4"
BASE_PORT=8765
EPISODES="${ME_U0_LIBERO_PLUS_EPISODES:-${LIBERO_PLUS_EPISODES:-1}}"
WORKERS=1
OUTPUT=""
OUTPUT_DIR=""
VIDEO_DIR=""
IMAGE_SIZE=224
ACTION_CHUNK_SIZE=""
SEED=42
MAX_BATCH_SIZE=""
MAX_WAIT_MS=""
NUM_INFERENCE_STEPS=""
PER_VIEW_SIZE=""            # override serving mosaic resolution; empty -> auto-read from config's per_view_size
GEN_FUTURE_VIDEO=1          # save one model-generated future video per task (set 0 to disable)
NORMALIZER_SOURCE=""
MAX_TASKS=""
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"

parse_eval_args() {
    local argv=("$@")
    local i=0
    while [[ ${i} -lt ${#argv[@]} ]]; do
        case "${argv[$i]}" in
            --config)       i=$((i + 1)); CONFIG="${argv[$i]:-}" ;;
            --checkpoint)   i=$((i + 1)); CHECKPOINT="${argv[$i]:-}" ;;
            --server)       i=$((i + 1)); SERVER="${argv[$i]:-}" ;;
            --task-suite)   i=$((i + 1)); TASK_SUITE="${argv[$i]:-}" ;;
            --server-gpu)   i=$((i + 1)); SERVER_GPUS="${argv[$i]:-}" ;;
            --sim-gpus)     i=$((i + 1)); SIM_GPUS="${argv[$i]:-}" ;;
            --base-port)    i=$((i + 1)); BASE_PORT="${argv[$i]:-}" ;;
            --episodes)     i=$((i + 1)); EPISODES="${argv[$i]:-}" ;;
            --workers)      i=$((i + 1)); WORKERS="${argv[$i]:-}" ;;
            --output)       i=$((i + 1)); OUTPUT="${argv[$i]:-}" ;;
            --output-dir)   i=$((i + 1)); OUTPUT_DIR="${argv[$i]:-}" ;;
            --video-dir)    i=$((i + 1)); VIDEO_DIR="${argv[$i]:-}" ;;
            --image-size)   i=$((i + 1)); IMAGE_SIZE="${argv[$i]:-}" ;;
            --action-chunk-size) i=$((i + 1)); ACTION_CHUNK_SIZE="${argv[$i]:-}" ;;
            --seed)         i=$((i + 1)); SEED="${argv[$i]:-}" ;;
            --max-batch-size) i=$((i + 1)); MAX_BATCH_SIZE="${argv[$i]:-}" ;;
            --max-wait-ms)  i=$((i + 1)); MAX_WAIT_MS="${argv[$i]:-}" ;;
            --num-inference-steps) i=$((i + 1)); NUM_INFERENCE_STEPS="${argv[$i]:-}" ;;
            --per-view-size) i=$((i + 1)); PER_VIEW_SIZE="${argv[$i]:-}" ;;
            --gen-future-video)    GEN_FUTURE_VIDEO=1 ;;
            --no-gen-future-video) GEN_FUTURE_VIDEO=0 ;;
            --normalizer-source) i=$((i + 1)); NORMALIZER_SOURCE="${argv[$i]:-}" ;;
            --max-tasks)    i=$((i + 1)); MAX_TASKS="${argv[$i]:-}" ;;
            --python)       i=$((i + 1)); CONDA_LIBERO_PY="${argv[$i]:-}" ;;
            --libero-home)  i=$((i + 1)); LIBERO_HOME="${argv[$i]:-}" ;;
            --mujoco-gl)    i=$((i + 1)); MUJOCO_GL_BACKEND="${argv[$i]:-}" ;;
        esac
        i=$((i + 1))
    done
}

split_list() {
    local raw="$1"
    local -n out_arr="$2"
    raw="${raw//,/ }"
    read -r -a out_arr <<< "${raw}"
}

parse_eval_args "${FORWARD_ARGS[@]}"

if [[ -n "${SERVER}" ]]; then
    echo "ERROR: sharded LIBERO-plus mode starts one policy server per shard; --server client-only mode is not supported." >&2
    exit 2
fi
if [[ -z "${CONFIG}" ]]; then
    echo "ERROR: --config is required for sharded LIBERO-plus mode" >&2
    exit 2
fi
if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: --checkpoint is required for sharded LIBERO-plus mode" >&2
    exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    CKPT_STEP_DIR="${CHECKPOINT%/}"
    [[ -f "${CKPT_STEP_DIR}" ]] && CKPT_STEP_DIR="$(dirname "${CKPT_STEP_DIR}")"
    [[ "$(basename "${CKPT_STEP_DIR}")" == "pytorch_model" ]] && CKPT_STEP_DIR="$(dirname "${CKPT_STEP_DIR}")"
    CKPT_NAME="$(basename "${CKPT_STEP_DIR}")"
    WORK_DIR="$(dirname "$(dirname "${CKPT_STEP_DIR}")")"
    OUTPUT_DIR="${WORK_DIR}/${EVAL_NAME}/${CKPT_NAME}"
fi

if [[ "${TASK_SUITE}" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
else
    SUITES=("${TASK_SUITE}")
fi

split_list "${SERVER_GPUS}" SRV_GPU_ARR
split_list "${SIM_GPUS}" SIM_GPU_ARR
if [[ ${#SRV_GPU_ARR[@]} -eq 0 ]]; then
    echo "ERROR: --server-gpu must contain at least one GPU id" >&2
    exit 2
fi
if [[ ${#SIM_GPU_ARR[@]} -eq 0 ]]; then
    echo "ERROR: --sim-gpus must contain at least one GPU id" >&2
    exit 2
fi

TOTAL_SLOTS=$((${#SUITES[@]} * SHARDS_PER_SUITE))

echo "============================================================"
echo " MachEmbodiedUnifiedModel LIBERO-plus Sharded Evaluation"
echo "============================================================"
echo "  config:            ${CONFIG}"
echo "  checkpoint:        ${CHECKPOINT}"
echo "  suites:            ${SUITES[*]}"
echo "  shards per suite:  ${SHARDS_PER_SUITE}"
echo "  total slots:       ${TOTAL_SLOTS}"
echo "  server GPUs:       ${SERVER_GPUS}"
echo "  sim GPUs:          ${SIM_GPUS}"
echo "  base port:         ${BASE_PORT}"
echo "  episodes:          ${EPISODES}"
echo "  workers:           ${WORKERS}"
[[ -n "${NUM_INFERENCE_STEPS}" ]] && echo "  infer steps:       ${NUM_INFERENCE_STEPS}"
[[ -n "${NORMALIZER_SOURCE}" ]] && echo "  normalizer:        ${NORMALIZER_SOURCE}"
[[ -n "${MAX_TASKS}" ]] && echo "  max_tasks:         ${MAX_TASKS}"
echo "  output dir:        ${OUTPUT_DIR}"
echo "============================================================"
echo ""

slot_gpu() {
    local slot_idx="$1"
    local -n arr="$2"
    echo "${arr[$((slot_idx % ${#arr[@]}))]}"
}

print_slot_plan() {
    local slot_idx=0
    local suite shard srv_gpu sim_gpu port local_slot
    for suite in "${SUITES[@]}"; do
        for ((shard = 0; shard < SHARDS_PER_SUITE; shard++)); do
            if (( slot_idx % NUM_NODES != NODE_RANK )); then
                slot_idx=$((slot_idx + 1))
                continue
            fi
            local_slot=$((slot_idx / NUM_NODES))
            srv_gpu="$(slot_gpu "${local_slot}" SRV_GPU_ARR)"
            sim_gpu="$(slot_gpu "${local_slot}" SIM_GPU_ARR)"
            port=$((BASE_PORT + local_slot))
            printf '[slot %02d] suite=%s shard=%d/%d server_gpu=%s sim_gpu=%s port=%d\n' \
                "${slot_idx}" "${suite}" "${shard}" "${SHARDS_PER_SUITE}" "${srv_gpu}" "${sim_gpu}" "${port}"
            slot_idx=$((slot_idx + 1))
        done
    done
}

print_slot_plan
echo ""

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "==> Dry run only; no servers or simulators started."
    exit 0
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_UNIFIED_ROOT}/config/$(basename "${LIBERO_HOME}")}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL_BACKEND}"
export PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND}"
export MUJOCO_RENDERER="${MUJOCO_GL_BACKEND}"

if [[ "${MUJOCO_GL_BACKEND}" == "egl" ]] && ! ldconfig -p 2>/dev/null | grep -q "libEGL.so.1"; then
    echo "==> libEGL.so.1 not found, installing libegl-dev..."
    apt-get update -qq && apt-get install -y -qq libegl-dev 2>/dev/null || \
        echo "WARNING: failed to install libegl-dev. MuJoCo EGL rendering may fail."
fi

mkdir -p "${OUTPUT_DIR}"

run_shard() (
    set -euo pipefail
    local suite="$1"
    local shard_idx="$2"
    local shard_count="$3"
    local srv_gpu="$4"
    local sim_gpu="$5"
    local port="$6"
    local slot_idx="$7"

    local suite_dir="${OUTPUT_DIR}/${suite}"
    local shard_dir="${suite_dir}/shards/shard_${shard_idx}_of_${shard_count}"
    local log_dir="${suite_dir}/logs"
    local vdir
    if [[ -n "${VIDEO_DIR}" ]]; then
        vdir="${VIDEO_DIR}/${suite}"
    else
        vdir="${suite_dir}/videos"
    fi
    mkdir -p "${shard_dir}" "${log_dir}" "${vdir}"

    local server_log="${log_dir}/shard_${shard_idx}_server.log"
    local eval_log="${log_dir}/shard_${shard_idx}_eval.log"
    local shard_result="${shard_dir}/results.json"
    local srv_pid=""

    cleanup_shard() {
        if [[ -n "${srv_pid}" ]] && kill -0 "${srv_pid}" 2>/dev/null; then
            kill "${srv_pid}" 2>/dev/null || true
            wait "${srv_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_shard EXIT

    echo "==> [slot ${slot_idx} ${suite} shard ${shard_idx}/${shard_count}] Starting MachEmbodiedUnifiedModel policy server on GPU ${srv_gpu}, port ${port}..."
    local srv_cmd=(bash scripts/ME_U0/run_policy_server.sh
        --config "${CONFIG}"
        --checkpoint "${CHECKPOINT}"
        --port "${port}"
        --device cuda
        --gpu "${srv_gpu}")
    [[ -n "${MAX_BATCH_SIZE}" ]] && srv_cmd+=(--max-batch-size "${MAX_BATCH_SIZE}")
    [[ -n "${MAX_WAIT_MS}" ]] && srv_cmd+=(--max-wait-ms "${MAX_WAIT_MS}")
    [[ -n "${NUM_INFERENCE_STEPS}" ]] && srv_cmd+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
    [[ -n "${PER_VIEW_SIZE}" ]] && srv_cmd+=(--per-view-size "${PER_VIEW_SIZE}")
    srv_cmd+=(--seed "${SEED}")
    [[ "${GEN_FUTURE_VIDEO}" == "1" ]] && srv_cmd+=(--video-out-dir "${shard_dir}/gen_videos")
    "${srv_cmd[@]}" > "${server_log}" 2>&1 &
    srv_pid=$!

    local waited=0
    local max_wait=600
    while ! grep -q "waiting for connections" "${server_log}" 2>/dev/null; do
        if ! kill -0 "${srv_pid}" 2>/dev/null; then
            echo "ERROR: [slot ${slot_idx} ${suite} shard ${shard_idx}] Server process died. Check ${server_log}" >&2
            return 1
        fi
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "ERROR: [slot ${slot_idx} ${suite} shard ${shard_idx}] Server not ready after ${max_wait}s. Check ${server_log}" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "==> [slot ${slot_idx} ${suite} shard ${shard_idx}/${shard_count}] Server ready. Starting eval on sim GPU ${sim_gpu}..."

    set +e
    LEAP_SIM_GPUS="${sim_gpu}" \
    CUDA_VISIBLE_DEVICES="${sim_gpu}" \
    MUJOCO_EGL_DEVICE_ID="${sim_gpu}" \
    LEAP_LIBERO_SHARD_SUITE="${suite}" \
    LEAP_LIBERO_SHARD_INDEX="${shard_idx}" \
    LEAP_LIBERO_SHARD_COUNT="${shard_count}" \
    LEAP_LIBERO_PORT="${port}" \
    LEAP_LIBERO_EPISODES="${EPISODES}" \
    LEAP_LIBERO_WORKERS="${WORKERS}" \
    LEAP_LIBERO_IMAGE_SIZE="${IMAGE_SIZE}" \
    LEAP_LIBERO_SEED="${SEED}" \
    LEAP_LIBERO_ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE}" \
    LEAP_LIBERO_VIDEO_DIR="${vdir}" \
    LEAP_LIBERO_OUTPUT="${shard_result}" \
    LEAP_LIBERO_NORMALIZER_SOURCE="${NORMALIZER_SOURCE}" \
    LEAP_LIBERO_MAX_TASKS="${MAX_TASKS}" \
    "${CONDA_LIBERO_PY}" - <<'PY' 2>&1 | tee "${eval_log}"
import json
import logging
import os
from pathlib import Path

from sim_eval.client import PolicyClient
from sim_eval.libero_env import LiberoEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

suite = os.environ["LEAP_LIBERO_SHARD_SUITE"]
shard_idx = int(os.environ["LEAP_LIBERO_SHARD_INDEX"])
shard_count = int(os.environ["LEAP_LIBERO_SHARD_COUNT"])
port = int(os.environ["LEAP_LIBERO_PORT"])
episodes_per_task = int(os.environ["LEAP_LIBERO_EPISODES"])
workers = int(os.environ["LEAP_LIBERO_WORKERS"])
image_size = int(os.environ["LEAP_LIBERO_IMAGE_SIZE"])
seed = int(os.environ["LEAP_LIBERO_SEED"])
action_chunk_raw = os.environ.get("LEAP_LIBERO_ACTION_CHUNK_SIZE", "").strip()
action_chunk_size = int(action_chunk_raw) if action_chunk_raw else None
video_dir = os.environ.get("LEAP_LIBERO_VIDEO_DIR", "").strip() or None
output = Path(os.environ["LEAP_LIBERO_OUTPUT"])
normalizer_source = os.environ.get("LEAP_LIBERO_NORMALIZER_SOURCE", "").strip() or None
max_tasks_raw = os.environ.get("LEAP_LIBERO_MAX_TASKS", "").strip()
max_tasks = int(max_tasks_raw) if max_tasks_raw else 0

sim_env = LiberoEnv(
    task_suite=suite,
    image_size=image_size,
    action_chunk_size=action_chunk_size,
    normalizer_source=normalizer_source,
)
n_tasks = sim_env.task_count()
base_task_ids = list(range(min(max_tasks, n_tasks))) if max_tasks > 0 else list(range(n_tasks))
task_ids = [task_id for task_id in base_task_ids if task_id % shard_count == shard_idx]
logging.info("LIBERO-plus shard: suite=%s shard=%d/%d tasks=%d/%d", suite, shard_idx, shard_count, len(task_ids), len(base_task_ids))
if task_ids:
    logging.info("Task id range sample: first=%s last=%s", task_ids[:8], task_ids[-8:])

rollouts = [(task_id, ep_idx) for task_id in task_ids for ep_idx in range(episodes_per_task)]
client = PolicyClient("localhost", port)
try:
    if workers > 1:
        results_list = sim_env._evaluate_multiprocess(client, rollouts, workers, video_dir, seed)
    else:
        results_list = sim_env._evaluate_single(client, rollouts, video_dir, seed)
    results = sim_env._aggregate_results(results_list, n_tasks, episodes_per_task, task_ids=task_ids)
finally:
    client.close()

per_task_by_id = {int(task_id): [] for task_id in task_ids}
for result in results_list:
    per_task_by_id.setdefault(int(result.task_id), []).append(bool(result.success))
results["per_task_success_rate_by_id"] = {
    str(task_id): (sum(successes) / len(successes) if successes else 0.0)
    for task_id, successes in sorted(per_task_by_id.items())
}
descs = sim_env.task_descriptions()
if descs is not None and len(descs) == n_tasks:
    results["task_names_by_id"] = {str(task_id): descs[task_id] for task_id in task_ids}

results["shard_index"] = shard_idx
results["shard_count"] = shard_count
results["task_suite"] = suite
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nShard results saved to {output}")
PY
    local rc=${PIPESTATUS[0]}
    set -e

    cleanup_shard
    trap - EXIT

    if [[ ${rc} -eq 0 ]]; then
        echo "==> [slot ${slot_idx} ${suite} shard ${shard_idx}/${shard_count}] Done. Results: ${shard_result}"
    else
        echo "==> [slot ${slot_idx} ${suite} shard ${shard_idx}/${shard_count}] FAILED (exit code ${rc}). Check ${eval_log}" >&2
    fi
    return ${rc}
)

merge_suite_results() {
    local suite="$1"
    local suite_dir="${OUTPUT_DIR}/${suite}"
    "${CONDA_LIBERO_PY}" - "${suite_dir}" "${SHARDS_PER_SUITE}" <<'PY'
import json
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
shard_count = int(sys.argv[2])
paths = [suite_dir / "shards" / f"shard_{i}_of_{shard_count}" / "results.json" for i in range(shard_count)]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit("missing shard result(s): " + ", ".join(missing))

by_task_id = {}
legacy_items = []
all_task_ids = set()
merge_warnings = []
total_episodes = 0
num_tasks_evaluated = 0
episodes_per_task = None
num_tasks_total = None
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("task_ids")
    if ids is not None:
        all_task_ids.update(int(task_id) for task_id in ids)

    rates_by_id = data.get("per_task_success_rate_by_id")
    names_by_id = data.get("task_names_by_id", {})
    if isinstance(rates_by_id, dict):
        for task_id_raw, rate in rates_by_id.items():
            task_id = int(task_id_raw)
            name = names_by_id.get(str(task_id), f"task_{task_id:03d}")
            by_task_id[task_id] = (name, float(rate))
            all_task_ids.add(task_id)
    else:
        items = list(data.get("per_task_success_rate", {}).items())
        if ids is None:
            ids = list(range(len(items)))
        if len(ids) == len(items):
            for task_id, (name, rate) in zip(ids, items):
                task_id = int(task_id)
                by_task_id[task_id] = (name, float(rate))
                all_task_ids.add(task_id)
        else:
            merge_warnings.append(
                f"legacy shard {path} has {len(ids)} task_ids but "
                f"{len(items)} per_task entries; duplicate task names were "
                "already collapsed before merge"
            )
            for name, rate in items:
                legacy_items.append((name, float(rate), path.parent.name))

    total_episodes += int(data.get("total_episodes", 0))
    num_tasks_evaluated += int(data.get("num_tasks_evaluated", len(ids) if ids is not None else len(items)))
    episodes_per_task = data.get("num_episodes_per_task", episodes_per_task)
    num_tasks_total = data.get("num_tasks_total", num_tasks_total)

def add_unique(per_task, name, rate, suffix):
    key = name
    if key in per_task:
        key = f"{name} [{suffix}]"
        i = 2
        while key in per_task:
            key = f"{name} [{suffix} #{i}]"
            i += 1
    per_task[key] = rate

per_task = {}
for task_id in sorted(by_task_id):
    name, rate = by_task_id[task_id]
    add_unique(per_task, name, rate, f"task_id={task_id}")
for name, rate, source in legacy_items:
    add_unique(per_task, name, rate, f"{source} legacy")

overall = sum(per_task.values()) / len(per_task) if per_task else 0.0
per_task_by_id = {str(task_id): by_task_id[task_id][1] for task_id in sorted(by_task_id)}
merged = {
    "overall_success_rate": overall,
    "per_task_success_rate": per_task,
    "per_task_success_rate_by_id": per_task_by_id,
    "num_episodes_per_task": episodes_per_task,
    "total_episodes": total_episodes,
    "num_tasks_evaluated": num_tasks_evaluated if num_tasks_evaluated else len(per_task),
    "num_tasks_total": num_tasks_total,
    "sharded": True,
    "shards_per_suite": shard_count,
    "task_ids": sorted(all_task_ids) if all_task_ids else sorted(by_task_id),
}
if merge_warnings:
    merged["merge_warnings"] = merge_warnings
out = suite_dir / "results.json"
out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"Merged {len(paths)} shard results into {out}")
for warning in merge_warnings:
    print(f"WARNING: {warning}", file=sys.stderr)
print(f"Overall success rate: {overall:.1%}")
PY
}

if (( NUM_NODES > 1 )); then
    mkdir -p "${OUTPUT_DIR}/nodes/node_${NODE_RANK}"
    rm -f "${OUTPUT_DIR}/nodes/node_${NODE_RANK}/shards.exit_code"
fi

SLOT_PIDS=()
slot_idx=0
for suite in "${SUITES[@]}"; do
    for ((shard = 0; shard < SHARDS_PER_SUITE; shard++)); do
        if (( slot_idx % NUM_NODES != NODE_RANK )); then
            slot_idx=$((slot_idx + 1))
            continue
        fi
        local_slot=$((slot_idx / NUM_NODES))
        srv_gpu="$(slot_gpu "${local_slot}" SRV_GPU_ARR)"
        sim_gpu="$(slot_gpu "${local_slot}" SIM_GPU_ARR)"
        port=$((BASE_PORT + local_slot))
        run_shard "${suite}" "${shard}" "${SHARDS_PER_SUITE}" "${srv_gpu}" "${sim_gpu}" "${port}" "${slot_idx}" &
        SLOT_PIDS+=("$!")
        slot_idx=$((slot_idx + 1))
        if [[ ${slot_idx} -lt ${TOTAL_SLOTS} ]]; then
            sleep 5
        fi
    done
done

echo ""
echo "==> Node ${NODE_RANK}/${NUM_NODES}: ${#SLOT_PIDS[@]} LIBERO-plus shard slots launched. Waiting for completion..."
FAILED=0
for pid in "${SLOT_PIDS[@]}"; do
    wait "${pid}" || FAILED=$((FAILED + 1))
done

if (( NUM_NODES > 1 )); then
    node_status="${OUTPUT_DIR}/nodes/node_${NODE_RANK}/shards.exit_code"
    printf '%s\n' "${FAILED}" > "${node_status}.tmp"
    mv "${node_status}.tmp" "${node_status}"
    if (( NODE_RANK != 0 )); then
        exit "$((FAILED > 0))"
    fi
    FAILED=0
    for ((node=0; node<NUM_NODES; node++)); do
        node_status="${OUTPUT_DIR}/nodes/node_${node}/shards.exit_code"
        while [[ ! -f "${node_status}" ]]; do sleep 5; done
        FAILED=$((FAILED + $(<"${node_status}")))
    done
fi

MERGE_FAILED=0
if [[ ${FAILED} -eq 0 ]]; then
    for suite in "${SUITES[@]}"; do
        merge_suite_results "${suite}" || MERGE_FAILED=$((MERGE_FAILED + 1))
        if [[ -n "${OUTPUT}" && ${#SUITES[@]} -eq 1 ]]; then
            cp "${OUTPUT_DIR}/${suite}/results.json" "${OUTPUT}"
        fi
    done
else
    echo "WARNING: skipping result merge because ${FAILED}/${TOTAL_SLOTS} shard slot(s) failed." >&2
fi

echo ""
echo "============================================================"
echo " LIBERO-plus sharded evaluations finished. (${FAILED}/${TOTAL_SLOTS} slots failed, ${MERGE_FAILED}/${#SUITES[@]} merges failed)"
echo " Results in: ${OUTPUT_DIR}/"
echo "============================================================"
if [[ ${FAILED} -gt 0 || ${MERGE_FAILED} -gt 0 ]]; then exit 1; fi
