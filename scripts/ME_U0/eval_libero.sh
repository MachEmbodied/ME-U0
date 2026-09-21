#!/usr/bin/env bash
# ============================================================
# LIBERO evaluation for MachEmbodiedUnifiedModel — complete server + client workflow.
# (copy of scripts/eval_libero.sh; server = scripts/ME_U0/run_policy_server.sh)
#
# Modes:
#   1. Auto mode (--config + --checkpoint): starts policy server,
#      waits for it, runs eval client, cleans up. All-in-one.
#   2. Client-only mode (--server host:port): connects to an
#      existing policy server (backward compatible).
#
# Usage (auto mode, single suite):
#   bash scripts/ME_U0/eval_libero.sh \
#       --config leap/configs/experiments/libero_posttraining.yaml \
#       --checkpoint /path/to/mp_rank_00_model_states.pt \
#       --task-suite libero_goal --server-gpu 0 --sim-gpus 4
#
# Usage (auto mode, all 4 suites in parallel):
# cd /path/to/leap_posttrain_open
# bash scripts/ME_U0/eval_libero.sh \
#   --config leap/configs/experiments/libero_posttraining.yaml \
#   --checkpoint /path/to/mp_rank_00_model_states.pt \
#   --task-suite all \
#   --server-gpu 0,1,2,3 --sim-gpus 4,5,6,7 \
#   --image-size 256 \
#   --num-inference-steps 24 \
#   --action-chunk-size 8 \

# Usage (client-only, connect to existing server):
#   bash scripts/ME_U0/eval_libero.sh \
#       --server localhost:8765 --task-suite libero_goal --sim-gpus 4
# ============================================================
set -euo pipefail

# -------------------- defaults --------------------
CONFIG=""
CHECKPOINT=""
SERVER=""
TASK_SUITE="libero_goal"
SERVER_GPUS="0"
SIM_GPUS="4"
BASE_PORT=8765
EPISODES=50
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
PER_VIEW_SIZE=""            # override serving mosaic resolution; leave empty to auto-read from config's per_view_size
GEN_FUTURE_VIDEO=1          # save one model-generated future video per task (set 0 to disable)
NORMALIZER_SOURCE=""
MAX_TASKS=""
EVAL_NAME="${EVAL_NAME:-eval_libero}"
export LIBERO_UNIFIED_ROOT="${LIBERO_UNIFIED_ROOT:-/path/to/unified_eval}"
CONDA_LIBERO_PY="${CONDA_LIBERO_PY:-${LIBERO_UNIFIED_ROOT}/conda/libero_py38/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-${LIBERO_UNIFIED_ROOT}/source/LIBERO}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"

# -------------------- parse --------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --server)       SERVER="$2";       shift 2 ;;
        --task-suite)   TASK_SUITE="$2";   shift 2 ;;
        --server-gpu)   SERVER_GPUS="$2";  shift 2 ;;
        --sim-gpus)     SIM_GPUS="$2";     shift 2 ;;
        --base-port)    BASE_PORT="$2";    shift 2 ;;
        --episodes)     EPISODES="$2";     shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        --output)       OUTPUT="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --video-dir)    VIDEO_DIR="$2";    shift 2 ;;
        --image-size)   IMAGE_SIZE="$2";   shift 2 ;;
        --action-chunk-size) ACTION_CHUNK_SIZE="$2"; shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-wait-ms)  MAX_WAIT_MS="$2";  shift 2 ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --per-view-size) PER_VIEW_SIZE="$2"; shift 2 ;;
        --gen-future-video)    GEN_FUTURE_VIDEO=1; shift ;;
        --no-gen-future-video) GEN_FUTURE_VIDEO=0; shift ;;
        --normalizer-source) NORMALIZER_SOURCE="$2"; shift 2 ;;
        --max-tasks)    MAX_TASKS="$2";    shift 2 ;;
        --python)       CONDA_LIBERO_PY="$2"; shift 2 ;;
        --libero-home)  LIBERO_HOME="$2";  shift 2 ;;
        --mujoco-gl)    MUJOCO_GL_BACKEND="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/ME_U0/eval_libero.sh [options]

Auto mode (starts server automatically):
  --config PATH             Training config YAML
  --checkpoint PATH         Model checkpoint
  --server-gpu IDS          GPU ids for policy servers (default: 0)
                            For --task-suite all, provide e.g. "0,1,2,3"
  --base-port N             Starting port (default: 8765)

Client-only mode (connect to existing server):
  --server host:port        Policy server address (e.g. localhost:8765)

Common:
  --task-suite NAME|all     libero_spatial / libero_object / libero_goal /
                            libero_10 / libero_90 / all (default: libero_goal)
                            "all" runs spatial+object+goal+10 in parallel.
  --sim-gpus IDS            GPU ids for MuJoCo rendering (default: 4)
                            For --task-suite all, provide e.g. "4,5,6,7"
  --episodes N              Rollouts per task (default: 50)
  --workers N               Parallel sim workers per suite (default: 8)
  --output PATH             Results JSON path (single suite, client-only mode)
  --output-dir DIR          Results directory (auto mode, default: auto)
  --video-dir DIR           Save per-episode videos here
  --image-size N            Camera frame size (default: 224)
  --action-chunk-size N     Actions per server call (default: full chunk)
  --max-batch-size N        Server batch size (default: 8)
  --max-wait-ms MS          Server batch wait time in ms (default: 10)
  --num-inference-steps N   Joint-denoise Euler steps (default: 24 from config)
  --no-gen-future-video     Disable saving model-generated future videos (default: on)
  --normalizer-source NAME  Force normalizer source key (e.g. libero_plus)
  --max-tasks N             Debug only: evaluate first N tasks (default: all)
  --seed N                  Simulator RNG seed; also policy sampling in auto mode
                            (default: 42)
  --mujoco-gl BACKEND       "egl" (default) or "osmesa"

Env:
  --python PATH             libero conda python
  --libero-home PATH        LIBERO source root
EOF
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# -------------------- locate repo root --------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# -------------------- common env setup --------------------
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_UNIFIED_ROOT}/config/$(basename "${LIBERO_HOME}")}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL_BACKEND}"

# -------------------- EGL dependency check --------------------
# MuJoCo EGL backend needs libEGL.so.1 on the system. NVIDIA drivers
# ship libEGL_nvidia.so but not the mesa dispatch library. If missing,
# try to install it automatically (common in fresh Docker containers).
if [[ "${MUJOCO_GL_BACKEND}" == "egl" ]] && ! ldconfig -p 2>/dev/null | grep -q "libEGL.so.1"; then
    echo "==> libEGL.so.1 not found, installing libegl-dev..."
    apt-get update -qq && apt-get install -y -qq libegl-dev 2>/dev/null || \
        echo "WARNING: failed to install libegl-dev. MuJoCo EGL rendering may fail."
fi
export PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND}"
export MUJOCO_RENDERER="${MUJOCO_GL_BACKEND}"

# ================================================================
# CLIENT-ONLY MODE: --server is set, no --config
# ================================================================
if [[ -n "${SERVER}" && -z "${CONFIG}" ]]; then
    export LEAP_SIM_GPUS="${SIM_GPUS}"

    echo "==> [client-only] server: ${SERVER}"
    echo "==> task_suite:  ${TASK_SUITE}"
    echo "==> episodes:    ${EPISODES}"
    echo "==> workers:     ${WORKERS}"
    echo "==> sim_gpus:    ${SIM_GPUS}"
    [[ -n "${NORMALIZER_SOURCE}" ]] && echo "==> normalizer:  ${NORMALIZER_SOURCE}"
    [[ -n "${MAX_TASKS}" ]] && echo "==> max_tasks:   ${MAX_TASKS}"
    echo ""

    CMD=("${CONDA_LIBERO_PY}" -m sim_eval
         --env libero
         --task-suite "${TASK_SUITE}"
         --server "${SERVER}"
         --episodes "${EPISODES}"
         --workers "${WORKERS}"
         --image-size "${IMAGE_SIZE}"
         --seed "${SEED}")
    [[ -n "${ACTION_CHUNK_SIZE}" ]] && CMD+=(--action-chunk-size "${ACTION_CHUNK_SIZE}")
    [[ -n "${VIDEO_DIR}" ]] && CMD+=(--video-dir "${VIDEO_DIR}")
    [[ -n "${OUTPUT}" ]]   && CMD+=(--output "${OUTPUT}")
    [[ -n "${NORMALIZER_SOURCE}" ]] && CMD+=(--normalizer-source "${NORMALIZER_SOURCE}")
    [[ -n "${MAX_TASKS}" ]] && CMD+=(--max-tasks "${MAX_TASKS}")

    exec "${CMD[@]}"
fi

# ================================================================
# AUTO MODE: --config + --checkpoint, manage server lifecycle
# ================================================================
if [[ -z "${CONFIG}" ]]; then echo "ERROR: --config is required (or use --server for client-only mode)" >&2; exit 2; fi
if [[ -z "${CHECKPOINT}" ]]; then echo "ERROR: --checkpoint is required" >&2; exit 2; fi

# -------------------- output dir --------------------
if [[ -z "${OUTPUT_DIR}" ]]; then
    # Place eval results parallel to checkpoints/ using the same step/epoch name.
    # e.g. .../work_dir/checkpoints/step_10000/pytorch_model/mp_rank_00_model_states.pt
    #   → .../work_dir/eval_libero/step_10000/
    CKPT_STEP_DIR="${CHECKPOINT%/}"
    [[ -f "${CKPT_STEP_DIR}" ]] && CKPT_STEP_DIR="$(dirname "${CKPT_STEP_DIR}")"
    [[ "$(basename "${CKPT_STEP_DIR}")" == "pytorch_model" ]] && CKPT_STEP_DIR="$(dirname "${CKPT_STEP_DIR}")"
    CKPT_NAME="$(basename "${CKPT_STEP_DIR}")"                # step_10000
    WORK_DIR="$(dirname "$(dirname "${CKPT_STEP_DIR}")")"     # .../work_dir
    OUTPUT_DIR="${WORK_DIR}/${EVAL_NAME}/${CKPT_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

# -------------------- resolve task suites --------------------
if [[ "${TASK_SUITE}" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
else
    SUITES=("${TASK_SUITE}")
fi

IFS=', ' read -ra SRV_GPU_ARR <<< "${SERVER_GPUS}"
IFS=', ' read -ra SIM_GPU_ARR <<< "${SIM_GPUS}"
N_SUITES=${#SUITES[@]}

echo "============================================================"
echo " LEAP LIBERO Evaluation"
echo "============================================================"
echo "  config:       ${CONFIG}"
echo "  checkpoint:   ${CHECKPOINT}"
echo "  suites:       ${SUITES[*]}"
echo "  server GPUs:  ${SERVER_GPUS}"
echo "  sim GPUs:     ${SIM_GPUS}"
echo "  base port:    ${BASE_PORT}"
echo "  episodes:     ${EPISODES}"
echo "  workers:      ${WORKERS}"
[[ -n "${NORMALIZER_SOURCE}" ]] && echo "  normalizer:   ${NORMALIZER_SOURCE}"
[[ -n "${MAX_TASKS}" ]] && echo "  max_tasks:    ${MAX_TASKS}"
echo "  output dir:   ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# -------------------- cleanup on exit --------------------
SERVER_PIDS=()
cleanup() {
    echo ""
    echo "==> Cleaning up server processes..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "    killing server PID $pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "==> Done."
}
trap cleanup EXIT

# -------------------- run one suite --------------------
run_suite() {
    local suite="$1"
    local srv_gpu="$2"
    local sim_gpu="$3"
    local port="$4"

    local suite_dir="${OUTPUT_DIR}/${suite}"
    local log_dir="${suite_dir}/logs"
    mkdir -p "${log_dir}"

    # Start policy server
    echo "==> [${suite}] Starting policy server on GPU ${srv_gpu}, port ${port}..."
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
    [[ "${GEN_FUTURE_VIDEO}" == "1" ]] && srv_cmd+=(--video-out-dir "${suite_dir}/gen_videos")
    "${srv_cmd[@]}" > "${log_dir}/server.log" 2>&1 &
    local srv_pid=$!
    SERVER_PIDS+=("${srv_pid}")
    echo "==> [${suite}] Server PID: ${srv_pid}, log: ${log_dir}/server.log"

    # Wait for server to be ready
    echo "==> [${suite}] Waiting for server to be ready..."
    local waited=0
    local max_wait=600
    while ! grep -q "waiting for connections" "${log_dir}/server.log" 2>/dev/null; do
        if ! kill -0 "${srv_pid}" 2>/dev/null; then
            echo "ERROR: [${suite}] Server process died. Check ${log_dir}/server.log"
            return 1
        fi
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "ERROR: [${suite}] Server not ready after ${max_wait}s. Check ${log_dir}/server.log"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [[ $((waited % 30)) -eq 0 ]]; then
            echo "==> [${suite}] Still waiting... (${waited}s)"
        fi
    done
    echo "==> [${suite}] Server ready! (took ${waited}s)"

    # Run eval client. Pin the sim process to its own GPU: MuJoCo's EGL backend
    # needs CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID to know which GPU to create
    # the render context on — without them EGL tries GPU 0 (a policy-server GPU) and
    # aborts at the first env render (C-level "Aborted" at step 0). Mirrors the
    # working eval_libero_plus.sh sim launch.
    export LEAP_SIM_GPUS="${sim_gpu}"
    export CUDA_VISIBLE_DEVICES="${sim_gpu}"
    export MUJOCO_EGL_DEVICE_ID="${sim_gpu}"
    local eval_cmd=("${CONDA_LIBERO_PY}" -m sim_eval
        --env libero
        --task-suite "${suite}"
        --server "localhost:${port}"
        --episodes "${EPISODES}"
        --workers "${WORKERS}"
        --image-size "${IMAGE_SIZE}"
        --seed "${SEED}"
        --output "${suite_dir}/results.json")
    [[ -n "${ACTION_CHUNK_SIZE}" ]] && eval_cmd+=(--action-chunk-size "${ACTION_CHUNK_SIZE}")
    [[ -n "${NORMALIZER_SOURCE}" ]] && eval_cmd+=(--normalizer-source "${NORMALIZER_SOURCE}")
    [[ -n "${MAX_TASKS}" ]] && eval_cmd+=(--max-tasks "${MAX_TASKS}")
    local vdir="${VIDEO_DIR:-${suite_dir}/videos}"
    mkdir -p "${vdir}"
    eval_cmd+=(--video-dir "${vdir}")

    echo "==> [${suite}] Starting evaluation..."
    "${eval_cmd[@]}" 2>&1 | tee "${log_dir}/eval.log"
    local rc=${PIPESTATUS[0]}

    # Kill server
    if kill -0 "${srv_pid}" 2>/dev/null; then
        kill "${srv_pid}" 2>/dev/null || true
        wait "${srv_pid}" 2>/dev/null || true
    fi

    if [[ ${rc} -eq 0 ]]; then
        echo "==> [${suite}] Done! Results: ${suite_dir}/results.json"
    else
        echo "==> [${suite}] FAILED (exit code ${rc}). Check ${log_dir}/eval.log"
    fi
    return ${rc}
}

# -------------------- launch --------------------
if [[ ${N_SUITES} -eq 1 ]]; then
    run_suite "${SUITES[0]}" "${SRV_GPU_ARR[0]}" "${SIM_GPU_ARR[0]}" "${BASE_PORT}"
else
    SUITE_PIDS=()
    for i in "${!SUITES[@]}"; do
        suite="${SUITES[$i]}"
        srv_gpu="${SRV_GPU_ARR[$((i % ${#SRV_GPU_ARR[@]}))]}"
        sim_gpu="${SIM_GPU_ARR[$((i % ${#SIM_GPU_ARR[@]}))]}"
        port=$((BASE_PORT + i))

        run_suite "${suite}" "${srv_gpu}" "${sim_gpu}" "${port}" &
        SUITE_PIDS+=($!)

        # Short stagger to avoid disk I/O spike from loading 4 models simultaneously
        if [[ $i -lt $((N_SUITES - 1)) ]]; then
            sleep 5
        fi
    done

    echo ""
    echo "==> All ${N_SUITES} suites launched. Waiting for completion..."
    FAILED=0
    for i in "${!SUITE_PIDS[@]}"; do
        wait "${SUITE_PIDS[$i]}" || FAILED=$((FAILED + 1))
    done

    echo ""
    echo "============================================================"
    echo " All evaluations finished. (${FAILED}/${N_SUITES} failed)"
    echo " Results in: ${OUTPUT_DIR}/"
    echo "============================================================"
    if [[ ${FAILED} -gt 0 ]]; then exit 1; fi
fi
