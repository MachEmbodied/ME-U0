#!/usr/bin/env bash
# PPU wrapper for ME_U0 training. Run the same command on every node.
# It stages the repository and all mutable runtime caches on node-local /tmp,
# while launcher logs, torchrun logs, checkpoints, and artifacts stay in WORK_ROOT.
set -euo pipefail

NNODES=""
NPROC_PER_NODE=""
SHARED_REPO=""
EXP_NAME=""
WORK_ROOT=""
TRAIN_CONFIG=""
RESUME=""
ATTEMPT_ID="${ME_U0_ATTEMPT_ID:-}"
OVERRIDES=()

die() {
  echo "[train_ppu] ERROR: $*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a non-empty value"
}

usage() {
  cat <<'EOF'
Usage:
  MASTER_PORT=29900 \
  bash scripts/ME_U0/train_ppu.sh \
    --nnodes 1 \
    --nproc-per-node 16 \
    --shared-repo /path/to/leap_posttrain_open \
    --exp-name libero_h24_posttrain \
    --work-root /path/to/work_dirs \
    --train-config leap/configs/experiments/libero_posttraining.yaml
    [--resume PATH|latest|auto] [--attempt-id ID] [key=value ...]

Required parameters have no user-specific defaults. Run the identical command
on every node. NODE_RANK/RANK and MASTER_ADDR are read when available; otherwise
the node rank is derived from the hostname and rank 0 publishes its IP through
the shared experiment directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes) need_value "$@"; NNODES="$2"; shift 2 ;;
    --nproc-per-node) need_value "$@"; NPROC_PER_NODE="$2"; shift 2 ;;
    --shared-repo) need_value "$@"; SHARED_REPO="$2"; shift 2 ;;
    --exp-name) need_value "$@"; EXP_NAME="$2"; shift 2 ;;
    --work-root) need_value "$@"; WORK_ROOT="$2"; shift 2 ;;
    --train-config|--config) need_value "$@"; TRAIN_CONFIG="$2"; shift 2 ;;
    --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
    --attempt-id) need_value "$@"; ATTEMPT_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *=*) OVERRIDES+=("$1"); shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_name in NNODES NPROC_PER_NODE SHARED_REPO EXP_NAME WORK_ROOT TRAIN_CONFIG; do
  [[ -n "${!required_name}" ]] || die "missing required parameter ${required_name}"
done
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
NNODES=$((10#${NNODES}))
NPROC_PER_NODE=$((10#${NPROC_PER_NODE}))
[[ "${EXP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
  die "EXP_NAME must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
[[ "${TRAIN_CONFIG}" != /* ]] || die "TRAIN_CONFIG must be relative to SHARED_REPO"

[[ -d "${SHARED_REPO}" ]] || die "shared repo does not exist: ${SHARED_REPO}"
SHARED_REPO="$(cd "${SHARED_REPO}" && pwd -P)"
for required_path in pyproject.toml leap scripts scripts/install.sh scripts/ME_U0/run_multinode.sh; do
  [[ -e "${SHARED_REPO}/${required_path}" ]] || \
    die "incomplete shared repo; missing ${SHARED_REPO}/${required_path}"
done
[[ -e "${SHARED_REPO}/.git" ]] || die "shared repo must include .git provenance"
[[ -f "${SHARED_REPO}/${TRAIN_CONFIG}" ]] || \
  die "training config not found in shared repo: ${TRAIN_CONFIG}"
mkdir -p "${WORK_ROOT}"
WORK_ROOT="$(cd "${WORK_ROOT}" && pwd -P)"

NODE_RANK_VALUE="${NODE_RANK:-${RANK:-}}"
if [[ -z "${NODE_RANK_VALUE}" ]]; then
  CURRENT_HOST="${HOSTNAME:-$(hostname)}"
  if [[ "${CURRENT_HOST}" =~ -([0-9]+)$ ]]; then
    NODE_RANK_VALUE="${BASH_REMATCH[1]}"
  elif (( NNODES == 1 )); then
    NODE_RANK_VALUE=0
  fi
fi
[[ "${NODE_RANK_VALUE}" =~ ^[0-9]+$ ]] || \
  die "unable to derive node rank; set NODE_RANK/RANK"
NODE_RANK_VALUE=$((10#${NODE_RANK_VALUE}))
(( NODE_RANK_VALUE < NNODES )) || \
  die "node rank must be in [0, ${NNODES}), got ${NODE_RANK_VALUE}"
if [[ -z "${ATTEMPT_ID}" ]]; then
  CURRENT_HOST="${HOSTNAME:-$(hostname)}"
  NODE_SUFFIX="-${NODE_RANK_VALUE}"
  ATTEMPT_ID="${CURRENT_HOST%"${NODE_SUFFIX}"}"
fi
[[ "${ATTEMPT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
  die "ATTEMPT_ID must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"

# PPU pods become unreliable when /tmp is nearly full. This threshold remains
# configurable, but all user/job-specific paths are required CLI parameters.
TMP_USAGE_ABORT_GB="${TMP_USAGE_ABORT_GB:-32}"
[[ "${TMP_USAGE_ABORT_GB}" =~ ^[1-9][0-9]*$ ]] || \
  die "TMP_USAGE_ABORT_GB must be a positive integer"
TMP_USED_KIB="$(du -skx /tmp 2>/dev/null | awk 'NR == 1 {print $1}')"
[[ "${TMP_USED_KIB:-}" =~ ^[0-9]+$ ]] || die "unable to measure /tmp usage"
TMP_ABORT_KIB=$((10#${TMP_USAGE_ABORT_GB} * 1024 * 1024))
(( TMP_USED_KIB < TMP_ABORT_KIB )) || \
  die "/tmp already uses $((TMP_USED_KIB / 1024 / 1024)) GiB; refusing to start"

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}" || true)"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || die "no usable Python interpreter"
export PYTHON="${PYTHON_BIN}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

LOCAL_JOB_ROOT="$(mktemp -d "/tmp/ME_U0-train.node_${NODE_RANK_VALUE}.XXXXXX")" || \
  die "failed to create node-local job directory"
LOCAL_REPO="${LOCAL_JOB_ROOT}/repo"
RUNTIME_TMP_ROOT=""
LOCAL_STATE_CLEANED=0

cleanup_local_state() {
  local original_rc=$?
  if [[ "${LOCAL_STATE_CLEANED}" == "1" ]]; then
    return "${original_rc}"
  fi
  LOCAL_STATE_CLEANED=1
  if [[ "${KEEP_LOCAL_TRAIN_CACHE:-0}" == "1" ]]; then
    echo "[train_ppu] preserving node-local state: ${LOCAL_JOB_ROOT}"
    [[ -z "${RUNTIME_TMP_ROOT}" ]] || \
      echo "[train_ppu] preserving torchrun state: ${RUNTIME_TMP_ROOT}"
  else
    case "${LOCAL_JOB_ROOT}" in
      /tmp/ME_U0-train.node_*) rm -rf -- "${LOCAL_JOB_ROOT}" ;;
      *) echo "[train_ppu] WARNING: refused unexpected cleanup path: ${LOCAL_JOB_ROOT}" >&2 ;;
    esac
    if [[ -n "${RUNTIME_TMP_ROOT}" ]]; then
      case "${RUNTIME_TMP_ROOT}" in
        /tmp/ME_U0/*/n*) rm -rf -- "${RUNTIME_TMP_ROOT}" ;;
        *) echo "[train_ppu] WARNING: refused unexpected cleanup path: ${RUNTIME_TMP_ROOT}" >&2 ;;
      esac
    fi
  fi
  return "${original_rc}"
}
trap cleanup_local_state EXIT

mkdir -p "${LOCAL_REPO}" "${LOCAL_JOB_ROOT}/tmp" \
  "${LOCAL_JOB_ROOT}/pip-cache" "${LOCAL_JOB_ROOT}/pycache"
STAGED_ITEMS=(pyproject.toml README.md leap scripts .git)
[[ ! -e "${SHARED_REPO}/.gitignore" ]] || STAGED_ITEMS+=(.gitignore)
if ! tar -C "${SHARED_REPO}" \
    --exclude='*/__pycache__' \
    --exclude='*/__pycache__/*' \
    --exclude='*.py[co]' \
    -cf - "${STAGED_ITEMS[@]}" | tar -C "${LOCAL_REPO}" -xf -; then
  die "failed to stage training code from ${SHARED_REPO}"
fi

export TMPDIR="${LOCAL_JOB_ROOT}/tmp"
export PIP_CACHE_DIR="${LOCAL_JOB_ROOT}/pip-cache"
export PYTHONPYCACHEPREFIX="${LOCAL_JOB_ROOT}/pycache"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LEAP=0.0.0
export PIP_DISABLE_PIP_VERSION_CHECK=1
export ME_U0_COMPILER_CACHE_NODE_ROOT="${LOCAL_JOB_ROOT}/compiler_cache"
export LEAP_WORK_ROOT="${WORK_ROOT}"
export NODE_RANK="${NODE_RANK_VALUE}"
export ME_U0_ATTEMPT_ID="${ATTEMPT_ID}"
# LPAI can inject port 2222 for SSH forwarding; do not use it for rendezvous.
export MASTER_PORT="${MASTER_PORT:-29900}"

CANONICAL_WORK_DIR="${WORK_ROOT}/${EXP_NAME}"
TMP_RUN_KEY="$("${PYTHON_BIN}" - "${CANONICAL_WORK_DIR}" "${ATTEMPT_ID}" <<'PY'
import hashlib
import sys
print(hashlib.sha256(":".join(sys.argv[1:]).encode("utf-8")).hexdigest()[:12])
PY
)"
RUNTIME_TMP_ROOT="/tmp/ME_U0/${TMP_RUN_KEY}/n${NODE_RANK_VALUE}"

TRAIN_LOG="${CANONICAL_WORK_DIR}/launcher_logs/${ATTEMPT_ID}/train_ppu_node_${NODE_RANK_VALUE}_$(hostname -s)_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${TRAIN_LOG}")"
echo "[train_ppu] node=${NODE_RANK_VALUE}/${NNODES} nproc_per_node=${NPROC_PER_NODE}"
echo "[train_ppu] attempt_id=${ATTEMPT_ID} resume=${RESUME:-none}"
echo "[train_ppu] shared_repo=${SHARED_REPO}"
echo "[train_ppu] local_repo=${LOCAL_REPO}"
echo "[train_ppu] config=${TRAIN_CONFIG}"
echo "[train_ppu] output=${CANONICAL_WORK_DIR}"
echo "[train_ppu] launcher_log=${TRAIN_LOG}"

cd "${LOCAL_REPO}"
LAUNCH_ARGS=(
  "${NNODES}"
  --nproc-per-node "${NPROC_PER_NODE}"
  --exp-name "${EXP_NAME}"
  --attempt-id "${ATTEMPT_ID}"
  --work-root "${WORK_ROOT}"
  --config "${TRAIN_CONFIG}"
)
[[ -z "${RESUME}" ]] || LAUNCH_ARGS+=(--resume "${RESUME}")
LAUNCH_ARGS+=("${OVERRIDES[@]}")

set +e
(
  bash scripts/install.sh &&
  MASTER_ADDR="${MASTER_ADDR:-}" NODE_RANK="${NODE_RANK_VALUE}" \
    bash scripts/ME_U0/run_multinode.sh "${LAUNCH_ARGS[@]}"
) 2>&1 | tee "${TRAIN_LOG}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

cd /tmp
cleanup_local_state
trap - EXIT

echo "[train_ppu] training ended rc=${TRAIN_RC}; log=${TRAIN_LOG}"
if (( TRAIN_RC != 0 )); then
  tail -n 40 "${TRAIN_LOG}" 2>/dev/null | sed 's/^/    /'
fi
exit "${TRAIN_RC}"
