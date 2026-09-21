#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIFIED_ROOT="${ROBODOJO_UNIFIED_ROOT:-/path/to/unified_eval}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${UNIFIED_ROOT}/source/RoboDojo-25691aa78fb3}"
SIM_RUNNER="${UNIFIED_ROOT}/scripts/run_robodojo_env.sh"
TASK_NAME="${ROBODOJO_TASK_NAME:-stack_blocks}"
ENV_CFG_TYPE="${ROBODOJO_ENV_CFG_TYPE:-arx_x5}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-19000}"
POLICY_GPU="${ROBODOJO_POLICY_GPU:-0}"
SIM_GPU="${ROBODOJO_SIM_GPU:-1}"
SEED="${ROBODOJO_EVAL_SEED:-0}"
EVAL_NUM="${EVAL_NUM:-1}"
START_POLICY_SERVER="${ROBODOJO_START_POLICY_SERVER:-1}"
RESUME="${ROBODOJO_RESUME:-1}"
STAMP="${ROBODOJO_EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROBODOJO_RUN_DIR:-${UNIFIED_ROOT}/results/${STAMP}_me_u0_step20000_${TASK_NAME}}"
POLICY_LOG="${RUN_DIR}/policy_server.log"
SIM_LOG="${RUN_DIR}/simulator.log"
ADDITIONAL_INFO="${ROBODOJO_ADDITIONAL_INFO:-me_u0_step20000_${STAMP}}"
EVAL_ROOT="${ROBODOJO_EVAL_ROOT:-${ROBODOJO_ROOT}/eval_result/RoboDojo}"
OFFICIAL_RESULT_DIR="${EVAL_ROOT}/${TASK_NAME}/ME_U0/${ENV_CFG_TYPE}/${SEED}_${ADDITIONAL_INFO}/${STAMP}"
OFFICIAL_RESULT_JSON="${OFFICIAL_RESULT_DIR}/_result.json"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "[eval] ROBODOJO_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME}" == "0" && ( -e "${RUN_DIR}" || -e "${OFFICIAL_RESULT_DIR}" ) ]]; then
  echo "[eval] run stamp already exists and resume is disabled: ${STAMP}" >&2
  exit 4
fi
mkdir -p "${RUN_DIR}"
if [[ "${ROBODOJO_INSTALL_XPOLICY_CLIENT:-1}" == "1" ]]; then
  "${REPO_ROOT}/scripts/ME_U0/robodojo/install_robodojo_xpolicy_client.sh"
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_POLICY_SERVER}" == "1" ]]; then
  echo "[eval] starting ME_U0 policy server task=${TASK_NAME} seed=${SEED} on GPU ${POLICY_GPU}, port ${POLICY_PORT}"
  printf '\n[eval] policy_server_started_at=%s task=%s seed=%s\n' \
    "$(date --iso-8601=seconds)" "${TASK_NAME}" "${SEED}" >>"${POLICY_LOG}"
  ROBODOJO_POLICY_GPU="${POLICY_GPU}" \
  ROBODOJO_POLICY_PORT="${POLICY_PORT}" \
  ROBODOJO_POLICY_SEED="${SEED}" \
  ROBODOJO_TASK_NAME="${TASK_NAME}" \
    "${REPO_ROOT}/scripts/ME_U0/robodojo/run_robodojo_policy_server.sh" \
    >>"${POLICY_LOG}" 2>&1 &
  server_pid=$!
  echo "${server_pid}" >"${RUN_DIR}/policy_server.pid"
fi

echo "[eval] waiting for ws://127.0.0.1:${POLICY_PORT}"
POLICY_SERVER_PID="${server_pid}" python3 - "${POLICY_PORT}" <<'PY'
import os
import socket
import sys
import time

port = int(sys.argv[1])
pid = int(os.environ.get("POLICY_SERVER_PID") or 0)
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit("policy server exited before opening its port")
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            print(f"policy port {port} is ready")
            break
    time.sleep(2)
else:
    raise SystemExit(f"timed out waiting for policy port {port}")
PY

echo "[eval] task=${TASK_NAME} env=${ENV_CFG_TYPE} sim_gpu=${SIM_GPU} eval_num=${EVAL_NUM}"
sim_log_offset=0
if [[ -f "${SIM_LOG}" ]]; then
  sim_log_offset="$(stat -c '%s' "${SIM_LOG}")"
fi
printf '\n[eval] attempt_started_at=%s task=%s seed=%s stamp=%s\n' \
  "$(date --iso-8601=seconds)" "${TASK_NAME}" "${SEED}" "${STAMP}" >>"${SIM_LOG}"
set +e
EVAL_NUM="${EVAL_NUM}" ROBODOJO_RUN_ID="${STAMP}" ROBODOJO_EVAL_ROOT="${EVAL_ROOT}" \
  "${SIM_RUNNER}" bash "${ROBODOJO_ROOT}/scripts/eval_policy.sh" \
    --root_dir "${ROBODOJO_ROOT}" \
    --task_name "${TASK_NAME}" \
    --env_cfg_type "${ENV_CFG_TYPE}" \
    --device_id "${SIM_GPU}" \
    --policy_name ME_U0 \
    --port "${POLICY_PORT}" \
    --host 127.0.0.1 \
    --protocol ws \
    --additional_info "${ADDITIONAL_INFO}" \
    --seed "${SEED}" \
    >>"${SIM_LOG}" 2>&1
eval_rc=$?
set -e

# A successful shell status alone is insufficient: Kit/Python failures have
# historically been swallowed by nested launchers. Require the benchmark's
# authoritative result JSON and turn a Python traceback into a hard failure.
if grep -q "Traceback (most recent call last)" \
    <(tail -c "+$((sim_log_offset + 1))" "${SIM_LOG}"); then
  echo "[eval] Python traceback detected in simulator log" >&2
  eval_rc=2
fi
if [[ "${eval_rc}" -eq 0 && ! -s "${OFFICIAL_RESULT_JSON}" ]]; then
  echo "[eval] missing authoritative result: ${OFFICIAL_RESULT_JSON}" >&2
  eval_rc=3
fi
if [[ -s "${OFFICIAL_RESULT_JSON}" ]]; then
  cp -f "${OFFICIAL_RESULT_JSON}" "${RUN_DIR}/result.json"
  python3 - "${RUN_DIR}/result.json" "${RUN_DIR}/result_summary.txt" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
summary = (
    f"eval_time={result.get('eval_time')} "
    f"success_rate={result.get('success_rate')} "
    f"score={result.get('score')}\n"
)
open(sys.argv[2], "w", encoding="utf-8").write(summary)
PY
fi
echo "${eval_rc}" >"${RUN_DIR}/eval.rc"
printf '%s\n' "${OFFICIAL_RESULT_JSON}" >"${RUN_DIR}/official_result_path.txt"
if [[ -s "${RUN_DIR}/result_summary.txt" ]]; then
  echo "[eval] $(cat "${RUN_DIR}/result_summary.txt")" || true
fi
echo "[eval] rc=${eval_rc}; logs=${RUN_DIR}" || true
if [[ "${eval_rc}" -ne 0 ]]; then
  tail -n 100 "${POLICY_LOG}" 2>/dev/null || true
  tail -n 200 "${SIM_LOG}" 2>/dev/null || true
fi
exit "${eval_rc}"
