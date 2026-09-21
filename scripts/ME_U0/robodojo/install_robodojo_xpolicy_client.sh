#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIFIED_ROOT="${ROBODOJO_UNIFIED_ROOT:-/path/to/unified_eval}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${UNIFIED_ROOT}/source/RoboDojo-25691aa78fb3}"
SOURCE_DIR="${REPO_ROOT}/scripts/ME_U0/robodojo/robodojo_xpolicy_client/ME_U0"
TARGET_DIR="${ROBODOJO_ROOT}/XPolicyLab/policy/ME_U0"
ACTION_TYPE="${ROBODOJO_ACTION_TYPE:-joint}"

if [[ "${ACTION_TYPE}" != "joint" && "${ACTION_TYPE}" != "ee" ]]; then
  echo "ROBODOJO_ACTION_TYPE must be joint or ee, got: ${ACTION_TYPE}" >&2
  exit 2
fi

test -d "${ROBODOJO_ROOT}/XPolicyLab/policy"
test -f "${SOURCE_DIR}/deploy.py"
mkdir -p "${TARGET_DIR}"
python3 - "${SOURCE_DIR}" "${TARGET_DIR}" "${ACTION_TYPE}" <<'PY'
import os
import sys
import tempfile

import yaml

source_dir, target_dir, action_type = sys.argv[1:]

for name in ("__init__.py", "deploy.py", "deploy.yml"):
    source = os.path.join(source_dir, name)
    target = os.path.join(target_dir, name)
    if name == "deploy.yml":
        with open(source, encoding="utf-8") as handle:
            deploy = yaml.safe_load(handle) or {}
        deploy["action_type"] = action_type
        content = yaml.safe_dump(deploy, sort_keys=False).encode()
    else:
        with open(source, "rb") as handle:
            content = handle.read()

    try:
        with open(target, "rb") as handle:
            if handle.read() == content:
                continue
    except FileNotFoundError:
        pass

    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
PY

echo "ME_U0 XPolicy client installed at ${TARGET_DIR} (action_type=${ACTION_TYPE})"
sha256sum "${TARGET_DIR}/deploy.py" "${TARGET_DIR}/deploy.yml"
