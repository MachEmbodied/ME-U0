#!/usr/bin/env bash
# ============================================================
#  project 开发环境安装脚本
#
#  用法:
#    bash scripts/install.sh                  # 安装全部依赖 (editable)
#    bash scripts/install.sh --venv .venv     # 在 venv 中安装
#    bash scripts/install.sh --profile ppu     # 使用 PPU 镜像自带 runtime
#    bash scripts/install.sh --profile nvidia  # 保留镜像 runtime，仅补 XPolicyLab WS
#    bash scripts/install.sh --wheelhouse DIR  # 优先使用持久 wheelhouse
#    bash scripts/install.sh --with-causal-conv1d  # (可选) 从源码编译 causal-conv1d
#    bash scripts/install.sh --with-flash-attn     # 从源码编译 flash-attn
#
#  核心依赖 (pyproject.toml dependencies):
#    transformers, torch, accelerate, deepspeed, datasets 等
#  可选依赖:
#    diffusers, dev (pytest+ruff)
#  可选 CUDA 扩展 (需手动安装):
#    causal-conv1d, flash-attn
# ============================================================
set -euo pipefail

# Ubuntu 24.04 marks /usr/bin/python3 as EXTERNALLY-MANAGED (PEP 668).  LPAI
# containers are intentionally provisioned by this script on every recreation,
# so allow pip to manage the container's system site-packages.  This only
# bypasses PEP 668; dependency bounds below still protect torch/numpy.
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

# -------------------- 默认配置 --------------------
USE_VENV=""
EXTRAS=""
PROFILE="${LEAP_INSTALL_PROFILE:-auto}"
WHEELHOUSE="${LEAP_WHEELHOUSE:-}"
OFFLINE="${LEAP_INSTALL_OFFLINE:-false}"
FORCE_CAUSAL_CONV1D=false
FORCE_FLASH_ATTN=false

# -------------------- 解析参数 --------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            USE_VENV="${2:-.venv}"
            shift 2 ;;
        --extras)
            EXTRAS="$2"
            shift 2 ;;
        --profile)
            PROFILE="${2:-}"
            shift 2 ;;
        --wheelhouse)
            WHEELHOUSE="${2:-}"
            shift 2 ;;
        --offline)
            OFFLINE=true
            shift ;;
        --with-causal-conv1d)
            FORCE_CAUSAL_CONV1D=true
            shift ;;
        --with-flash-attn)
            FORCE_FLASH_ATTN=true
            shift ;;
        -h|--help)
            echo "用法: bash scripts/install.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --venv [DIR]           创建并使用虚拟环境 (默认 .venv)"
            echo "  --profile NAME         auto|ppu|nvidia|5880 (默认: auto)"
            echo "  --extras NAMES         安装额外可选依赖 (逗号分隔, 如 diffusion,dev)"
            echo "  --wheelhouse DIR       通过 --find-links 优先使用本地/持久 wheel"
            echo "  --offline              与 --wheelhouse 配合，完全禁用 package index"
            echo "  --with-causal-conv1d   (可选) 从源码编译 causal-conv1d"
            echo "  --with-flash-attn      (可选) 从源码编译 flash-attn"
            echo ""
            echo "核心依赖 (pip install -e . 自动安装):"
            echo "  torch, transformers, accelerate, deepspeed,"
            echo "  omegaconf, tensorboard, datasets 等"
            echo ""
            echo "可选 CUDA 扩展 (需手动安装或使用上述选项):"
            echo "  causal-conv1d, flash-attn"
            echo ""
            echo "可选依赖组:"
            echo "  xpolicy      - XPolicyLab WebSocket/observation runtime (NVIDIA profile)"
            echo "  diffusion  - diffusers 扩散模型"
            echo "  dev        - 开发工具 (pytest, ruff)"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# -------------------- 定位项目根目录 --------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "==> 项目根目录: ${PROJECT_ROOT}"

# -------------------- Python 版本检查 --------------------
PYTHON=${PYTHON:-python3}
if ! command -v "${PYTHON}" &>/dev/null; then
    echo "错误: 未找到 ${PYTHON}，请先安装 Python >= 3.10"
    exit 1
fi

PY_VERSION=$("${PYTHON}" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
PY_MAJOR=$("${PYTHON}" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("${PYTHON}" -c "import sys; print(sys.version_info.minor)")

if [[ ${PY_MAJOR} -lt 3 ]] || [[ ${PY_MAJOR} -eq 3 && ${PY_MINOR} -lt 10 ]]; then
    echo "错误: 需要 Python >= 3.10，当前版本: ${PY_VERSION}"
    exit 1
fi
echo "==> Python 版本: ${PY_VERSION}"

# -------------------- 虚拟环境 (可选) --------------------
if [[ -n "${USE_VENV}" ]]; then
    if [[ ! -d "${USE_VENV}" ]]; then
        echo "==> 创建虚拟环境: ${USE_VENV}"
        "${PYTHON}" -m venv "${USE_VENV}"
    else
        echo "==> 虚拟环境已存在: ${USE_VENV}"
    fi
    echo "==> 激活虚拟环境"
    source "${USE_VENV}/bin/activate"
    PYTHON=python
fi

# -------------------- 安装 profile --------------------
if [[ "${PROFILE}" == "auto" ]]; then
    if [[ -n "${PPU_SDK:-}" ]] || \
        command -v ppu-smi &>/dev/null || \
        [[ -x /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi ]]; then
        PROFILE="ppu"
    elif command -v nvidia-smi &>/dev/null; then
        PROFILE="nvidia"
    else
        echo "错误: 无法自动识别 PPU/NVIDIA 环境，请传 --profile ppu|nvidia"
        exit 1
    fi
fi

case "${PROFILE}" in
    ppu)
        PROFILE_EXTRAS=""
        ;;
    nvidia|5880)
        PROFILE="nvidia"
        PROFILE_EXTRAS="xpolicy"
        ;;
    *)
        echo "错误: --profile 仅支持 auto|ppu|nvidia|5880，当前: ${PROFILE}"
        exit 1
        ;;
esac

INSTALL_EXTRAS="${PROFILE_EXTRAS}"
if [[ -n "${EXTRAS}" ]]; then
    if [[ -n "${INSTALL_EXTRAS}" ]]; then
        INSTALL_EXTRAS="${INSTALL_EXTRAS},${EXTRAS}"
    else
        INSTALL_EXTRAS="${EXTRAS}"
    fi
fi

echo "==> 安装 profile: ${PROFILE}"
echo "==> Python executable: $(command -v "${PYTHON}")"
echo "==> PIP_BREAK_SYSTEM_PACKAGES=${PIP_BREAK_SYSTEM_PACKAGES}"
echo "==> Extras: ${INSTALL_EXTRAS:-<none>}"

# -------------------- pip source / offline wheelhouse --------------------
PIP_SOURCE_ARGS=()
if [[ -n "${WHEELHOUSE}" ]]; then
    if [[ ! -d "${WHEELHOUSE}" ]]; then
        echo "错误: wheelhouse 不存在: ${WHEELHOUSE}"
        exit 1
    fi
    WHEELHOUSE="$(cd "${WHEELHOUSE}" && pwd)"
    PIP_SOURCE_ARGS+=(--find-links "${WHEELHOUSE}")
    echo "==> pip wheelhouse: ${WHEELHOUSE}"
fi

case "${OFFLINE,,}" in
    1|true|yes|on)
        if [[ -z "${WHEELHOUSE}" ]]; then
            echo "错误: --offline/LEAP_INSTALL_OFFLINE=1 必须同时提供 --wheelhouse"
            exit 1
        fi
        PIP_SOURCE_ARGS+=(--no-index)
        OFFLINE=true
        ;;
    0|false|no|off|"")
        OFFLINE=false
        ;;
    *)
        echo "错误: LEAP_INSTALL_OFFLINE 必须是 0/1/true/false，当前: ${OFFLINE}"
        exit 1
        ;;
esac
echo "==> pip offline: ${OFFLINE}"

pip_install() {
    "${PYTHON}" -m pip install "${PIP_SOURCE_ARGS[@]}" "$@"
}

# NVIDIA NGC 容器可能通过 PIP_CONSTRAINT 锁定 pyarrow/dill/torch；Ubuntu
# system Python 则由上方 PIP_BREAK_SYSTEM_PACKAGES 处理 PEP 668。两者都要
# 在第一个 pip 调用（包括 pip 自身升级）之前处理。
if [[ -n "${PIP_CONSTRAINT:-}" ]]; then
    echo "==> 检测到 PIP_CONSTRAINT=${PIP_CONSTRAINT}，安装时临时解除"
    unset PIP_CONSTRAINT
fi

# -------------------- 系统依赖 --------------------
APT_UPDATED=false
apt_install_missing() {
    if ! command -v apt-get &>/dev/null; then
        return 1
    fi
    local missing=()
    local pkg
    for pkg in "$@"; do
        if command -v dpkg-query &>/dev/null && \
            dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
            continue
        fi
        missing+=("${pkg}")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi
    if [[ "${APT_UPDATED}" != "true" ]]; then
        apt-get update -qq
        APT_UPDATED=true
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
}

# lerobot 的 torchcodec 需要 FFmpeg 解码视频
if ! command -v ffmpeg &>/dev/null; then
    echo "==> 安装 FFmpeg (torchcodec 视频解码依赖)..."
    if ! apt_install_missing ffmpeg; then
        if command -v yum &>/dev/null; then
            yum install -y -q ffmpeg
        else
            echo "警告: 未找到 apt-get/yum, 请手动安装 FFmpeg"
        fi
    fi
else
    echo "==> FFmpeg 已安装: $(ffmpeg -version 2>&1 | head -1)"
fi

# RoboCasa / robosuite / MuJoCo 的 headless EGL 渲染依赖 GLVND 系统库。
# 新提交机器的基础镜像可能缺 libEGL / libOpenGL，导致
# AttributeError: NoneType has no attribute eglQueryString。
if command -v apt-get &>/dev/null; then
    if ! ldconfig -p 2>/dev/null | grep -q libEGL\.so; then
        echo "==> 安装 libegl1 (MuJoCo EGL 渲染依赖)..."
        apt_install_missing libegl1
    fi
    if ! ldconfig -p 2>/dev/null | grep -q libOpenGL\.so; then
        echo "==> 安装 libopengl0 (PyOpenGL / GLVND 依赖)..."
        apt_install_missing libopengl0
    fi
    if ! ldconfig -p 2>/dev/null | grep -q libGLdispatch\.so; then
        echo "==> 安装 libglvnd0 (GL dispatch 依赖)..."
        apt_install_missing libglvnd0
    fi
    # VLA-Arena 在 eval_vla_arena.sh 中默认使用 OSMesa 软件渲染，以避开
    # 多进程 EGL 在部分 LPAI 机器上触发的 NVIDIA Xid / core dump。
    if ! ldconfig -p 2>/dev/null | grep -q libOSMesa\.so; then
        echo "==> 安装 libosmesa6/libosmesa6-dev (VLA-Arena OSMesa 渲染依赖)..."
        apt_install_missing libosmesa6 libosmesa6-dev
    fi
fi

# NVIDIA/5880 evaluation nodes also run Isaac Sim in a separate persistent
# conda environment.  Its Python wheels still dlopen these host libraries for
# headless Vulkan/RTX and MDL materials.  Keep them in this per-container
# bootstrap so recreating the container does not require a second manual apt
# step; Isaac Sim itself remains isolated from the default project Python.
if [[ "${PROFILE}" == "nvidia" ]] && command -v apt-get &>/dev/null; then
    echo "==> 检查 Isaac Sim headless 系统依赖..."
    apt_install_missing \
        libglu1-mesa libgl1 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libsm6 libice6 libxi6 \
        libxrandr2 libxinerama1 libxcursor1 libxkbcommon0
fi

# LIBERO-plus 官方环境依赖。wand Python 包会动态加载 MagickWand 系统库；
# 新 LPAI 容器只复用 conda env 时，常见失败是 MagickWand shared library not found。
if command -v apt-get &>/dev/null; then
    echo "==> 检查 LIBERO-plus/ImageMagick 系统依赖..."
    apt_install_missing libmagickwand-dev libexpat1 libfontconfig1-dev libpython3-stdlib
fi

# -------------------- 升级 pip --------------------
echo "==> 升级 pip"
pip_install --upgrade pip -q

# -------------------- 安装 leap (editable) --------------------
echo "==> 安装 leap (editable 模式)..."
if [[ "${PROFILE}" == "ppu" ]]; then
    # PPU images provide hardware-specific builds of these four packages.
    # Install every other dependency from the single pyproject dependency list,
    # then register project itself without asking pip to resolve the runtime again.
    mapfile -t PROJECT_REQUIREMENTS < <("${PYTHON}" - "${PROJECT_ROOT}/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    project = tomllib.load(handle)["project"]

excluded = {"torch", "torchvision", "deepspeed", "torchcodec"}
for requirement in project.get("dependencies", []):
    name = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
    if name.lower().replace("_", "-") not in excluded:
        print(requirement)
PY
    )
    pip_install "${PROJECT_REQUIREMENTS[@]}"
    pip_install -e . --no-deps
elif [[ -n "${INSTALL_EXTRAS}" ]]; then
    pip_install -e ".[${INSTALL_EXTRAS}]"
else
    pip_install -e "."
fi

# -------------------- 单独装 lerobot --------------------
# lerobot 不在 pyproject.toml 的 dependencies 里, 因为 0.5.x 硬依赖 numpy>=2.0 /
# torch>=2.7 / av>=15 会被 pip 一并升级, 把 NGC 容器的 numpy 1.x / torch 2.7a ABI
# 拉坏。这里 --no-deps 单装, 跟当前 datasets 4.x 配套就够 dataset reader 用。
pip_install --no-deps "lerobot==0.5.1"

# -------------------- av<14 兼容 --------------------
echo "==> 安装 av<14 (torchvision 兼容)"
pip_install "av<14" -q

# -------------------- imageio (Lance val 视频/图像导出) --------------------
pip_install imageio "imageio-ffmpeg" -q || echo "WARNING: imageio 安装失败 (仅 validation 视频导出需要)"

# -------------------- numpy 兼容 --------------------
# 解除 PIP_CONSTRAINT 后 datasets 等可能把 numpy 升到 2.x，
# 但 NVIDIA 容器的 torch 是用 numpy 1.x ABI 编译的，必须降回来。
TORCH_NUMPY_VER=$("${PYTHON}" -c "
try:
    import torch, numpy
    if numpy.__version__.startswith('2'):
        print('downgrade')
    else:
        print('ok')
except Exception:
    print('ok')
")
if [[ "${TORCH_NUMPY_VER}" == "downgrade" ]]; then
    echo "==> numpy 2.x 与当前 torch ABI 不兼容, 降级到 numpy<2"
    pip_install "numpy<2" -q
fi

# -------------------- causal-conv1d 源码编译 (可选) --------------------
if ${FORCE_CAUSAL_CONV1D}; then
    echo "==> 从源码编译 causal-conv1d (解决 ABI 不匹配问题)..."
    "${PYTHON}" -m pip uninstall causal-conv1d -y 2>/dev/null || true
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip_install \
        --no-cache-dir --no-build-isolation "causal-conv1d>=1.6"
fi

# -------------------- flash-attn 源码编译 (可选) --------------------
if ${FORCE_FLASH_ATTN}; then
    echo "==> 从源码编译 flash-attn (解决 ABI 不匹配问题)..."
    "${PYTHON}" -m pip uninstall flash-attn -y 2>/dev/null || true
    pip_install \
        --no-cache-dir --no-build-isolation "flash-attn>=2.5"
fi

# -------------------- 验证安装 --------------------
echo ""
echo "==> 验证安装..."
"${PYTHON}" -c "
import torch, transformers, omegaconf, accelerate
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  omegaconf:    {omegaconf.__version__}')
print(f'  accelerate:   {accelerate.__version__}')

import deepspeed
print(f'  deepspeed:    {deepspeed.__version__}')

if '${PROFILE}' == 'nvidia':
    import cv2
    import h5py
    import msgpack
    import torchcodec
    import msgpack_numpy
    import pydantic
    import websockets
    import yaml
    print(f'  torchcodec:   {torchcodec.__version__}')
    print(f'  websockets:   {websockets.__version__}')
    print(f'  opencv:       {cv2.__version__}')
    print(f'  h5py:         {h5py.__version__}')
    print(f'  pydantic:     {pydantic.__version__}')
    print('  msgpack_numpy:' + getattr(msgpack_numpy, '__version__', 'installed'))
    if not torch.cuda.is_available():
        raise RuntimeError('nvidia profile requires torch.cuda.is_available() == True')
    print(f'  CUDA devices: {torch.cuda.device_count()}')

import lerobot
print(f'  lerobot:      {lerobot.__version__}')

try:
    import causal_conv1d
    print(f'  causal-conv1d: {causal_conv1d.__version__}')
except ImportError:
    print(f'  causal-conv1d: (未安装或 ABI 不匹配, 使用 --with-causal-conv1d 从源码编译)')

try:
    import flash_attn
    print(f'  flash-attn:    {flash_attn.__version__}')
except ImportError:
    print(f'  flash-attn:    (未安装或 ABI 不匹配, 使用 --with-flash-attn 从源码编译)')
"

echo ""
echo "==> pip dependency check..."
# lerobot 0.5.1 is deliberately installed with --no-deps above: installing its
# declared numpy>=2 / av>=15 requirements would break the NVIDIA image ABI and
# the training data path.  Consequently a global pip check can report lerobot
# metadata conflicts (and base-image conflicts such as cudf/pyarrow) even when
# the imports required by project have passed.  Keep the report visible, but do not
# turn those known metadata conflicts into an install failure.
if ! "${PYTHON}" -m pip check; then
    echo "WARNING: pip check reported dependency conflicts."
    echo "WARNING: lerobot --no-deps conflicts are expected; inspect any non-lerobot/base-image entries above."
fi

# 验证 leap 包路径指向当前 repo
LEAP_PATH=$("${PYTHON}" -c "import leap; print(leap.__file__)")
echo "  leap 包路径:  ${LEAP_PATH}"

# 用 realpath 做比较，避免符号链接误报
REAL_PROJECT=$(realpath "${PROJECT_ROOT}")
REAL_LEAP=$(realpath "$(dirname "$(dirname "${LEAP_PATH}")")")
if [[ "${REAL_LEAP}" == "${REAL_PROJECT}" ]]; then
    echo "  leap 包正确指向当前 repo"
else
    echo "  警告: leap 包指向 ${LEAP_PATH}，不是当前 repo"
    echo "    这可能是因为另一个 editable install 覆盖了路径"
    echo "    建议: pip uninstall leap && pip install -e ."
fi

# -------------------- 清理 pip cache --------------------
echo ""
echo "==> 清理 pip cache..."
"${PYTHON}" -m pip cache purge 2>/dev/null || true

echo ""
echo "========== 安装完成 =========="
echo ""
echo "训练入口: python scripts/ME_U0/train.py --config PATH"
echo "多机入口: bash scripts/ME_U0/train_ppu.sh --help"
