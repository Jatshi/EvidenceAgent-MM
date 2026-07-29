#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_VENV_DIR:-${REPO_DIR}/.venv-train}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.cache/uv}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/.cache/torch_extensions}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "${REPO_DIR}"
mkdir -p \
  "${HF_HOME}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${EAMM_OUTPUT_ROOT:-/root/autodl-tmp/eamm-v2}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip uv
"${VENV_DIR}/bin/uv" pip install \
  --python "${VENV_DIR}/bin/python" \
  torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
training_extras="train,dev"
if [[ "${EAMM_INSTALL_DEEPSPEED:-1}" == "1" ]]; then
  training_extras="train,distributed,dev"
fi
"${VENV_DIR}/bin/uv" pip install \
  --python "${VENV_DIR}/bin/python" \
  -e ".[${training_extras}]"
"${VENV_DIR}/bin/python" scripts/build_v2_training_data.py

echo "Bootstrap complete: ${VENV_DIR}"
echo "Next: bash scripts/autodl_v2_preflight.sh"
