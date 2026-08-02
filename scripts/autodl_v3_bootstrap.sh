#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_V3_VENV:-/root/autodl-tmp/portfolio-v3/envs/evidence-agentic}"
PYTHON_BIN="${EAMM_V3_PYTHON:-python3}"
mkdir -p "$(dirname "$VENV_DIR")" /root/autodl-tmp/huggingface "$REPO_ROOT/artifacts/v3"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
python -m pip install -e "$REPO_ROOT[train,distributed,dev]"
# verl 0.8 supports vLLM only through 0.12. vLLM 0.13+ also raises the
# OpenCV floor to a NumPy-2-only build, while verl 0.8 requires NumPy <2.
# This exact set keeps the AutoDL driver-560/CUDA-12.8 stack ABI-compatible.
python -m pip install \
  'numpy>=1.26,<2' \
  opencv-python-headless==4.11.0.86 \
  cupy-cuda12x==13.6.0 \
  verl==0.8.0 vllm==0.12.0 pyarrow mlflow
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -c 'import importlib.metadata as m; assert m.version("verl") == "0.8.0"; assert m.version("vllm") == "0.12.0"; assert m.version("opencv-python-headless") == "4.11.0.86"; assert m.version("cupy-cuda12x") == "13.6.0"'
python -m pip freeze > "$REPO_ROOT/artifacts/v3/environment.freeze.txt"
