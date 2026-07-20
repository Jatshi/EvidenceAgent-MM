#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -e '.[ocr]'

python - <<'PY'
import paddle

paddle.utils.run_check()
assert paddle.device.is_compiled_with_cuda(), "PaddlePaddle is not a CUDA build"
print({"paddle": paddle.__version__, "device": paddle.device.get_device()})
PY
