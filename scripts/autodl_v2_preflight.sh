#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_VENV_DIR:-${REPO_DIR}/.venv-train}"
OUTPUT_ROOT="${EAMM_OUTPUT_ROOT:-/root/autodl-tmp/eamm-v2}"
DEEPSPEED_MODE="${EAMM_DEEPSPEED:-none}"
WORLD_SIZE="${EAMM_WORLD_SIZE:-1}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "${REPO_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}; run scripts/autodl_v2_bootstrap.sh first." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
"${VENV_DIR}/bin/python" scripts/build_v2_training_data.py
preflight_args=(--output "${OUTPUT_ROOT}/preflight_v2.json")
if [[ "${EAMM_PREFLIGHT_ALLOW_CPU:-0}" == "1" ]]; then
  preflight_args+=(--allow-cpu)
fi
case "${DEEPSPEED_MODE}" in
  none)
    ;;
  zero2)
    preflight_args+=(--deepspeed-config "configs/deepspeed/zero2_cpu_offload.json")
    ;;
  zero3)
    preflight_args+=(--deepspeed-config "configs/deepspeed/zero3_cpu_offload.json")
    ;;
  *)
    if [[ ! -f "${DEEPSPEED_MODE}" ]]; then
      echo "EAMM_DEEPSPEED must be none, zero2, zero3, or a JSON path." >&2
      exit 2
    fi
    preflight_args+=(--deepspeed-config "${DEEPSPEED_MODE}")
    ;;
esac
preflight_args+=(--world-size "${WORLD_SIZE}")
"${VENV_DIR}/bin/python" scripts/autodl_v2_preflight.py "${preflight_args[@]}"

echo "Preflight passed. Report: ${OUTPUT_ROOT}/preflight_v2.json"
