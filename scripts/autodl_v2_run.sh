#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_VENV_DIR:-${REPO_DIR}/.venv-train}"
OUTPUT_ROOT="${EAMM_OUTPUT_ROOT:-/root/autodl-tmp/eamm-v2}"
MODE="${PORTFOLIO_V2_MODE:-smoke}"
STAGES="${EAMM_V2_STAGES:-sft,dpo,grpo}"
DRY_RUN_ONLY="${EAMM_V2_DRY_RUN_ONLY:-0}"
DEEPSPEED_MODE="${EAMM_DEEPSPEED:-none}"
WORLD_SIZE="${EAMM_WORLD_SIZE:-1}"
CHAIN_STAGES="${EAMM_CHAIN_STAGES:-1}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/.cache/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "PORTFOLIO_V2_MODE must be smoke or full, got: ${MODE}" >&2
  exit 2
fi
if [[ "${CHAIN_STAGES}" != "0" && "${CHAIN_STAGES}" != "1" ]]; then
  echo "EAMM_CHAIN_STAGES must be 0 or 1, got: ${CHAIN_STAGES}" >&2
  exit 2
fi

cd "${REPO_DIR}"
bash scripts/autodl_v2_preflight.sh
deepspeed_config=""
case "${DEEPSPEED_MODE}" in
  none)
    if [[ "${WORLD_SIZE}" != "1" ]]; then
      echo "EAMM_WORLD_SIZE>1 requires EAMM_DEEPSPEED=zero2|zero3|path." >&2
      exit 2
    fi
    ;;
  zero2)
    deepspeed_config="configs/deepspeed/zero2_cpu_offload.json"
    ;;
  zero3)
    deepspeed_config="configs/deepspeed/zero3_cpu_offload.json"
    ;;
  *)
    deepspeed_config="${DEEPSPEED_MODE}"
    ;;
esac
IFS=',' read -r -a stage_list <<< "${STAGES}"
previous_output=""
for stage in "${stage_list[@]}"; do
  config="configs/${stage}_4090.json"
  if [[ ! -f "${config}" ]]; then
    echo "Unknown stage or missing config: ${stage}" >&2
    exit 2
  fi
  train_args=(
    --config "${config}"
    --output-dir "${OUTPUT_ROOT}/${stage}"
    --world-size "${WORLD_SIZE}"
  )
  model_override=""
  case "${stage}" in
    dpo)
      model_override="${EAMM_DPO_MODEL:-}"
      ;;
    grpo)
      model_override="${EAMM_GRPO_MODEL:-}"
      ;;
  esac
  if [[ -z "${model_override}" && "${CHAIN_STAGES}" == "1" ]]; then
    model_override="${previous_output}"
  fi
  if [[ -n "${model_override}" ]]; then
    if [[ "${DRY_RUN_ONLY}" != "1" && ! -d "${model_override}" ]]; then
      echo "Chained stage input does not exist: ${model_override}" >&2
      exit 2
    fi
    train_args+=(--model-name-or-path "${model_override}")
  fi
  if [[ -n "${deepspeed_config}" ]]; then
    train_args+=(--deepspeed-config "${deepspeed_config}" --no-4bit)
  fi
  if [[ -n "${deepspeed_config}" && "${DRY_RUN_ONLY}" != "1" ]]; then
    command=(
      "${VENV_DIR}/bin/deepspeed"
      --num_gpus "${WORLD_SIZE}"
      scripts/train_v2.py
      "${train_args[@]}"
    )
  else
    command=("${VENV_DIR}/bin/python" scripts/train_v2.py "${train_args[@]}")
  fi
  if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
    command+=(--dry-run)
  elif [[ "${MODE}" == "smoke" ]]; then
    command+=(--smoke)
  fi
  echo "Running ${stage} (${MODE})"
  "${command[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
  previous_output="${OUTPUT_ROOT}/${stage}"
done

echo "EvidenceAgent-MM v2 run complete: ${OUTPUT_ROOT}"
