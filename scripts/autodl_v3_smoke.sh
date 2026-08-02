#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_V3_VENV:-/root/autodl-tmp/portfolio-v3/envs/evidence-agentic}"
ARTIFACT_DIR="${EAMM_V3_SMOKE_DIR:-$REPO_ROOT/artifacts/v3/smoke}"
source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_DIR" "$REPO_ROOT/data/v3/verl"

python -m pytest -q tests/test_agentic_v3.py
python scripts/build_verl_v3_dataset.py \
  --input benchmarks/eamm_v3_hard/smoke_cases.jsonl \
  --output-dir data/v3/verl
python scripts/verl_v3_api_smoke.py --output "$ARTIFACT_DIR/verl_tools.json"
python scripts/qwen_smoke.py \
  --model "${EAMM_V3_MODEL:-Qwen/Qwen3-1.7B}" \
  --output "$ARTIFACT_DIR/qwen.json"

if [[ "${EAMM_V3_RUN_TRAIN_SMOKE:-1}" == "1" ]]; then
  TRAIN_MARKER="$ARTIFACT_DIR/train/training_completed.json"
  if [[ -s "$TRAIN_MARKER" ]]; then
    TRAIN_MARKER="$TRAIN_MARKER" python -c 'import json, os; assert json.load(open(os.environ["TRAIN_MARKER"], encoding="utf-8"))["status"] == "completed"'
    echo "Existing completed agentic training marker verified; skipping duplicate GPU work."
  else
    EAMM_V3_TOTAL_STEPS=1 \
    EAMM_V3_OUTPUT_DIR="$ARTIFACT_DIR/train" \
      bash scripts/autodl_v3_agentic_train.sh
  fi
fi

python scripts/write_v3_run_manifest.py \
  --stage smoke \
  --artifact "$ARTIFACT_DIR/verl_tools.json" \
  --artifact "$ARTIFACT_DIR/qwen.json" \
  --output "$ARTIFACT_DIR/run_manifest.json"
