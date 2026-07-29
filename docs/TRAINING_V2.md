# EvidenceAgent-MM 2.0 training guide

## What is trained

The v0.1 state machine remains the default production baseline. Version 2.0 trains
a JSON-constrained policy that receives a question plus provenance-preserving
evidence and predicts one of three states:

- `answered`, with claim-level evidence IDs;
- `needs_clarification`, with one targeted follow-up;
- `abstained`, with the missing evidence named explicitly.

All three stages use the same `AgentTrainingTarget`, so SFT, preference learning,
GRPO, offline scoring, and data-flywheel corrections cannot silently drift into
different output formats.

The runtime renders every example with the selected tokenizer's native chat
template. For Qwen3 it sets `enable_thinking=false`: the policy must emit the
machine-checkable JSON directly instead of spending the GRPO completion budget on
an unscored reasoning preamble.

## Data contracts

Run:

```bash
python scripts/build_v2_training_data.py
```

This validates Bronze fixtures and questions, groups splits by session, and writes:

```text
benchmarks/eamm_v2_training/
  sft.jsonl
  dpo.jsonl
  grpo.jsonl
  manifest.json
```

Every row has `schema_version=2.0`, a stable content-derived ID, source/license,
split, evidence atoms, prompts, and the stage-specific target. An audio-derived
atom includes structured `asr` and `acoustic` attributes; the prompt builder
serializes them rather than reducing audio to transcript text alone.

Bronze is a synthetic contract dataset. Replace or mix it with licensed real or
semi-real meeting data before making quality claims.

## Reward definition

`score_completion` parses and validates the model JSON before scoring. The default
normalized reward is:

```text
R = 0.10 R_format
  + 0.25 R_status
  + 0.25 R_citation_F1
  + 0.25 R_grounding
  + 0.15 R_abstention
```

`R_grounding` requires every claimed citation to exist in the supplied evidence
context and every claim to link to cited IDs. `R_abstention` penalizes both unsafe
answering and over-refusal: answerable references must be answered; unanswerable
references must name the correct missing evidence; ambiguous references must ask
a clarification. There is no model judge and no network call in the reward.

Malformed non-JSON output still receives zero. A JSON object that is close to the
contract but fails one Pydantic invariant receives deterministic component-level
credit for field types, status, citation F1, grounding, and refusal behavior,
capped at `0.75`. Only a fully valid `AgentTrainingTarget` can score above that
cap. This avoids the all-zero GRPO cold-start failure while preserving a strict
incentive to satisfy every cross-field constraint.

## Local validation

```bash
python -m pip install -e '.[dev]'
python scripts/build_v2_training_data.py
for stage in sft dpo grpo; do
  python scripts/train_v2.py --config "configs/${stage}_4090.json" --dry-run
done
pytest
ruff check .
ruff format --check .
```

Dry-run reads every JSONL row through its Pydantic contract, filters the requested
split, rejects duplicate IDs, validates context budgets, and reports dataset
statistics. It does not import Torch, Transformers, TRL, PEFT, or bitsandbytes.

## AutoDL commands

From the repository root on the instance:

```bash
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_preflight.sh
PORTFOLIO_V2_MODE=smoke bash scripts/autodl_v2_run.sh
PORTFOLIO_V2_MODE=full bash scripts/autodl_v2_run.sh
```

The default full order is a real adapter chain:

```text
Qwen3 base -> SFT adapter -> continued DPO adapter -> continued GRPO adapter
```

`EAMM_CHAIN_STAGES=1` is the default. To resume a later stage from an already
validated adapter, set `EAMM_DPO_MODEL=/absolute/sft/path` or
`EAMM_GRPO_MODEL=/absolute/dpo/path`. Set `EAMM_CHAIN_STAGES=0` only for an
explicit independent-stage experiment, and record that deviation.

Optional controls:

```bash
EAMM_V2_STAGES=sft,dpo PORTFOLIO_V2_MODE=smoke bash scripts/autodl_v2_run.sh
EAMM_OUTPUT_ROOT=/root/autodl-tmp/my-run PORTFOLIO_V2_MODE=full \
  bash scripts/autodl_v2_run.sh
```

## DeepSpeed and multi-GPU evidence

DeepSpeed is optional at package level:

```bash
python -m pip install -e '.[train,distributed]'
```

The AutoDL bootstrap includes it by default. Select a checked configuration using
environment variables; the same selector applies to SFT, DPO, and GRPO:

```bash
# ZeRO-2, optimizer state on CPU, one GPU compatibility smoke:
EAMM_DEEPSPEED=zero2 EAMM_WORLD_SIZE=1 PORTFOLIO_V2_MODE=smoke \
  bash scripts/autodl_v2_run.sh

# ZeRO-3, optimizer and parameters on CPU, future two-GPU acceptance:
EAMM_DEEPSPEED=zero3 EAMM_WORLD_SIZE=2 PORTFOLIO_V2_MODE=smoke \
  bash scripts/autodl_v2_run.sh
```

`EAMM_DEEPSPEED` accepts `none`, `zero2`, `zero3`, or a JSON path.
`EAMM_WORLD_SIZE` is passed to the DeepSpeed launcher and written to every run
manifest. Preflight rejects a world size larger than the GPU count.

Important boundaries:

- `world_size=1` proves only that the Trainer/DeepSpeed integration can start on
  one process. It is not multi-GPU evidence and says nothing about scaling.
- A dry-run with `world_size=2` validates configuration only. Real evidence
  requires a machine with at least two visible GPUs and a completed run manifest
  reporting `environment.world_size=2`.
- DeepSpeed configurations require `load_in_4bit=false`. The runtime removes
  `device_map="auto"` because it conflicts with distributed placement.
- ZeRO-2 offloads optimizer states; ZeRO-3 also offloads parameters. Both can be
  slower due to CPU/GPU transfers and need adequate host RAM.
- The current rented single-4090 AutoDL instance cannot satisfy the real multi-GPU
  acceptance gate. Do not write “distributed training verified” on a resume from
  a one-GPU smoke test.

Dependency and TRL/Transformers compatibility are checked locally at the config
surface and must still be confirmed by one optimizer step for each selected stage
on the target Linux/CUDA environment.

Smoke mode runs one optimizer step per requested stage. Full mode uses the config's
epochs or `max_steps`. Both first run preflight. Each stage writes logs, trainer
state, adapter weights, metrics, and a `run_manifest.json` containing versions,
GPU, CUDA, Git revision, metrics, and peak allocated VRAM.

After evaluating an adapter, validate and upload exactly that artifact directory:

```bash
HF_TOKEN=... python scripts/publish_v2_adapter.py \
  --artifact-dir /root/autodl-tmp/eamm-v2/grpo \
  --repo-id jatshi/EvidenceAgent-MM-GRPO-2.0
```

The publisher refuses directories without both a completed run manifest and an
adapter file. `HF_TOKEN` is read only from the environment.

### ZeRO-3 checkpoint handling

The checked training recipes are LoRA/PEFT runs. `trainer.save_model()` writes the
adapter artifact; publishing the adapter does not require reconstructing a full
FP32 base model. First verify:

```bash
test -f /root/autodl-tmp/eamm-v2/grpo/run_manifest.json
find /root/autodl-tmp/eamm-v2/grpo -maxdepth 2 \
  \( -name adapter_model.safetensors -o -name adapter_model.bin \) -print
```

If a future non-PEFT ZeRO-3 run intentionally needs a consolidated FP32 model,
DeepSpeed places `zero_to_fp32.py` in the checkpoint directory. Refuse to merge
unless the completed checkpoint and helper both exist:

```bash
CHECKPOINT=/absolute/path/to/completed-zero3-checkpoint
test -d "${CHECKPOINT}"
test -f "${CHECKPOINT}/zero_to_fp32.py" || {
  echo "Not a mergeable ZeRO checkpoint: zero_to_fp32.py is missing" >&2
  exit 2
}
python "${CHECKPOINT}/zero_to_fp32.py" \
  "${CHECKPOINT}" \
  "${CHECKPOINT}/consolidated_fp32.bin"
```

This command is intentionally not part of the LoRA publisher. Never call an
adapter-only directory a consolidated full-model checkpoint.

## Hard-case data flywheel

`FeedbackStore` accepts only typed `HardCaseFeedback`. A record has a stable
fingerprint, failure reason, observed output, optional correction, evidence,
license, reviewer, timestamp, and `consent_for_training`.

Export includes only records that both contain a correction and explicitly allow
training. Duplicate feedback is ignored. This prevents user corrections from
silently becoming training data without provenance or consent.

```bash
python scripts/feedback_v2.py add corrected_case.json
python scripts/feedback_v2.py export \
  --output data/processed/eamm_v2_feedback_training
```

## Failure diagnosis

- `nvidia-smi did not report a usable GPU`: the instance is not attached to a GPU.
- `<20 GiB VRAM`: wrong instance type or GPU partition.
- `<35 GiB disk`: clear model caches/checkpoints or enlarge storage before download.
- CUDA unavailable after bootstrap: inspect driver and installed cu128 Torch wheel.
- OOM in GRPO: keep 1.7B, reduce `grpo_num_generations` from 4 to 2, then reduce
  completion length; record every deviation in the run manifest.
- Invalid reward outputs: inspect the raw completion; malformed JSON receives zero.

Do not delete older checkpoints until one complete stage has a valid manifest and
the adapter can be reloaded.
