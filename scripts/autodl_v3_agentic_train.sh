#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EAMM_V3_VENV:-/root/autodl-tmp/portfolio-v3/envs/evidence-agentic}"
DATA_DIR="${EAMM_V3_DATA_DIR:-$REPO_ROOT/data/v3/verl}"
OUTPUT_DIR="${EAMM_V3_OUTPUT_DIR:-$REPO_ROOT/artifacts/v3/agentic_train}"
MODEL_PATH="${EAMM_V3_MODEL:-Qwen/Qwen3-1.7B}"
ROLLOUT_ENGINE="${EAMM_V3_ROLLOUT_ENGINE:-vllm}"
TOTAL_STEPS="${EAMM_V3_TOTAL_STEPS:-1}"
SAVE_FREQ="${EAMM_V3_SAVE_FREQ:-1}"
TRAIN_BATCH="${EAMM_V3_TRAIN_BATCH:-2}"
MINI_BATCH="${EAMM_V3_MINI_BATCH:-1}"
ROLLOUT_N="${EAMM_V3_ROLLOUT_N:-2}"
MAX_RESPONSE="${EAMM_V3_MAX_RESPONSE:-512}"
LORA_RANK="${EAMM_V3_LORA_RANK:-8}"
LORA_ALPHA="${EAMM_V3_LORA_ALPHA:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${EAMM_V3_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/compat${PYTHONPATH:+:$PYTHONPATH}"
# vLLM sleep mode uses CuMemAllocator; expandable_segments is incompatible
# with that allocator. A bounded split size reduces fragmentation without
# disabling the memory pool used for colocated rollout/training.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}"
test -f "$DATA_DIR/train.parquet"
test -f "$DATA_DIR/validation.parquet"
command -v nvidia-smi >/dev/null
python -c 'import torch; assert torch.cuda.device_count() == 1; assert torch.cuda.is_available()'
python -c 'import importlib.metadata as m; assert m.version("verl") == "0.8.0"'
mkdir -p "$OUTPUT_DIR"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rollout_correction.bypass_mode=True \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/validation.parquet" \
  data.return_raw_chat=True \
  data.train_batch_size="$TRAIN_BATCH" \
  data.max_prompt_length=1024 \
  data.max_response_length="$MAX_RESPONSE" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  reward.custom_reward_function.path="$REPO_ROOT/scripts/verl_reward_v3.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO_ROOT/configs/verl/tools_v3.json" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
  trainer.val_before_train=True \
  trainer.log_val_generations=2 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.logger="['console']" \
  trainer.project_name=evidenceagent_mm_v3 \
  trainer.experiment_name="qwen3_1_7b_agentic_${TOTAL_STEPS}step" \
  trainer.default_local_dir="$OUTPUT_DIR"

MARKER="$OUTPUT_DIR/training_completed.json"
MARKER="$MARKER" python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["MARKER"]); p.write_text(json.dumps({"status":"completed"},indent=2)+"\n")'
