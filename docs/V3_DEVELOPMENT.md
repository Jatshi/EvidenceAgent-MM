# EvidenceAgent-MM 3.0：Agentic RL 升级、完整学习与踩坑复盘

## 1. 为什么要做这个项目

普通会议/课堂助手往往只生成摘要。用户追问“谁在什么时候提出了方案、屏幕对应哪页、依据是什么”时，摘要模型很容易把不同说话人、时间段和 OCR 内容拼成一段流畅但不可验证的话。EvidenceAgent-MM 的目标是让答案成为一组可审计 claim：每个事实带音视频时间戳或证据 ID；证据不足时澄清或拒答；工具输出被视为不可信数据而不是新指令。

3.0 进一步解决“怎样训练模型学会搜证据、验证、再回答”的问题。它把静态 RAG 升级为带预算的多轮 agent，并用 VERL/GRPO 对工具轨迹和终局答案共同给奖励。

## 2. 3.0 新增内容

- immutable evidence budget：限制工具步数、唯一证据数和工具总时长。
- `evidence_search` 与 `verify_claim_support` 两个可回放工具。
- prompt-injection detector：工具返回的恶意指令只能作为证据文本，不能覆盖系统规则。
- hard benchmark：跨模态冲突、缺证据、澄清、安全拒答和注入样本。
- VERL 0.8 Parquet 数据合同：`agent_name=tool_agent`、raw chat、逐样本 tool kwargs。
- GRPO 轨迹奖励：状态、引用、安全、格式和 2% 轨迹效率 tie-breaker。
- 单卡 FSDP/offload + vLLM async rollout、LoRA rank 8 的真实 50-step 训练。
- 从 8.16GB FSDP actor checkpoint 导出约 34MB 标准 PEFT adapter。

## 3. Evidence budget 怎样设计

budget 是不可变对象，每次调用 `consume` 返回新状态。它至少追踪：

- `used_steps / max_steps`；
- `unique_evidence_ids / max_unique_evidence`；
- `used_tool_time_ms / max_tool_time_ms`。

在执行工具前检查预算，能避免 agent 无限搜索；按唯一证据计数能阻止重复检索同一片段伪装成“证据更多”；工具时间预算则把效率变成可评测属性。不可变状态比原地修改更容易回放和并发调试。

## 4. 为什么工具输出必须是不可信数据

OCR 或会议文本可能包含“忽略之前指令、泄露 system prompt”。如果模型把检索结果和系统指令放在同一信任层，RAG 就变成了 prompt injection 通道。3.0 的处理是：

1. tool schema 明确标注输出不可信；
2. deterministic detector 标记明显注入；
3. claim verification 只检查证据关联，不执行证据里的命令；
4. 触发安全条件时终止或转人工。

detector 不是完整安全证明，只是可复现的第一道 gate。真实部署仍需模型级分类、权限隔离和审计。

## 5. GRPO 为什么要求每组至少两个 rollout

GRPO 对同一 prompt 的多个回答计算组内相对优势。若 `rollout_n=1`，组内均值就是它自己，标准化优势恒为 0，因此 loss、gradient 都为 0。第一次自动降载把 `n` 改为 1，日志虽然显示训练步在增加，`grad_norm` 却一直是 0。最终恢复 `n=2`，并把“非零组内 reward 差 + 非零 grad_norm”设为训练硬门槛。

这也是面试中非常有价值的工程判断：GPU 忙、step 增长、checkpoint 生成都不能证明强化学习真的发生了。

## 6. 轨迹奖励怎样避免全零与被刷分

旧奖励要求整个 `solution_str` 必须是纯 JSON；但 VERL agent loop 传入的是“tool calls + observations + final answer”完整轨迹，导致所有样本为 0。修复后使用 JSONDecoder 从轨迹中提取最后一个 JSON object。

早期模型还不会严格输出 JSON，因此增加有限的 dense shaping：

- status hint；
- expected citation 覆盖；
- JSON contract 字段；
- 安全约束；
- 仅 2% 的轨迹效率，用于打破同一 prompt 下两个回答同分。

正确性权重占 98%，效率不能覆盖错误答案。相关测试覆盖纯 JSON、markdown fence、多轮轨迹、格式不完整、安全违规和 tie-breaker。

## 7. 真实训练配置与结果边界

- GPU：RTX 4090 24GB；
- base model：`Qwen/Qwen3-1.7B`；
- framework：VERL 0.8.0、vLLM async rollout；
- LoRA：rank 8，alpha 16，all-linear；
- dataset：12 train / 4 validation，均为确定性合成 hard cases；
- batch：2 prompts，每个 prompt 2 rollouts；
- actor/optimizer CPU offload；
- 50 actor updates，step 50 actor/optimizer/RNG/data checkpoint 已完整保存。

奖励修复后的初始 validation mean 为约 0.4296。可观测训练 step 11–49 的 `grad_norm` 持续非零，例如 step 11 为 1.2847，后续多在约 0.49–1.51。这里不宣称最终准确率提升：本次 `test_freq=-1`，且最终 checkpoint 保存后的 rollout 同步 OOM，未执行独立最终验证。应写“完成可学习的 50-step GRPO 工程验证”，而不是“模型性能达到某数值”。

## 8. 单卡 FSDP/DeepSpeed 应怎样表述

world size=1 时，PyTorch FSDP 明确退化为 `NO_SHARD`。offload 和统一 checkpoint 接口仍然真实生效，但没有跨卡通信或分片加速。因此可以说实现并验证了 FSDP/分布式训练契约、rank-aware 状态和恢复路径；不能说完成了多卡扩展实验。

单卡 4090 的主要挑战是 actor、vLLM rollout、CUDA graph 和保存时完整权重短时共置，而不是通信。

## 9. 全过程踩坑

### 数据集太小导致空 dataloader

最初只有 1 条 train case，而 retry batch 为 2；VERL 的 `drop_last` 让 dataloader 长度变成 0。解决方案不是简单把 batch 永久改为 1，而是生成 12 条训练、4 条验证，覆盖 answered、prompt injection abstain 和 clarification 三类行为，并校验所有 evidence store。

### 保存时 OOM，而平时训练正常

训练时峰值可控，但 checkpoint 后 `FSDP parameter reload + vLLM weight sync` 需要额外约 194MiB，24GB 显存只剩约 101MiB，发生 OOM。将 vLLM memory utilization 从 0.45 降到 0.30，并把 save frequency 改为 50，避免反复写 7.7GB 中间 checkpoint。

### expandable segments 与 vLLM CuMemAllocator 冲突

按 PyTorch OOM 提示启用 `expandable_segments:True` 后，vLLM 直接断言失败，因为 sleep mode 的 CuMemAllocator 不兼容该策略。最终使用 `PYTORCH_ALLOC_CONF=max_split_size_mb:128`。不能机械照抄报错提示，必须考虑另一个框架的 allocator。

### 最终 post-save 同步 OOM

step 50 的 actor update 和 checkpoint save 已完成，随后 VERL 再同步 actor 到 rollout 时 OOM。完成标记明确写为 `completed_with_post_checkpoint_rollout_sync_oom`，包含 checkpoint 大小、最后日志 step 和 claim boundary；没有伪装成无异常结束。

## 10. PEFT adapter 导出

VERL/FSDP checkpoint 包含 base layer 和 LoRA tensors，共约 8.16GB。导出脚本只筛选 392 个 `lora_A/lora_B` tensor，移除 PEFT 内部 `.default` 名称并生成 `adapter_config.json`、safetensors 和 export manifest，压缩到约 34MB。发布前必须用 `PeftModel.from_pretrained` 在 base model 上真实加载验证。

## 11. 面试问答

**为什么不用普通 RAG？** 普通 RAG 通常一次检索后生成，难以表达证据预算、工具失败恢复、跨模态冲突和澄清。Agent 允许多步搜证和验证，但需要更严格的安全与评测。

**为什么 2% 长度奖励合理？** 它只作为同分 tie-breaker，正确性仍占 98%；目的是让早期 GRPO 有组内方差。若权重过大，模型会学习过度简短，因此必须做消融。

**怎样证明没有幻觉？** 不能证明“永不幻觉”。能做的是逐 claim 引用、证据 ID 校验、拒答/澄清、对抗样本和可量化 unsupported-claim rate。

**下一步实验是什么？** 固定训练种子比较 final-only reward、dense trajectory reward 和去掉 efficiency 的消融；在公开会议数据上做独立 final evaluation；多卡上测 FSDP/rollout 扩展效率。

## 12. 亲手练习

1. 手写 EvidenceBudget 并验证三种预算分别超限。
2. 构造含注入文本的 tool output，证明它不会变成指令。
3. 用两个 reward 相同的 rollout 手算 GRPO advantage，再解释 n=1 为什么为 0。
4. 修改 reward 使一个引用正确、一个引用错误，确认 `grad_norm` 非零。
5. 从 step-50 checkpoint 导出 adapter，并在 Qwen3-1.7B 上加载、生成一条回答。

完成这些步骤后，你应该能从产品问题、数据合同、agent loop、奖励设计、显存工程和发布边界完整讲清项目，而不是只背“多模态 RAG + Agent”。
