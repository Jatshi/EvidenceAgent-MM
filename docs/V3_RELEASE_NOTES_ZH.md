# EvidenceAgent-MM 3.0 新增内容与发布说明

## 版本定位

3.0 把 2.0 的“可引用多模态 RAG”升级为“会主动搜证、验证并在证据不足时停止的
Agent”。这不是简单增加一个 LLM 调用，而是把工具预算、工具轨迹、奖励函数、训练
数据合同、检查点导出和声明边界做成可测试、可复盘的完整链路。

## 新增能力

### 1. 有预算的 Agentic 检索

- 新增不可变 `EvidenceBudget`，同时限制工具步数、唯一证据数和工具总耗时；
- 新增 `evidence_search` 和 `verify_claim_support` 两类可回放工具；
- 每次调用都保留输入、结果、耗时和剩余预算，便于重放失败轨迹；
- 超预算时返回结构化失败，不允许 Agent 无限检索。

### 2. 工具输出安全边界

- OCR、转写和检索结果统一视为不可信数据；
- 增加确定性 prompt-injection 检测与安全拒答样本；
- claim verifier 只验证证据关联，不执行证据文本中的命令；
- hard benchmark 覆盖跨模态冲突、证据缺失、澄清、拒答和注入攻击。

### 3. VERL/GRPO 训练链路

- 新增 VERL 0.8 Parquet 数据合同与逐样本工具参数；
- 使用 Qwen3-1.7B、LoRA rank 8、vLLM async rollout 和 FSDP/offload；
- 奖励同时衡量回答状态、引用覆盖、安全性、JSON 合同与轨迹效率；
- 轨迹效率只占 2%，用于同分破局，不能覆盖事实错误；
- 加入“组内 reward 必须有差异、`grad_norm` 必须非零”的训练硬门槛。

### 4. 检查点与发布

- 保存 step 10、step 50 和 smoke 的完整 actor/optimizer/RNG/data 状态；
- 从约 8.16GB FSDP actor checkpoint 导出约 34MB 标准 PEFT adapter；
- 导出时仅保留 392 个 LoRA tensor，并记录源检查点和 SHA-256；
- 在 RTX 4090 上真实加载 adapter 并完成 12-token 生成验证；
- 权重发布于 [Hugging Face](https://huggingface.co/jatshi/EvidenceAgent-MM-Qwen3-1.7B-Agentic-GRPO-v3)。

## 真实验证结果

| 项目 | 结果 |
|---|---|
| GPU | RTX 4090 24GB |
| 数据 | 12 train / 4 validation hard cases |
| 训练 | 50 个 actor update，step-50 完整 checkpoint 已保存 |
| 有效学习证据 | step 11–49 `grad_norm` 持续非零，约 0.49–1.51 |
| adapter | 34,916,688 字节，CUDA 加载与生成通过 |
| 本地回归 | 79 tests passed |

## 重要声明边界

step 50 的 actor update 与 checkpoint 保存已完成，但保存后 FSDP 参数重新装载并同步到
vLLM rollout 时仍差约 194MiB 显存，发生 OOM。因此 3.0 可以声明“完成可学习的 50-step
GRPO 工程验证并导出可加载 adapter”，不能声明“独立最终集准确率提升”。完成标记保留
`completed_with_post_checkpoint_rollout_sync_oom`，没有把异常隐藏为正常退出。

单卡 world size=1 下 FSDP 退化为 `NO_SHARD`。项目验证了 offload、rank-aware 状态、
checkpoint 和恢复合同，但没有多卡通信或扩展效率数据。

## 主要新增文件

```text
src/evidenceagent_mm/agentic.py       预算、轨迹和三态决策
src/evidenceagent_mm/verl_tools.py    VERL 工具边界
src/evidenceagent_mm/verl_dataset.py  Parquet 数据合同
scripts/verl_reward_v3.py             轨迹奖励
scripts/export_verl_lora_v3.py        FSDP → PEFT 导出
benchmarks/eamm_v3_hard/              hard cases 与证据库
configs/verl/tools_v3.json            工具 schema
```

## 从 2.0 升级

2.0 的 API、确定性检索、引用与拒答逻辑继续保留。3.0 是增量 Agentic/训练层，不要求
已有调用方立即迁移。需要训练时使用 v3 脚本；只需要可审计问答时仍可运行原有 API。

深入原理、完整排障过程与面试问答见
[3.0 学习与踩坑手册](V3_LEARNING_AND_INTERVIEW_ZH.md) 和
[完整开发复盘](V3_DEVELOPMENT.md)。
