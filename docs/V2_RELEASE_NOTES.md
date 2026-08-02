# EvidenceAgent-MM 2.0：新增能力、证据与边界

发布日期：2026-08-02
版本：`v2.0.0`
训练硬件：单张 NVIDIA RTX 4090 24 GiB

## 1. 为什么要做 2.0

1.0 证明了“回答必须能回到证据”的系统骨架：音频、屏幕、说话段被统一成
`EvidenceAtom`，Agent 根据证据充分性选择回答、追问或拒答。但确定性规则只能说明
合同可执行，不能让语言模型稳定地产出符合合同的结构化结果，也没有形成从用户纠错到
下一轮训练的数据闭环。

2.0 因此不推翻 1.0，而是在它上面增加两层：可验证后训练和受控数据飞轮。确定性
Agent 仍掌握最终决策权，Qwen adapter 学习 JSON、引用、grounding 和拒答行为。

## 2. 2.0 全量新增内容

### 2.1 版本化训练合同

- 新增 SFT、DPO、GRPO 三类 Pydantic/JSONL schema；
- 每条样本保留 session、speaker、时间戳、OCR 页面和 acoustic attributes；
- 按 session 切分 8/2/2，防止同一会议的问题泄漏到 train/test；
- 数据构建是确定性的，manifest 保存数量、配置和内容哈希。

### 2.2 真正串联的 SFT → DPO → GRPO

- SFT 从 Qwen3-1.7B 学习三态 JSON 和引用格式；
- DPO 从 SFT adapter 继续训练，学习 grounded answer 优于无引用/越权回答；
- GRPO 从 DPO adapter 继续训练，使用程序化 reward 做在线可验证优化；
- 每阶段 manifest 记录输入 adapter、基座 revision、库版本、GPU 和峰值显存，避免
  “三个阶段都从 base 独立开始”的伪流水线。

### 2.3 六类可测试 reward

reward 被写成无模型依赖的纯函数，分别检查：JSON 格式、三态 status、引用 ID、
grounding、拒答/追问和综合合同。非法 JSON 或未知引用 fail closed；单元测试覆盖合法、
缺字段、伪造引用和应拒答却回答等路径。

### 2.4 单卡训练和 DeepSpeed 证据

- 支持 BF16 LoRA、gradient checkpointing、断点恢复和 AutoDL 一键脚本；
- 提供 plain BF16、ZeRO-2 optimizer offload、ZeRO-3 parameter/optimizer offload；
- 单卡对照实测峰值显存为 8.97/4.73/3.48 GiB，吞吐为 2.023/1.355/0.286 samples/s；
- 这些结果只证明 Engine、offload 与显存—吞吐权衡，不声称多 GPU 分片经验。

### 2.5 反馈数据飞轮

- 用户反馈必须显式同意训练并声明可用 license；
- 内容哈希去重，避免同一纠错被多次放大；
- 导出 SFT/DPO 候选前执行 schema 和 citation 校验；
- 导出结果带 manifest 与 SHA-256，可追溯输入和输出。

### 2.6 发布与复现

- 新增 AutoDL bootstrap、preflight、smoke/full runner 和 adapter 发布脚本；
- 最终 GRPO LoRA 公开在 [Hugging Face](https://huggingface.co/jatshi/EvidenceAgent-MM-Qwen3-1.7B-GRPO-LoRA)；
- 新增训练手册、模型卡、数据卡、演示 GIF 和机器可读结果；
- 核心 51 项测试、Ruff、strict mypy 和 GitHub Actions 均作为发布门禁。

## 3. 已验证结果

| 项目 | Validation | Test |
| --- | ---: | ---: |
| Composite contract score | 0.920 | 0.920 |
| Valid JSON | 1.000 | 1.000 |
| Grounding | 1.000 | 1.000 |
| Citation | 0.800 | 0.800 |
| Abstention | 0.800 | 0.800 |

GRPO 完成 100 optimizer steps，平均 shaped reward 为 `0.7101`；前 20 步均值
`0.5532`，后 20 步 `0.7796`。测试集生成 P95 为 `6.281 s`。数据只有 120 个合成
问题，因此这些数字证明合同学习与工程闭环，不代表真实会议泛化能力。

## 4. 负结果也属于发布内容

- `top_k=1` 将状态准确率降到 0.20、召回降到 0.50，说明单证据截断会破坏跨模态回答；
- 去掉 graph 或 visual gate 在 Bronze 上均无变化，说明当前 benchmark 太干净，不能
  用它证明这两个模块有效；
- citation/abstention 仍只有 0.80，不能用 JSON 合法率 1.0 代替事实正确性。

## 5. 从代码角度看变更

| 路径 | 作用 |
| --- | --- |
| `src/evidenceagent_mm/training_contracts.py` | 三阶段数据与结果合同 |
| `src/evidenceagent_mm/training_data.py` | session 级切分与可复现数据构建 |
| `src/evidenceagent_mm/training_rewards.py` | 离线可验证组合 reward |
| `src/evidenceagent_mm/training.py` | LoRA/TRL 训练和 manifest |
| `src/evidenceagent_mm/feedback.py` | consent、license、去重与导出 |
| `scripts/autodl_v2_*.sh` | 单卡环境、预检、运行和恢复 |
| `configs/deepspeed/` | ZeRO-2/ZeRO-3 对照配置 |

## 6. 面试时最准确的表述

> 我没有让模型自己决定证据是否充分，而是保留确定性 evidence gate，再用
> Qwen3-1.7B LoRA 串联 SFT、DPO 和 GRPO 学习三态 JSON、引用与拒答合同。我在单张
> 4090 上记录了训练 manifest、固定 split、消融和 DeepSpeed 单卡对照，并把最终
> adapter、模型卡与可复现实验公开。当前结果证明合同学习，不冒充真实会议 SOTA。

完整原理、源码阅读、复现、故障和追问答案见
[从零手搓学习手册](tutorials/evidenceagent_mm_from_scratch_tutorial.md)。
