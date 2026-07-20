# EvidenceAgent-MM：可验证多模态会议助手

用户可以问“谁在什么时候提出了什么方案、屏幕上是哪一页、依据是什么”。系统不会只生成一段摘要，而是返回逐主张引用、音视频时间戳、匿名说话人、屏幕页码、置信度和工具调用轨迹。

证据门有三种输出：

1. `answered`：证据充分，每个 claim 都能回放；
2. `needs_clarification`：问题存在可消除的指代歧义，系统提出具体追问；
3. `abstained`：缺少必要模态或独立支持，系统明确说明缺什么。

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make benchmark
eamm --db /tmp/eamm.db serve
```

浏览器打开 `http://127.0.0.1:8000`。未经鉴权时不要监听公网地址。

## 已验证边界

- 12 场、120 问的 CC0 合成 Bronze contract benchmark 已完整运行；
- 三态控制流和 Evidence Recall@5 均为 1.0；
- 这些数字不代表真实会议泛化能力；
- 当前置信度尚未校准，ECE-10 约为 0.413，必须在独立 validation set 上校准后才能作为产品分数。

架构、数据、模型和安全细节分别见 `docs/ARCHITECTURE.md`、`docs/DATASET_CARD.md`、`docs/MODEL_CARD.md` 和 `SECURITY.md`。
