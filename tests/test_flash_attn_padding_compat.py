from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compat"))

from flash_attn.bert_padding import pad_input, unpad_input  # noqa: E402


def test_padding_compat_roundtrip_and_gradient() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int64)

    packed, indices, cu_seqlens, max_seqlen, used_seqlens = unpad_input(values, mask)
    restored = pad_input(packed, indices, batch=2, seqlen=4)

    assert packed.shape == (4, 3)
    assert cu_seqlens.tolist() == [0, 2, 4]
    assert max_seqlen == 2
    assert used_seqlens.tolist() == [2, 2]
    assert torch.equal(restored[mask.bool()], values.detach()[mask.bool()])
    assert torch.count_nonzero(restored[~mask.bool()]) == 0

    packed.sum().backward()
    expected_grad = mask.unsqueeze(-1).expand_as(values)
    assert torch.equal(values.grad, expected_grad)
