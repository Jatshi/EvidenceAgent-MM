"""Autograd-safe padding helpers compatible with ``flash_attn.bert_padding``.

The implementation is derived from FlashAttention's Apache-2.0 licensed
``bert_padding.py`` and mirrors the fallback shipped by VERL for NPU devices.
No attention kernel is provided or claimed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rearrange(values: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
    """Implement the three reshape-only einops patterns used by this API."""
    if pattern == "b ... -> b (...)":
        return values.reshape(values.shape[0], -1)
    if pattern == "b s ... -> (b s) ...":
        return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])
    if pattern == "(b s) ... -> b s ...":
        batch = axes["b"]
        return values.reshape(batch, values.shape[0] // batch, *values.shape[1:])
    raise NotImplementedError(f"unsupported padding compatibility pattern: {pattern}")


class _IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("values must have at least two dimensions")
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = values.shape[0]
        other_shape = values.shape[1:]
        second_dim = other_shape.numel()
        flat = rearrange(values, "b ... -> b (...)")
        gather_index = indices[:, None].expand(-1, second_dim)
        gathered = torch.gather(flat, 0, gather_index)
        return gathered.reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        flat_grad = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            (ctx.first_axis_dim, flat_grad.shape[1]),
            device=flat_grad.device,
            dtype=flat_grad.dtype,
        )
        scatter_index = indices[:, None].expand(-1, flat_grad.shape[1])
        grad_input.scatter_(0, scatter_index, flat_grad)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class _IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int
    ) -> torch.Tensor:
        if indices.ndim != 1 or values.ndim < 2:
            raise ValueError("indices must be 1-D and values must be at least 2-D")
        ctx.save_for_backward(indices)
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = _IndexFirstAxis.apply
index_put_first_axis = _IndexPutFirstAxis.apply


def pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    """Restore packed token rows to a zero-padded ``(batch, seqlen, ...)`` tensor."""
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: torch.Tensor | None = None,
):
    """Pack valid token rows and return FlashAttention-compatible metadata."""
    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    packed = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)
    return packed, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
