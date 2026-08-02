"""Padding-only compatibility namespace for VERL.

This project deliberately uses PyTorch SDPA, not FlashAttention kernels.  VERL
0.8.0 still imports ``flash_attn.bert_padding`` for generic packing helpers;
the sibling module provides only that small, tested API.
"""
