"""INT4 / FP16 matrix multiplication.

Supports two modes:
1. Speed mode: weights pre-dequantized to FP16 via dequantize_all()
2. Memory-efficient mode: weights kept as (packed_int8, scales) tuples, dequantized on-the-fly
"""

import torch


def dequantize_all(weights: dict) -> dict:
    """Dequantize all INT4 linear weights to FP16 in-place. Frees INT4 memory."""
    for layer in weights["layers"]:
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = layer["self_attn"][proj_name]
            if isinstance(w, tuple):
                layer["self_attn"][proj_name] = _dequant_pytorch(*w)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            w = layer["mlp"][proj_name]
            if isinstance(w, tuple):
                layer["mlp"][proj_name] = _dequant_pytorch(*w)

    torch.cuda.empty_cache()
    return weights


def _dequant_pytorch(packed: torch.Tensor, scales: torch.Tensor,
                     groupsize: int = 128) -> torch.Tensor:
    """Dequantize INT4 packed weights to FP16."""
    N, K_half = packed.shape
    K = K_half * 2
    num_groups = K // groupsize

    packed_int = packed.to(torch.uint8)
    low = (packed_int & 0xF).to(torch.int8)
    high = ((packed_int >> 4) & 0xF).to(torch.int8)

    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    w_q = torch.stack([low, high], dim=-1).reshape(N, K)
    w_q = w_q.reshape(N, num_groups, groupsize)
    w = w_q.float() * scales.unsqueeze(-1).float()
    return w.reshape(N, K).to(torch.float16)


def int4_matmul(x: torch.Tensor, w, scales=None, groupsize: int = 128) -> torch.Tensor:
    """
    FP16 input × weight → FP16 output.
    w: FP16 tensor [N, K] (speed mode) OR INT8 packed tensor with scales in tuple (memory mode)
    Returns: [..., N] FP16
    """
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])

    if isinstance(w, tuple):
        packed, scales = w
        w_fp16 = _dequant_pytorch(packed, scales, groupsize)
    else:
        w_fp16 = w

    out = torch.matmul(x.to(w_fp16.dtype), w_fp16.T)
    return out.reshape(*orig_shape[:-1], w_fp16.shape[0])
