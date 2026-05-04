"""Rotary Position Embedding (RoPE) for Llama-style models.

RoPE is memory-bandwidth bound; a pure PyTorch implementation is used here.
"""

import torch


def precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 500000.0,
                     device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin frequency tables for all positions up to max_seq_len."""
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freq)  # [max_seq_len, head_dim//2]
    angles = torch.cat([angles, angles], dim=-1)  # [max_seq_len, head_dim]
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap pairs of dimensions: [..., d] with [..., d//2] and [..., d//2:] swapped and negated."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q: [batch, num_q_heads, head_dim]
    k: [batch, num_kv_heads, head_dim]
    cos, sin: [batch, head_dim]
    Returns rotated q, k of same shapes.
    """
    cos_q = cos.unsqueeze(1)  # [batch, 1, head_dim]
    sin_q = sin.unsqueeze(1)
    q_rot = q * cos_q + rotate_half(q) * sin_q

    cos_k = cos.unsqueeze(1)  # [batch, 1, head_dim]
    sin_k = sin.unsqueeze(1)
    k_rot = k * cos_k + rotate_half(k) * sin_k

    return q_rot, k_rot
