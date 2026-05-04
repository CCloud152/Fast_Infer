"""RMSNorm Triton kernel for Llama models."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd(x_ptr, weight_ptr, output_ptr,
                   hidden_size: tl.constexpr, eps: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    """RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    offs = row * hidden_size + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / hidden_size
    rms = tl.math.rsqrt(mean_sq + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rms * weight

    tl.store(output_ptr + offs, out.to(output_ptr.dtype.element_ty), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    x: [..., hidden_size], FP16
    weight: [hidden_size], FP16
    Returns: [..., hidden_size], FP16
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    batch, hidden_size = x_flat.shape
    output = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    _rms_norm_fwd[(batch,)](x_flat, weight, output,
                              hidden_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output.reshape(orig_shape)
