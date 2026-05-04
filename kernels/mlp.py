"""SwiGLU MLP with fused SiLU activation."""

import torch
import triton
import triton.language as tl
from fast_infer.kernels.matmul import int4_matmul


@triton.jit
def _silu_mul_fwd(gate_ptr, up_ptr, out_ptr, n_elements: tl.constexpr,
                   BLOCK: tl.constexpr):
    """Fused SiLU(gate) * up, element-wise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu = gate * tl.sigmoid(gate)
    out = silu * up

    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up. Both inputs [..., D]."""
    flat_gate = gate.reshape(-1)
    flat_up = up.reshape(-1)
    out = torch.empty_like(flat_gate)
    n = flat_gate.numel()
    BLOCK = min(triton.next_power_of_2(n), 1024)
    _silu_mul_fwd[(triton.cdiv(n, BLOCK),)](flat_gate, flat_up, out, n, BLOCK=BLOCK)
    return out.reshape_as(gate)


def swiglu_mlp(x: torch.Tensor,
               gate_w: torch.Tensor,
               up_w: torch.Tensor,
               down_w: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU MLP forward.
    x: [batch, hidden_size] FP16
    gate_w, up_w, down_w: [intermediate_size, hidden_size] or [hidden_size, intermediate_size] FP16
    Returns: [batch, hidden_size] FP16
    """
    gate = int4_matmul(x, gate_w)   # [batch, intermediate_size]
    up = int4_matmul(x, up_w)        # [batch, intermediate_size]
    act = silu_mul(gate, up)         # fused SiLU + multiply
    out = int4_matmul(act, down_w)   # [batch, hidden_size]
    return out
