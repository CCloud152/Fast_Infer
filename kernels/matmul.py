"""INT4 / FP16 matrix multiplication with fused dequantization.

Speed mode (default): weights pre-dequantized to FP16 via dequantize_all(), then torch.matmul.
Memory mode (memory_efficient=True): weights stay INT4 packed, fused Triton kernel dequants on-the-fly.
"""

import torch
import triton
import triton.language as tl


# ---- Fused INT4 dequant + matmul Triton kernel ----

@triton.jit
def _fused_int4_matmul_kernel(
    a_ptr, w_ptr, scales_ptr, c_ptr,
    M, N, K,
    groupsize: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HALF_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Compute C = A @ dequant(W).T in a single fused kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Even/odd K offsets to avoid nibble interleaving
    offs_even = tl.arange(0, HALF_K) * 2      # [0, 2, 4, ..., BLOCK_K-2]
    offs_odd  = offs_even + 1                  # [1, 3, 5, ..., BLOCK_K-1]
    offs_widx = tl.arange(0, HALF_K)           # [0, 1, 2, ..., HALF_K-1] packed index

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        # Load A tile — split into even and odd column sets
        k_even = k_start + offs_even
        k_odd = k_start + offs_odd
        m_mask = offs_m < M

        a_even = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_even[None, :] * stride_ak,
            mask=m_mask[:, None] & (k_even[None, :] < K), other=0.0
        )  # [BLOCK_M, half_k]

        a_odd = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_odd[None, :] * stride_ak,
            mask=m_mask[:, None] & (k_odd[None, :] < K), other=0.0
        )  # [BLOCK_M, half_k]

        # Load packed W — one byte holds two int4 values
        w_packed_idx = (k_start // 2) + offs_widx  # k//2 for the packed dimension
        w_packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + w_packed_idx[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (w_packed_idx[None, :] < K // 2), other=0
        )  # [BLOCK_N, half_k] uint8 (implicit from int8 storage)

        # Unpack nibbles
        w_low = w_packed & 0xF       # lower 4 bits → even K positions
        w_high = (w_packed >> 4) & 0xF  # upper 4 bits → odd K positions

        # Unsigned [0,15] → signed [-8,7]
        w_low = tl.where(w_low >= 8, w_low - 16, w_low)
        w_high = tl.where(w_high >= 8, w_high - 16, w_high)

        # Load scale for this group (one scale per groupsize K elements)
        scale_idx = k_start // groupsize
        scale = tl.load(
            scales_ptr + offs_n * stride_sn + scale_idx * stride_sg,
            mask=offs_n < N, other=0.0
        )  # [BLOCK_N]

        # Apply scale and accumulate
        w_low_fp = (w_low * scale[:, None]).to(tl.float16)    # [BLOCK_N, half_k]
        w_high_fp = (w_high * scale[:, None]).to(tl.float16)  # [BLOCK_N, half_k]

        acc += tl.dot(a_even.to(tl.float16), tl.trans(w_low_fp))
        acc += tl.dot(a_odd.to(tl.float16), tl.trans(w_high_fp))

    # Store output tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _fused_int4_matmul(x: torch.Tensor, packed: torch.Tensor,
                        scales: torch.Tensor, groupsize: int = 128) -> torch.Tensor:
    """Dispatch fused INT4 matmul via Triton kernel."""
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x.shape
    N = packed.shape[0]

    assert K % groupsize == 0, f"K ({K}) must be multiple of groupsize ({groupsize})"
    assert packed.shape[1] == K // 2, f"Weight packed dim mismatch: {packed.shape[1]} != {K // 2}"

    c = torch.empty(M, N, dtype=torch.float16, device=x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = groupsize  # one group per K tile — simplest scale handling
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fused_int4_matmul_kernel[grid](
        x, packed, scales, c,
        M, N, K,
        groupsize,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HALF_K=BLOCK_K // 2,
        GROUP_M=GROUP_M,
    )

    return c.reshape(*orig_shape[:-1], N)


# ---- FP16 path (for pre-dequantized weights) ----

def _fp16_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Plain FP16 matmul for dequantized weights."""
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    out = torch.matmul(x.to(w.dtype), w.T)
    return out.reshape(*orig_shape[:-1], w.shape[0])


# ---- Public API ----

def int4_matmul(x: torch.Tensor, w, scales=None, groupsize: int = 128) -> torch.Tensor:
    """
    FP16 input × weight → FP16 output.

    w can be:
      - FP16 tensor [N, K] (speed mode, already dequantized)
      - tuple (packed_int8, scales) for memory mode → uses fused Triton kernel

    Returns: [..., N] FP16
    """
    if isinstance(w, tuple):
        packed, scales = w
        return _fused_int4_matmul(x, packed, scales, groupsize)
    else:
        return _fp16_matmul(x, w)


# ---- Weight preparation ----

def dequantize_all(weights: dict) -> dict:
    """Dequantize all INT4 linear weights to FP16 in-place. Frees INT4 memory."""
    for layer in weights["layers"]:
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = layer["self_attn"][proj_name]
            if isinstance(w, tuple):
                layer["self_attn"][proj_name] = _dequant_storage(*w)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            w = layer["mlp"][proj_name]
            if isinstance(w, tuple):
                layer["mlp"][proj_name] = _dequant_storage(*w)
    torch.cuda.empty_cache()
    return weights


def _dequant_storage(packed: torch.Tensor, scales: torch.Tensor,
                     groupsize: int = 128) -> torch.Tensor:
    """Dequantize INT4 packed weights to FP16 for permanent storage."""
    N, K_half = packed.shape
    K = K_half * 2
    num_groups = K // groupsize

    packed_int = packed.to(torch.uint8)
    low = (packed_int & 0xF).to(torch.int8)
    high = ((packed_int >> 4) & 0xF).to(torch.int8)

    low = torch.where(low >= 8, low - 16, low).to(torch.float16)
    high = torch.where(high >= 8, high - 16, high).to(torch.float16)

    w_q = torch.stack([low, high], dim=-1).reshape(N, K)
    w_q = w_q.reshape(N, num_groups, groupsize)
    w = w_q.float() * scales.unsqueeze(-1).float()
    return w.reshape(N, K).to(torch.float16)
