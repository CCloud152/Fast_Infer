"""FlashAttention (prefill) and paged attention (decode) for Llama models."""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    num_heads,
    seq_len, head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """FlashAttention v2 with causal masking."""
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = o_ptr + batch_idx * stride_ob + head_idx * stride_oh

    for start_m in range(0, seq_len, BLOCK_M):
        m_offs = start_m + offs_m
        m_mask = m_offs < seq_len

        # Reset per-Q-tile accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        q = tl.load(q_base + m_offs[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        q = q.to(tl.float32)

        for start_n in range(0, start_m + BLOCK_M, BLOCK_N):
            n_offs = start_n + offs_n
            n_mask = n_offs < seq_len

            k = tl.load(k_base + n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                       mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
            k = k.to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # Causal mask
            qk = tl.where(m_offs[:, None] >= n_offs[None, :], qk, float("-inf"))
            qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v = tl.load(v_base + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                       mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
            acc += tl.dot(p, v.to(tl.float32))

            m_i = m_new

        result = (acc / l_i[:, None]).to(tl.float16)

        tl.store(o_base + m_offs[:, None] * stride_om + offs_d[None, :] * stride_od,
                result, mask=m_mask[:, None] & (offs_d[None, :] < head_dim))


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    sm_scale: float = None) -> torch.Tensor:
    """
    FlashAttention for prefill phase.
    q, k, v: [batch, num_heads, seq_len, head_dim] FP16
    Returns: [batch, num_heads, seq_len, head_dim] FP16
    """
    batch, num_heads, seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    BLOCK_D = triton.next_power_of_2(head_dim)
    # Shared memory budget: RTX 5060 has 99 KB / SM.
    # Q/K/V tiles alone need 3 * BLOCK * BLOCK_D * 2 bytes each.
    # Keep blocks small enough to fit with accumulator and softmax workspace.
    BLOCK_M = max(16, min(32, triton.next_power_of_2(seq_len)))
    BLOCK_N = max(16, min(32, triton.next_power_of_2(seq_len)))

    grid = (batch * num_heads,)
    o = torch.empty_like(q)

    # We pass num_heads as num_warps implicitly; grid shape handles it
    _flash_attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        num_heads,
        seq_len, head_dim,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o


@triton.jit
def _paged_attention_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr, block_table_ptr,
    num_tokens,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_btb, stride_btl,
    num_q_heads, num_kv_heads,
    num_blocks,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """PagedAttention: one program per (batch, q_head), online softmax over KV blocks."""
    pid = tl.program_id(0)
    batch_idx = pid // num_q_heads
    q_head_idx = pid % num_q_heads

    # GQA routing: each Q head reads a single KV head (no materialized expansion)
    kv_groups = num_q_heads // num_kv_heads
    kv_head_idx = q_head_idx // kv_groups

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Load query vector
    q = tl.load(q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh + offs_d * stride_qd,
                mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_s = tl.arange(0, block_size)

    for blk_idx in range(num_blocks):
        phys = tl.load(block_table_ptr + batch_idx * stride_btb + blk_idx * stride_btl)

        token_base = blk_idx * block_size
        token_pos = token_base + offs_s
        token_mask = token_pos < num_tokens

        # Load K tile [block_size, head_dim] for the mapped KV head
        k_base = k_cache_ptr + phys * stride_kb + kv_head_idx * stride_kh
        k = tl.load(k_base + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Scores: q @ K^T * sm_scale  →  [block_size]
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(token_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha

        # Load V tile and accumulate P @ V
        v_base = v_cache_ptr + phys * stride_vb + kv_head_idx * stride_vh
        v = tl.load(v_base + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # Write output
    out = (acc / l_i).to(tl.float16)
    tl.store(o_ptr + batch_idx * stride_ob + q_head_idx * stride_oh + offs_d * stride_od,
             out, mask=d_mask)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_table: torch.Tensor, block_size: int,
                    num_tokens: int,
                    sm_scale: float = None) -> torch.Tensor:
    """
    PagedAttention for decode phase (Triton kernel).
    q: [batch, num_q_heads, head_dim] — single token query
    k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: [batch, max_blocks_per_seq] with -1 for unused slots
    block_size: number of tokens per cache block
    num_tokens: actual number of valid tokens (excluding padding)
    Returns: [batch, num_q_heads, head_dim]
    """
    if num_tokens == 0:
        return torch.zeros_like(q)

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    num_blocks = (num_tokens + block_size - 1) // block_size

    BLOCK_D = triton.next_power_of_2(head_dim)
    o = torch.empty_like(q)

    grid = (batch * num_q_heads,)
    _paged_attention_kernel[grid](
        q, k_cache, v_cache, o, block_table,
        num_tokens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        block_table.stride(0), block_table.stride(1),
        num_q_heads, num_kv_heads,
        num_blocks,
        head_dim, block_size,
        sm_scale,
        BLOCK_D=BLOCK_D,
    )
    return o
