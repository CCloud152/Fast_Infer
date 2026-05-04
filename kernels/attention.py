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

    BLOCK_M = max(16, min(64, triton.next_power_of_2(seq_len)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(seq_len)))
    BLOCK_D = triton.next_power_of_2(head_dim)

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


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_table: torch.Tensor, block_size: int,
                    num_tokens: int,
                    sm_scale: float = None) -> torch.Tensor:
    """
    PagedAttention for decode phase.
    q: [batch, num_heads, head_dim] — single token query
    k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: [batch, max_blocks_per_seq] with -1 for unused slots
    num_tokens: actual number of valid tokens (excluding padding)
    Returns: [batch, num_heads, head_dim]
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_groups = num_heads // num_kv_heads

    outputs = []
    for b in range(batch):
        physical = block_table[b][block_table[b] >= 0].tolist()
        if not physical:
            outputs.append(torch.zeros(num_heads, head_dim, device=q.device, dtype=q.dtype))
            continue

        k_seq = k_cache[physical].reshape(-1, num_kv_heads, head_dim)
        v_seq = v_cache[physical].reshape(-1, num_kv_heads, head_dim)

        # Only use valid tokens, discard padding zeros
        k_seq = k_seq[:num_tokens]
        v_seq = v_seq[:num_tokens]

        # Expand KV heads for GQA
        k_seq = k_seq.repeat_interleave(kv_groups, dim=1)
        v_seq = v_seq.repeat_interleave(kv_groups, dim=1)

        scores = torch.einsum('hd,thd->ht', q[b].float(), k_seq.float()) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('ht,thd->hd', attn, v_seq.float()).to(q.dtype)
        outputs.append(out)

    return torch.stack(outputs)
