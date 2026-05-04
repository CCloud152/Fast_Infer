"""LlamaForCausalLM assembled from quantized weights and Triton kernels."""

import torch
from fast_infer.config import LlamaConfig
from fast_infer.kernels.rms_norm import rms_norm
from fast_infer.kernels.rope import precompute_freqs, apply_rope
from fast_infer.kernels.matmul import int4_matmul, dequantize_all
from fast_infer.kernels.mlp import swiglu_mlp
from fast_infer.kernels.attention import flash_attention, paged_attention
from fast_infer.kv_cache import KVCache


class LlamaForCausalLM:
    def __init__(self, config: LlamaConfig, weights: dict, device: torch.device = None,
                 memory_efficient: bool = True):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_efficient = memory_efficient

        if not memory_efficient:
            weights = dequantize_all(weights)
        self.weights = weights
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_freqs(
            config.head_dim, config.max_position_embeddings, config.rope_theta, self.device
        )

        # KV cache (created on first use with actual batch size)
        self.kv_cache = None

    def _ensure_kv_cache(self, batch_size: int):
        if self.kv_cache is None or self.kv_cache.block_table.shape[0] < batch_size:
            self.kv_cache = KVCache(
                self.num_layers, self.num_kv_heads, self.head_dim,
                self.config.max_seq_len, self.config.block_size, self.device
            )
            self.kv_cache._grow_batch(batch_size)
            self._token_count = 0
            self._allocated_blocks = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass handling both prefill and decode.
        input_ids: [batch, seq_len] (prefill) or [batch, 1] (decode)
        Returns logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        self._ensure_kv_cache(batch_size)

        is_prefill = seq_len > 1 or self._token_count == 0

        # Embedding
        h = self.weights["embed_tokens"][input_ids]  # [batch, seq_len, hidden_size]
        h = h.to(torch.float16)

        # Position IDs
        if is_prefill:
            positions = torch.arange(seq_len, device=self.device)
            self._token_count = seq_len
        else:
            positions = torch.tensor([self._token_count], device=self.device)
            self._token_count += 1

        cos_pos = self.cos[positions]  # [seq_len, head_dim]
        sin_pos = self.sin[positions]

        for layer_idx in range(self.num_layers):
            layer_w = self.weights["layers"][layer_idx]

            # --- Attention block ---
            residual = h
            h_normed = rms_norm(h, layer_w["input_layernorm"], self.config.rms_norm_eps)

            # Q, K, V projections (INT4)
            q = int4_matmul(h_normed, layer_w["self_attn"]["q_proj"])
            k = int4_matmul(h_normed, layer_w["self_attn"]["k_proj"])
            v = int4_matmul(h_normed, layer_w["self_attn"]["v_proj"])

            # Reshape to multi-head
            q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            # Apply RoPE
            q, k = apply_rope(q, k, cos_pos, sin_pos)

            # Transpose to [batch, heads, seq_len, head_dim] for attention
            q_attn = q.transpose(1, 2)  # [batch, num_q_heads, seq_len, head_dim]
            k_attn = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
            v_attn = v.transpose(1, 2)

            if is_prefill:
                # For GQA: repeat KV heads to match Q heads
                k_attn = k_attn.repeat_interleave(self.kv_groups, dim=1)
                v_attn = v_attn.repeat_interleave(self.kv_groups, dim=1)

                attn_out = flash_attention(q_attn, k_attn, v_attn)  # [batch, num_q_heads, seq_len, head_dim]
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

                # Store K, V to cache
                self._store_kv_prefill(layer_idx, k, v)
            else:
                # Store single token K, V
                self._store_kv_decode(layer_idx, k.squeeze(1), v.squeeze(1))

                # Paged attention with cache
                attn_out = paged_attention(
                    q_attn.squeeze(2),  # [batch, num_q_heads, head_dim]
                    self.kv_cache.k[layer_idx],
                    self.kv_cache.v[layer_idx],
                    self.kv_cache.block_table,
                    self.config.block_size,
                    self.kv_cache.num_tokens,
                )
                attn_out = attn_out.view(batch_size, self.hidden_size).unsqueeze(1)

            # O projection
            h_out = int4_matmul(attn_out, layer_w["self_attn"]["o_proj"])
            h = residual + h_out

            # --- MLP block ---
            residual = h
            h_normed = rms_norm(h, layer_w["post_attention_layernorm"], self.config.rms_norm_eps)
            h_out = swiglu_mlp(
                h_normed,
                layer_w["mlp"]["gate_proj"],
                layer_w["mlp"]["up_proj"],
                layer_w["mlp"]["down_proj"],
            )
            h = residual + h_out

        # Final RMSNorm
        h_normed = rms_norm(h, self.weights["norm"], self.config.rms_norm_eps)

        # LM head (stay in FP16 — the vocab matmul is expensive enough without FP32 upcast)
        logits = torch.matmul(h_normed, self.weights["lm_head"].T)
        return logits.float()  # cast to FP32 only for the (small) output

    def _store_kv_prefill(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store prefill K, V into cache. k, v: [batch, seq_len, num_kv_heads, head_dim]"""
        batch_size, seq_len = k.shape[0], k.shape[1]
        tokens_per_block = self.config.block_size
        num_needed = (seq_len + tokens_per_block - 1) // tokens_per_block

        # Track token count on first layer
        if layer_idx == 0:
            self.kv_cache.num_tokens = seq_len

        for b in range(batch_size):
            while (self.kv_cache.block_table[b] >= 0).sum() < num_needed:
                self.kv_cache.alloc_block(b)

            for blk in range(num_needed):
                start = blk * tokens_per_block
                end = min(start + tokens_per_block, seq_len)
                n_tokens = end - start

                phys = self.kv_cache.block_table[b, blk].item()
                self.kv_cache.k[layer_idx, phys, :n_tokens] = k[b, start:end]
                self.kv_cache.v[layer_idx, phys, :n_tokens] = v[b, start:end]

    def _store_kv_decode(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store single decode token K, V. k, v: [batch, num_kv_heads, head_dim]"""
        batch_size = k.shape[0]
        token_pos = self._token_count - 1  # position we just computed
        tokens_per_block = self.config.block_size
        block_idx = token_pos // tokens_per_block
        offset = token_pos % tokens_per_block

        if layer_idx == 0:
            self.kv_cache.num_tokens += 1

        for b in range(batch_size):
            while (self.kv_cache.block_table[b] >= 0).sum() <= block_idx:
                self.kv_cache.alloc_block(b)
            phys = self.kv_cache.block_table[b, block_idx].item()
            self.kv_cache.k[layer_idx, phys, offset] = k[b]
            self.kv_cache.v[layer_idx, phys, offset] = v[b]

    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()
        self._token_count = 0
