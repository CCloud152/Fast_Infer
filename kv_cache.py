"""Block-based KV cache with PageAttention-style management."""

import torch


class KVCache:
    """
    Paged KV cache for a single sequence (or batch of sequences).
    Allocates physical blocks from a pre-allocated pool.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int, block_size: int, device: torch.device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.device = device

        self.num_blocks = (max_seq_len + block_size - 1) // block_size
        self.max_blocks_per_seq = self.num_blocks  # single sequence can use all blocks

        # Physical cache: [num_blocks, block_size, num_heads, head_dim] × 2 (K, V) × num_layers
        self.k = torch.zeros(num_layers, self.num_blocks, block_size, num_heads, head_dim,
                            dtype=torch.float16, device=device)
        self.v = torch.zeros(num_layers, self.num_blocks, block_size, num_heads, head_dim,
                            dtype=torch.float16, device=device)

        # Block table: maps logical block → physical block (-1 = unallocated)
        # Shape: [batch_size, max_blocks_per_seq]
        # We'll resize batch as needed
        self.block_table = torch.full((1, self.max_blocks_per_seq), -1,
                                      dtype=torch.int32, device=device)

        # Free block list
        self.free_blocks = list(range(self.num_blocks))
        self._allocated = 0     # total allocated logical blocks
        self.num_tokens = 0     # actual number of stored tokens (excluding padding)

    def _grow_batch(self, batch_size: int):
        """Expand block_table for larger batch."""
        if batch_size <= self.block_table.shape[0]:
            return
        new_table = torch.full((batch_size, self.max_blocks_per_seq), -1,
                               dtype=torch.int32, device=self.device)
        new_table[:self.block_table.shape[0]] = self.block_table
        self.block_table = new_table

    def alloc_block(self, batch_idx: int = 0) -> int:
        """Allocate a new block for a sequence. Returns logical block index."""
        if not self.free_blocks:
            raise RuntimeError("KV cache out of memory")
        phys_block = self.free_blocks.pop(0)
        # Find first free logical slot
        row = self.block_table[batch_idx]
        empty_slots = (row == -1).nonzero(as_tuple=True)[0]
        if len(empty_slots) == 0:
            raise RuntimeError("Sequence block table full")
        logical_idx = empty_slots[0].item()
        row[logical_idx] = phys_block
        self._allocated += 1
        return logical_idx

    def get_physical_blocks(self, batch_idx: int = 0) -> torch.Tensor:
        """Return tensor of physical block indices for a sequence."""
        return self.block_table[batch_idx]

    def store(self, layer_idx: int, logical_block: int, token_offset: int,
              k: torch.Tensor, v: torch.Tensor):
        """
        Store a single token's K, V into the cache.
        k, v: [num_heads, head_dim] FP16 (single token)
        """
        phys_block = self.block_table[0, logical_block].item()
        self.k[layer_idx, phys_block, token_offset] = k
        self.v[layer_idx, phys_block, token_offset] = v

    def store_batch(self, layer_idx: int, start_block: int, start_offset: int,
                    k: torch.Tensor, v: torch.Tensor):
        """
        Store multiple tokens' K, V into contiguous cache blocks.
        k, v: [num_tokens, num_heads, head_dim] FP16
        """
        num_tokens = k.shape[0]
        remaining = num_tokens
        block_idx = start_block
        offset = start_offset

        pos = 0
        while remaining > 0:
            phys_block = self.block_table[0, block_idx].item()
            slots_left = self.block_size - offset
            to_store = min(remaining, slots_left)

            self.k[layer_idx, phys_block, offset:offset + to_store] = k[pos:pos + to_store]
            self.v[layer_idx, phys_block, offset:offset + to_store] = v[pos:pos + to_store]

            pos += to_store
            remaining -= to_store
            offset = 0
            block_idx += 1

    def allocated_tokens(self, batch_idx: int = 0) -> int:
        """Number of tokens currently stored for a sequence."""
        return self._allocated * self.block_size  # upper bound; actual = sum of allocated blocks * block_size

    def reset(self):
        """Free all blocks (for a new sequence)."""
        self.block_table.fill_(-1)
        self.free_blocks = list(range(self.num_blocks))
        self._allocated = 0
        self.num_tokens = 0
