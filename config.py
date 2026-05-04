from dataclasses import dataclass


@dataclass
class LlamaConfig:
    # Architecture (Llama 3.2 3B Instruct)
    # From config.json: 24 Q heads × 128 head_dim = 3072 hidden_size
    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    kv_groups: int = 3                    # 24 / 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True

    # Inference defaults
    max_seq_len: int = 4096
    block_size: int = 64
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
