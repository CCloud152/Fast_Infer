"""Weight loading and INT4 quantization for Llama 3.2 models."""

import json
import torch
from pathlib import Path
from safetensors import safe_open


def quantize_int4_groupwise(
    w: torch.Tensor, groupsize: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric groupwise INT4 quantization.
    w: [out_features, in_features], FP16 on GPU
    Returns (packed_int4, scales) both on GPU.
      - packed_int4: int8 [out_features, in_features // 2]
      - scales:      fp16 [out_features, in_features // groupsize]
    """
    out_features, in_features = w.shape
    w = w.reshape(out_features, -1, groupsize)  # [out, in//g, g]

    # Per-group scale: max absolute value / 7
    w_max = w.abs().amax(dim=-1, keepdim=True)  # [out, in//g, 1]
    w_max = torch.clamp(w_max, min=1e-6)
    scale = w_max / 7.0  # fp16

    # Quantize: round(w / scale), clamp to [-8, 7], then shift to [0, 15]
    w_q = torch.round(w / scale.to(w.dtype))  # [-8, 7]
    w_q = torch.clamp(w_q, -8, 7).to(torch.int8)  # signed int8 but values in [-8,7]

    # Pack two int4 values per int8 byte
    # Reshape so that we can pair adjacent values along groupsize dim
    w_q = w_q.reshape(out_features, -1, groupsize // 2, 2)  # [out, in//g * g//2, 2]
    # Lower nibble (bits 0-3): first value (low = w_q[..., 0] & 0xF)
    # Upper nibble (bits 4-7): second value (high = (w_q[..., 1] & 0xF) << 4)
    packed = (w_q[..., 0] & 0xF) | ((w_q[..., 1] & 0xF) << 4)
    packed = packed.reshape(out_features, in_features // 2).to(torch.int8)

    scale = scale.reshape(out_features, in_features // groupsize).to(torch.float16)

    return packed, scale


def is_linear_weight(name: str) -> bool:
    """Check if a weight tensor is a linear projection that should be quantized."""
    linear_suffixes = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    )
    return name.endswith(linear_suffixes)


def load_weights(model_dir: str, device: str = "cuda") -> dict:
    """
    Load all safetensors from a HuggingFace model directory.
    Quantize linear layers to INT4; keep embeddings and norms in FP16.

    Returns structured dict:
      { "embed_tokens": FP16 tensor,
        "layers": [ { "input_layernorm", "self_attn": {...}, "post_attention_layernorm", "mlp": {...} }, ... ],
        "norm": FP16 tensor,
        "lm_head": FP16 tensor }
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        num_layers = config["num_hidden_layers"]
    else:
        num_layers = 28

    # Find all safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    index_json = model_dir / "model.safetensors.index.json"

    if index_json.exists():
        with open(index_json) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        file_set = set(weight_map.values())
        safetensor_files = [model_dir / f for f in file_set]

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    # Determine device
    gpu_device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    cpu_device = torch.device("cpu")

    # Initialize result structure
    embed_tokens = None
    lm_head = None
    norm = None
    layers = [{
        "input_layernorm": None,
        "self_attn": {"q_proj": None, "k_proj": None, "v_proj": None, "o_proj": None},
        "post_attention_layernorm": None,
        "mlp": {"gate_proj": None, "up_proj": None, "down_proj": None},
    } for _ in range(num_layers)]

    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if key == "model.embed_tokens.weight":
                    embed_tokens = tensor.to(dtype=torch.float16, device=gpu_device)
                elif key == "lm_head.weight":
                    lm_head = tensor.to(dtype=torch.float16, device=gpu_device)
                elif key == "model.norm.weight":
                    norm = tensor.to(dtype=torch.float16, device=gpu_device)
                elif "self_attn" in key:
                    layer_idx = int(key.split(".")[2])
                    if key.endswith("q_proj.weight"):
                        layers[layer_idx]["self_attn"]["q_proj"] = _process_linear(tensor, gpu_device)
                    elif key.endswith("k_proj.weight"):
                        layers[layer_idx]["self_attn"]["k_proj"] = _process_linear(tensor, gpu_device)
                    elif key.endswith("v_proj.weight"):
                        layers[layer_idx]["self_attn"]["v_proj"] = _process_linear(tensor, gpu_device)
                    elif key.endswith("o_proj.weight"):
                        layers[layer_idx]["self_attn"]["o_proj"] = _process_linear(tensor, gpu_device)
                elif "mlp" in key:
                    layer_idx = int(key.split(".")[2])
                    if key.endswith("gate_proj.weight"):
                        layers[layer_idx]["mlp"]["gate_proj"] = _process_linear(tensor, gpu_device)
                    elif key.endswith("up_proj.weight"):
                        layers[layer_idx]["mlp"]["up_proj"] = _process_linear(tensor, gpu_device)
                    elif key.endswith("down_proj.weight"):
                        layers[layer_idx]["mlp"]["down_proj"] = _process_linear(tensor, gpu_device)
                elif "input_layernorm" in key:
                    layer_idx = int(key.split(".")[2])
                    layers[layer_idx]["input_layernorm"] = tensor.to(dtype=torch.float16, device=gpu_device)
                elif "post_attention_layernorm" in key:
                    layer_idx = int(key.split(".")[2])
                    layers[layer_idx]["post_attention_layernorm"] = tensor.to(dtype=torch.float16, device=gpu_device)

    # Tie lm_head to embed_tokens if missing
    if lm_head is None and embed_tokens is not None:
        lm_head = embed_tokens

    return {
        "embed_tokens": embed_tokens,
        "layers": layers,
        "norm": norm,
        "lm_head": lm_head,
    }


def _process_linear(tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a linear weight tensor to INT4. Returns (packed_int4, scales)."""
    w = tensor.to(dtype=torch.float16, device=device)
    return quantize_int4_groupwise(w)


def dequantize_int4(
    packed: torch.Tensor, scales: torch.Tensor, groupsize: int = 128
) -> torch.Tensor:
    """
    Dequantize INT4 packed weights back to FP16.
    packed: int8 [out_features, in_features // 2]
    scales: fp16 [out_features, in_features // groupsize]
    Returns: fp16 [out_features, in_features]
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2
    num_groups = scales.shape[1]

    # Unpack int8 -> two int4 values
    low = (packed & 0xF).to(torch.int8)  # lower nibble
    high = ((packed >> 4) & 0xF).to(torch.int8)  # upper nibble

    # Convert from [0,15] unsigned back to [-8,7] signed
    # values >= 8 are negative in 4-bit two's complement
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    # Interleave low and high: low[0], high[0], low[1], high[1], ...
    w_q = torch.stack([low, high], dim=-1).reshape(out_features, -1)  # [out, in]

    # Reshape to groups and apply scales
    w_q = w_q.reshape(out_features, num_groups, groupsize)
    w = w_q.float() * scales.unsqueeze(-1).float()
    w = w.reshape(out_features, in_features)

    return w.to(torch.float16)
