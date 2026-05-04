"""High-level generation engine."""

import torch
from pathlib import Path
from fast_infer.config import LlamaConfig
from fast_infer.loader import load_weights
from fast_infer.model import LlamaForCausalLM
from fast_infer.sampler import sample


class InferenceEngine:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.config = LlamaConfig()

        print(f"Loading model from {model_dir}...")
        self.weights = load_weights(model_dir, str(self.device))

        print("Building model...")
        self.model = LlamaForCausalLM(self.config, self.weights, self.device)

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, clean_up_tokenization_spaces=False
        )

        if self.tokenizer.pad_token is None:
            eos = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def generate(self, prompt: str, max_new_tokens: int = None,
                 temperature: float = None, top_p: float = None,
                 repetition_penalty: float = 1.0) -> str:
        """Generate text from a prompt string."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        self.model.reset_cache()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = []

        with torch.no_grad():
            logits = self.model.forward(input_ids)
            next_token = sample(logits[:, -1, :], temperature, top_p,
                               generated_ids=generated,
                               repetition_penalty=repetition_penalty)
            generated.append(next_token.item())

            for _ in range(max_new_tokens - 1):
                if next_token.item() in self.eos_ids:
                    break
                logits = self.model.forward(next_token.unsqueeze(0))
                next_token = sample(logits[:, -1, :], temperature, top_p,
                                   generated_ids=generated,
                                   repetition_penalty=repetition_penalty)
                generated.append(next_token.item())

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = None,
                        temperature: float = None, top_p: float = None,
                        repetition_penalty: float = 1.0):
        """Generator yielding tokens one at a time."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        self.model.reset_cache()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = []

        with torch.no_grad():
            logits = self.model.forward(input_ids)
            next_token = sample(logits[:, -1, :], temperature, top_p,
                               generated_ids=generated,
                               repetition_penalty=repetition_penalty)
            generated.append(next_token.item())
            yield next_token.item()

            for _ in range(max_new_tokens - 1):
                if next_token.item() in self.eos_ids:
                    break
                logits = self.model.forward(next_token.unsqueeze(0))
                next_token = sample(logits[:, -1, :], temperature, top_p,
                                   generated_ids=generated,
                                   repetition_penalty=repetition_penalty)
                generated.append(next_token.item())
                yield next_token.item()
