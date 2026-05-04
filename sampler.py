"""Temperature scaling, top-p (nucleus) sampling, and repetition penalty."""

import torch


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int],
                             penalty: float) -> torch.Tensor:
    """
    Penalize already-generated tokens to reduce repetition.
    logits: [batch, vocab_size]
    token_ids: list of token IDs already generated (across all batches)
    penalty: typically 1.0-1.2 (1.0 = no penalty, 1.2 = moderate)
    """
    if penalty <= 1.0 or not token_ids:
        return logits

    for token_id in set(token_ids):
        logits[:, token_id] = torch.where(
            logits[:, token_id] < 0,
            logits[:, token_id] * penalty,
            logits[:, token_id] / penalty,
        )
    return logits


def sample(logits: torch.Tensor, temperature: float = 0.6, top_p: float = 0.9,
           generated_ids: list[int] = None, repetition_penalty: float = 1.0) -> torch.Tensor:
    """
    Sample next token from logits.
    logits: [batch, vocab_size]
    Returns: [batch] token ids
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    # Repetition penalty
    if repetition_penalty > 1.0 and generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Temperature scaling
    scaled = logits / temperature

    # Softmax
    probs = torch.softmax(scaled, dim=-1)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
