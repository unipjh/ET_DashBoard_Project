from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AdditiveAttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.query(torch.tanh(self.projection(sequence))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class UserEncoder(nn.Module):
    """Encode a recent clicked/read sequence into one 768d user vector."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        attention_hidden_dim: int = 128,
        dropout: float = 0.1,
        max_history: int = 20,
    ):
        super().__init__()
        self.max_history = max_history
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pooling = AdditiveAttentionPooling(embedding_dim, attention_hidden_dim)

    def forward(self, history_vectors: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if history_vectors.dim() == 2:
            history_vectors = history_vectors.unsqueeze(0)
        if history_vectors.size(1) > self.max_history:
            history_vectors = history_vectors[:, -self.max_history :, :]
            if mask is not None:
                mask = mask[:, -self.max_history :]

        key_padding_mask = None if mask is None else ~mask.bool()
        attended, _ = self.self_attention(
            history_vectors.float(),
            history_vectors.float(),
            history_vectors.float(),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        sequence = self.layer_norm(history_vectors.float() + attended)
        user_vector = self.pooling(sequence, mask)
        return F.normalize(user_vector, p=2, dim=-1, eps=1e-12)
