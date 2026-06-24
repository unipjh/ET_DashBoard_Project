from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NewsEncoder(nn.Module):
    """Combine title, summary, and category views into one 768d news vector."""

    def __init__(self, embedding_dim: int = 768, category_dim: int = 16, attention_hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.category_dim = category_dim
        self.category_projection = nn.Linear(category_dim, embedding_dim)
        self.view_attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(
        self,
        title_embedding: torch.Tensor,
        summary_embedding: torch.Tensor,
        category_onehot: torch.Tensor,
    ) -> torch.Tensor:
        if title_embedding.dim() == 1:
            title_embedding = title_embedding.unsqueeze(0)
        if summary_embedding.dim() == 1:
            summary_embedding = summary_embedding.unsqueeze(0)
        if category_onehot.dim() == 1:
            category_onehot = category_onehot.unsqueeze(0)

        category_embedding = self.category_projection(category_onehot.float())
        views = torch.stack(
            [
                title_embedding.float(),
                summary_embedding.float(),
                category_embedding,
            ],
            dim=1,
        )
        attention_logits = self.view_attention(views).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        news_vector = torch.sum(views * attention_weights, dim=1)
        return F.normalize(news_vector, p=2, dim=-1, eps=1e-12)

