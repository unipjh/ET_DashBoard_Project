from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

from backend.services import repo

try:
    import torch
    from backend.training.news_encoder import NewsEncoder
    from backend.training.user_encoder import UserEncoder
except Exception:
    torch = None
    NewsEncoder = None
    UserEncoder = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model_weights" / "attention_encoders.pt"


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0] * 768
    return [value / norm for value in vector]


@lru_cache(maxsize=1)
def _load_models():
    if torch is None or NewsEncoder is None or UserEncoder is None:
        return None
    if not DEFAULT_MODEL_PATH.exists():
        return None

    checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location="cpu")
    config = checkpoint.get("config", {})
    category_vocab = checkpoint.get("category_vocab", {"unknown": 0})
    news_encoder = NewsEncoder(
        embedding_dim=config.get("embedding_dim", 768),
        category_dim=config.get("category_dim", len(category_vocab)),
    )
    user_encoder = UserEncoder(
        embedding_dim=config.get("embedding_dim", 768),
        max_history=config.get("history_size", 20),
    )
    news_encoder.load_state_dict(checkpoint["news_encoder"])
    user_encoder.load_state_dict(checkpoint["user_encoder"])
    news_encoder.eval()
    user_encoder.eval()
    return {
        "news_encoder": news_encoder,
        "user_encoder": user_encoder,
        "category_vocab": category_vocab,
    }


def is_model_ready() -> bool:
    return _load_models() is not None


def _category_onehot(category: str, vocab: dict[str, int]):
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    vector[vocab.get(str(category or "unknown"), vocab.get("unknown", 0))] = 1.0
    return vector


def _article_base_embedding(article: dict) -> list[float]:
    return (
        repo._parse_vector(article.get("learned_embedding"))
        or repo._parse_vector(article.get("embed_summary"))
        or [0.0] * 768
    )


def encode_news(article: dict) -> list[float]:
    models = _load_models()
    if models is None or torch is None:
        return _normalize(_article_base_embedding(article))

    summary_embedding = repo._parse_vector(article.get("embed_summary")) or [0.0] * 768
    title_embedding = repo._parse_vector(article.get("title_embedding")) or summary_embedding
    category = _category_onehot(article.get("category") or "unknown", models["category_vocab"])

    with torch.no_grad():
        vector = models["news_encoder"](
            torch.tensor(title_embedding),
            torch.tensor(summary_embedding),
            category,
        )
    return vector.squeeze(0).tolist()


def encode_user(history_sequence: list[dict]) -> list[float]:
    if not history_sequence:
        return [0.0] * 768

    models = _load_models()
    news_vectors = [encode_news(article) for article in history_sequence[-20:]]
    if models is None or torch is None:
        avg = [
            sum(vector[idx] for vector in news_vectors) / len(news_vectors)
            for idx in range(768)
        ]
        return _normalize(avg)

    with torch.no_grad():
        history_tensor = torch.tensor(news_vectors, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, history_tensor.shape[1], dtype=torch.bool)
        vector = models["user_encoder"](history_tensor, mask)
    return vector.squeeze(0).tolist()

