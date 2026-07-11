from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from backend.services import repo
from backend.training.samples import (  # noqa: F401 — 하위 호환 재노출
    POSITIVE_EVENT_TYPES,
    InteractionSample,
    build_samples_from_rows,
    extract_impression_article_ids,
    load_samples_from_db,
    parse_event_data,
)


@dataclass(frozen=True)
class ArticleFeatures:
    article_id: str
    title_embedding: list[float]
    summary_embedding: list[float]
    category: str


def make_category_vocab(categories: Iterable[str]) -> dict[str, int]:
    values = sorted({str(category or "unknown") for category in categories})
    if "unknown" not in values:
        values.insert(0, "unknown")
    return {category: idx for idx, category in enumerate(values)}


def category_onehot(category: str, vocab: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    vector[vocab.get(str(category or "unknown"), vocab["unknown"])] = 1.0
    return vector


class RecommendationDataset(Dataset):
    def __init__(
        self,
        samples: list[InteractionSample],
        article_features: dict[str, ArticleFeatures],
        category_vocab: dict[str, int],
        history_size: int = 20,
    ):
        self.samples = samples
        self.article_features = article_features
        self.category_vocab = category_vocab
        self.history_size = history_size

    def __len__(self) -> int:
        return len(self.samples)

    def _feature_tensors(self, article_id: str):
        features = self.article_features[article_id]
        return (
            torch.tensor(features.title_embedding, dtype=torch.float32),
            torch.tensor(features.summary_embedding, dtype=torch.float32),
            category_onehot(features.category, self.category_vocab),
        )

    def __getitem__(self, index: int):
        sample = self.samples[index]
        history_ids = sample.history_article_ids[-self.history_size :]
        candidates = [sample.positive_article_id, *sample.negative_article_ids]

        return {
            "history": [self._feature_tensors(article_id) for article_id in history_ids],
            "history_mask": torch.ones(len(history_ids), dtype=torch.bool),
            "candidates": [self._feature_tensors(article_id) for article_id in candidates],
            "label": torch.tensor(0, dtype=torch.long),
        }


def collate_recommendation_batch(batch: list[dict]):
    max_history = max(len(item["history"]) for item in batch)
    candidate_count = len(batch[0]["candidates"])
    category_dim = batch[0]["candidates"][0][2].shape[0]

    def zeros_feature():
        return (
            torch.zeros(768, dtype=torch.float32),
            torch.zeros(768, dtype=torch.float32),
            torch.zeros(category_dim, dtype=torch.float32),
        )

    history = []
    history_mask = []
    candidates = []
    labels = []
    for item in batch:
        padded_history = list(item["history"]) + [zeros_feature()] * (max_history - len(item["history"]))
        history.append(padded_history)
        history_mask.append(
            torch.cat(
                [
                    item["history_mask"],
                    torch.zeros(max_history - len(item["history"]), dtype=torch.bool),
                ]
            )
        )
        candidates.append(item["candidates"])
        labels.append(item["label"])

    def stack_feature_list(items):
        titles = torch.stack([feature[0] for feature in items])
        summaries = torch.stack([feature[1] for feature in items])
        categories = torch.stack([feature[2] for feature in items])
        return titles, summaries, categories

    history_titles, history_summaries, history_categories = zip(
        *[stack_feature_list(item) for item in history]
    )
    candidate_titles, candidate_summaries, candidate_categories = zip(
        *[stack_feature_list(item) for item in candidates]
    )

    return {
        "history_title": torch.stack(history_titles),
        "history_summary": torch.stack(history_summaries),
        "history_category": torch.stack(history_categories),
        "history_mask": torch.stack(history_mask),
        "candidate_title": torch.stack(candidate_titles),
        "candidate_summary": torch.stack(candidate_summaries),
        "candidate_category": torch.stack(candidate_categories),
        "label": torch.stack(labels),
        "candidate_count": candidate_count,
    }


def load_article_features_from_db() -> tuple[dict[str, ArticleFeatures], dict[str, int]]:
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute(
            """
            SELECT article_id, category, embed_summary
            FROM articles
            WHERE article_id IS NOT NULL
            """
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        con.close()

    features: dict[str, ArticleFeatures] = {}
    categories = []
    for article_id, category, embed_summary in rows:
        summary_embedding = repo._parse_vector(embed_summary) or [0.0] * 768
        features[str(article_id)] = ArticleFeatures(
            article_id=str(article_id),
            title_embedding=summary_embedding,
            summary_embedding=summary_embedding,
            category=str(category or "unknown"),
        )
        categories.append(str(category or "unknown"))
    return features, make_category_vocab(categories)
