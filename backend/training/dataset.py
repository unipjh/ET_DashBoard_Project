from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from backend.services import repo


POSITIVE_EVENT_TYPES = {
    "view_article_detail",
    "view_detail",
    "click_article",
    "click_related",
    "click_feedback",
}


@dataclass(frozen=True)
class ArticleFeatures:
    article_id: str
    title_embedding: list[float]
    summary_embedding: list[float]
    category: str


@dataclass(frozen=True)
class InteractionSample:
    subject_id: str
    history_article_ids: list[str]
    positive_article_id: str
    negative_article_ids: list[str]


def parse_event_data(raw_value) -> dict:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        return json.loads(str(raw_value))
    except Exception:
        return {}


def extract_impression_article_ids(raw_event_data) -> list[str]:
    event_data = parse_event_data(raw_event_data)
    articles = event_data.get("articles") or []
    if not articles and event_data.get("article_ids"):
        articles = event_data.get("article_ids") or []

    article_ids: list[str] = []
    for item in articles:
        article_id = item.get("article_id") if isinstance(item, dict) else item
        if article_id:
            article_ids.append(str(article_id))
    return article_ids


def make_category_vocab(categories: Iterable[str]) -> dict[str, int]:
    values = sorted({str(category or "unknown") for category in categories})
    if "unknown" not in values:
        values.insert(0, "unknown")
    return {category: idx for idx, category in enumerate(values)}


def category_onehot(category: str, vocab: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    vector[vocab.get(str(category or "unknown"), vocab["unknown"])] = 1.0
    return vector


def build_samples_from_rows(
    event_rows: list[dict],
    feedback_rows: list[dict],
    all_article_ids: list[str],
    history_size: int = 20,
    negative_count: int = 4,
) -> list[InteractionSample]:
    positives_by_subject: dict[str, list[tuple[str, str]]] = {}
    impressions_by_subject: dict[str, list[str]] = {}
    clicked_by_subject: dict[str, set[str]] = {}

    for row in event_rows:
        subject_id = str(row.get("user_id") or row.get("session_id") or "guest")
        event_type = row.get("event_type")
        article_id = row.get("article_id")
        created_at = str(row.get("created_at") or "")
        if event_type == "impression":
            impressions_by_subject.setdefault(subject_id, []).extend(
                extract_impression_article_ids(row.get("event_data"))
            )
        elif event_type in POSITIVE_EVENT_TYPES and article_id:
            positives_by_subject.setdefault(subject_id, []).append((created_at, str(article_id)))
            clicked_by_subject.setdefault(subject_id, set()).add(str(article_id))

    for row in feedback_rows:
        if row.get("feedback_type") != "like":
            continue
        subject_id = str(row.get("user_id") or "guest")
        article_id = row.get("article_id")
        created_at = str(row.get("created_at") or "")
        if article_id:
            positives_by_subject.setdefault(subject_id, []).append((created_at, str(article_id)))
            clicked_by_subject.setdefault(subject_id, set()).add(str(article_id))

    samples: list[InteractionSample] = []
    article_pool = [str(article_id) for article_id in all_article_ids]
    for subject_id, positives in positives_by_subject.items():
        positives = sorted(positives, key=lambda item: item[0])
        history: list[str] = []
        clicked = clicked_by_subject.get(subject_id, set())
        impression_negatives = [
            article_id
            for article_id in impressions_by_subject.get(subject_id, [])
            if article_id not in clicked
        ]

        for _, positive_article_id in positives:
            if history:
                negatives = list(dict.fromkeys(impression_negatives))
                if len(negatives) < negative_count:
                    fallback = [
                        article_id
                        for article_id in article_pool
                        if article_id != positive_article_id and article_id not in history
                    ]
                    random.shuffle(fallback)
                    negatives.extend(fallback[: negative_count - len(negatives)])
                samples.append(
                    InteractionSample(
                        subject_id=subject_id,
                        history_article_ids=history[-history_size:],
                        positive_article_id=positive_article_id,
                        negative_article_ids=negatives[:negative_count],
                    )
                )
            history.append(positive_article_id)
    return samples


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


def load_samples_from_db(history_size: int = 20, negative_count: int = 4) -> list[InteractionSample]:
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT article_id FROM articles")
        all_article_ids = [row[0] for row in cur.fetchall()]
        cur.execute(
            """
            SELECT session_id, user_id, event_type, article_id, event_data, created_at
            FROM event_logs
            ORDER BY created_at ASC
            """
        )
        event_rows = [
            {
                "session_id": row[0],
                "user_id": row[1],
                "event_type": row[2],
                "article_id": row[3],
                "event_data": row[4],
                "created_at": row[5],
            }
            for row in cur.fetchall()
        ]
        cur.execute(
            """
            SELECT user_id, article_id, feedback_type, created_at
            FROM feedback_logs
            ORDER BY created_at ASC
            """
        )
        feedback_rows = [
            {
                "user_id": row[0],
                "article_id": row[1],
                "feedback_type": row[2],
                "created_at": row[3],
            }
            for row in cur.fetchall()
        ]
    finally:
        cur.close()
        con.close()

    return build_samples_from_rows(
        event_rows=event_rows,
        feedback_rows=feedback_rows,
        all_article_ids=all_article_ids,
        history_size=history_size,
        negative_count=negative_count,
    )
