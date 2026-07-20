"""Human-intuition persona benchmark for lightweight recommendation tuning.

The synthetic click rule is deliberately explicit and deterministic. It is not
intended to imitate production traffic; it encodes product hypotheses that can
be reviewed by humans and used as a stable regression benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.services import repo
from backend.training.samples import InteractionSample


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    category_preferences: dict[str, float]
    category_weight: float
    semantic_weight: float
    trust_weight: float
    freshness_weight: float
    novelty_weight: float
    rationale: str


PERSONAS = [
    PersonaSpec("IT전문독자", {"IT/과학": 1.0, "경제": 0.35, "세계": 0.20}, .45, .35, .10, .10, .00,
                "전문 분야의 연속성을 가장 중시하고 관련 산업 기사도 읽는다."),
    PersonaSpec("경제실용독자", {"경제": 1.0, "정치": 0.35, "IT/과학": 0.30}, .40, .30, .20, .10, .00,
                "경제 주제와 기존 관심의 연속성을 선호하되 근거가 약한 기사는 피한다."),
    PersonaSpec("시민정책독자", {"정치": 1.0, "사회": 0.70, "세계": 0.25}, .35, .20, .35, .10, .00,
                "정책·사회 이슈를 보지만 출처와 근거 신뢰도를 강하게 고려한다."),
    PersonaSpec("국제정세독자", {"세계": 1.0, "정치": 0.50, "경제": 0.45}, .40, .25, .20, .15, .00,
                "국제 이슈의 맥락과 최신성을 함께 본다."),
    PersonaSpec("생활정보독자", {"생활/문화": 1.0, "사회": 0.35, "IT/과학": 0.20}, .40, .20, .10, .15, .15,
                "생활 주제를 중심으로 읽지만 가끔 새로운 실용 주제를 탐색한다."),
    PersonaSpec("호기심탐색독자", {}, .15, .15, .15, .20, .35,
                "최근에 읽지 않은 분야와 새로운 기사를 의도적으로 탐색한다."),
    PersonaSpec("신뢰우선독자", {}, .15, .15, .55, .10, .05,
                "분야보다 신뢰도와 근거 품질을 최우선으로 본다."),
    PersonaSpec("속보스캐너", {"정치": .35, "경제": .35, "세계": .35, "사회": .35}, .20, .10, .15, .50, .05,
                "여러 분야를 빠르게 훑으며 최신 기사를 우선한다."),
]

# 다중 seed에서 반복 검증할 배포 후보. 임베딩은 후보 검색과 10% 재정렬 신호로 유지한다.
DEPLOYMENT_CANDIDATE_WEIGHTS = (0.1, 0.6, 0.1, 0.2)


@dataclass(frozen=True)
class ArticleMeta:
    article_id: str
    category: str
    embedding: np.ndarray
    trust: float
    freshness: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def load_catalog() -> dict[str, ArticleMeta]:
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT article_id, category, published_at, trust_score, embed_summary FROM articles")
        rows = cur.fetchall()
    finally:
        cur.close()
        con.close()

    valid = []
    for article_id, category, published_at, trust_score, embedding in rows:
        parsed = repo._parse_vector(embedding)
        if parsed:
            valid.append((str(article_id), str(category or "unknown"), str(published_at or ""), trust_score, parsed))
    date_order = {value: idx for idx, value in enumerate(sorted({row[2] for row in valid}))}
    max_date_rank = max(date_order.values(), default=1) or 1
    return {
        article_id: ArticleMeta(
            article_id=article_id,
            category=category,
            embedding=_normalize(np.asarray(embedding, dtype=np.float32)),
            trust=(float(trust_score) / 100.0) if trust_score not in (None, 0) else 0.5,
            freshness=date_order[published_at] / max_date_rank,
        )
        for article_id, category, published_at, trust_score, embedding in valid
    }


def _category_preference(persona: PersonaSpec, category: str) -> float:
    if not persona.category_preferences:
        return 0.5
    return persona.category_preferences.get(category, 0.05)


def _history_semantic(article: ArticleMeta, history: list[str], catalog: dict[str, ArticleMeta]) -> float:
    if not history:
        return 0.5
    recent = [_normalize(catalog[article_id].embedding) for article_id in history[-5:]]
    profile = _normalize(np.mean(np.stack(recent), axis=0))
    return (float(profile @ article.embedding) + 1.0) / 2.0


def _utility(
    article: ArticleMeta,
    persona: PersonaSpec,
    history: list[str],
    intent_category: str | None,
    catalog: dict[str, ArticleMeta],
    rng: random.Random,
    noise: float,
) -> float:
    category = _category_preference(persona, article.category)
    if intent_category:
        category = 0.65 * category + 0.35 * float(article.category == intent_category)
    semantic = _history_semantic(article, history, catalog)
    recent_categories = {catalog[article_id].category for article_id in history[-5:]}
    novelty = float(bool(history) and article.category not in recent_categories)
    return (
        persona.category_weight * category
        + persona.semantic_weight * semantic
        + persona.trust_weight * article.trust
        + persona.freshness_weight * article.freshness
        + persona.novelty_weight * novelty
        + rng.gauss(0.0, noise)
    )


def generate_persona_samples(
    catalog: dict[str, ArticleMeta],
    users_per_persona: int = 20,
    sessions_per_user: int = 12,
    impression_size: int = 12,
    negative_count: int = 4,
    history_size: int = 20,
    seed: int = 42,
    noise: float = 0.04,
) -> list[InteractionSample]:
    if len(catalog) < impression_size + 2:
        raise RuntimeError(f"유효 임베딩 기사 {len(catalog)}개로 impression_size={impression_size}를 만들 수 없습니다.")
    article_ids = sorted(catalog)
    all_samples: list[InteractionSample] = []
    for persona_index, persona in enumerate(PERSONAS):
        intent_categories = list(persona.category_preferences) or sorted({item.category for item in catalog.values()})
        for user_index in range(users_per_persona):
            rng = random.Random(seed + persona_index * 10_000 + user_index)
            history: list[str] = []
            intent_category = rng.choice(intent_categories)
            for session_index in range(sessions_per_user):
                if session_index % 4 == 0:
                    intent_category = rng.choice(intent_categories)
                recent_ids = set(history[-5:])
                available = [article_id for article_id in article_ids if article_id not in recent_ids]
                pool = rng.sample(available, min(impression_size, len(available)))
                scored = sorted(
                    pool,
                    key=lambda article_id: (
                        _utility(catalog[article_id], persona, history, intent_category, catalog, rng, noise),
                        article_id,
                    ),
                    reverse=True,
                )
                positive = scored[0]
                if history:
                    all_samples.append(
                        InteractionSample(
                            subject_id=f"persona:{persona.name}|user:{user_index:02d}",
                            history_article_ids=list(history[-history_size:]),
                            positive_article_id=positive,
                            negative_article_ids=scored[1:1 + negative_count],
                            event_at=f"synthetic:{session_index:02d}",
                        )
                    )
                history.append(positive)
    return all_samples


def _minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    spread = float(array.max() - array.min())
    return np.zeros_like(array) if spread <= 1e-12 else (array - array.min()) / spread


def sample_components(sample: InteractionSample, catalog: dict[str, ArticleMeta]) -> tuple[list[str], np.ndarray]:
    candidates = [sample.positive_article_id, *sample.negative_article_ids]
    history = sample.history_article_ids[-20:]
    ages = np.arange(len(history) - 1, -1, -1, dtype=np.float32)
    recency = np.power(0.5, ages / 5.0)
    profile = _normalize(np.average(np.stack([catalog[item].embedding for item in history]), axis=0, weights=recency))
    semantic = _minmax([float(profile @ catalog[item].embedding) for item in candidates])
    category_weights: Counter = Counter()
    for index, item in enumerate(history):
        age = len(history) - 1 - index
        category_weights[catalog[item].category] += 0.5 ** (age / 5.0)
    category = _minmax([category_weights[catalog[item].category] for item in candidates])
    trust = _minmax([catalog[item].trust for item in candidates])
    freshness = _minmax([catalog[item].freshness for item in candidates])
    return candidates, np.stack([semantic, category, trust, freshness], axis=1)


def _metrics(samples: list[InteractionSample], components: dict[tuple[str, str, str], tuple[list[str], np.ndarray]], weights: tuple[float, ...], k: int = 3) -> dict:
    ranks = []
    persona_ranks: dict[str, list[int]] = defaultdict(list)
    low_trust_topk = 0
    topk_total = 0
    for sample in samples:
        candidates, matrix = components[(sample.subject_id, sample.event_at, sample.positive_article_id)]
        scores = matrix @ np.asarray(weights)
        ordered = sorted(range(len(candidates)), key=lambda idx: (float(scores[idx]), candidates[idx]), reverse=True)
        rank = ordered.index(0) + 1
        ranks.append(rank)
        persona = sample.subject_id.split("|", 1)[0].removeprefix("persona:")
        persona_ranks[persona].append(rank)
        for idx in ordered[:k]:
            low_trust_topk += int(catalog_global[candidates[idx]].trust < 0.4)
            topk_total += 1
    def ndcg(values):
        return sum(1 / math.log2(rank + 1) if rank <= k else 0 for rank in values) / len(values)
    return {
        "n": len(ranks),
        f"hit_rate_at_{k}": sum(rank <= k for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        f"ndcg_at_{k}": ndcg(ranks),
        "macro_persona_ndcg": sum(ndcg(values) for values in persona_ranks.values()) / len(persona_ranks),
        "worst_persona_ndcg": min(ndcg(values) for values in persona_ranks.values()),
        "low_trust_topk_rate": low_trust_topk / topk_total if topk_total else 0.0,
    }


catalog_global: dict[str, ArticleMeta] = {}


def run_benchmark(seed: int = 42) -> dict:
    global catalog_global
    catalog_global = load_catalog()
    samples = generate_persona_samples(catalog_global, seed=seed)
    components = {
        (sample.subject_id, sample.event_at, sample.positive_article_id): sample_components(sample, catalog_global)
        for sample in samples
    }
    train = [sample for sample in samples if int(sample.subject_id.rsplit(":", 1)[1]) < 14]
    validation = [sample for sample in samples if int(sample.subject_id.rsplit(":", 1)[1]) >= 14]

    candidates = []
    for semantic in range(0, 11):
        for category in range(0, 11 - semantic):
            for trust in range(0, 11 - semantic - category):
                freshness = 10 - semantic - category - trust
                weights = tuple(value / 10 for value in (semantic, category, trust, freshness))
                metrics = _metrics(train, components, weights)
                candidates.append((metrics["macro_persona_ndcg"], metrics["mrr"], weights, metrics))
    _, _, selected_weights, train_metrics = max(candidates)
    validation_metrics = _metrics(validation, components, selected_weights)
    baselines = {
        "semantic_only": _metrics(validation, components, (1.0, 0.0, 0.0, 0.0)),
        "category_only": _metrics(validation, components, (0.0, 1.0, 0.0, 0.0)),
        "deployment_candidate": _metrics(validation, components, DEPLOYMENT_CANDIDATE_WEIGHTS),
        "selected_hybrid": validation_metrics,
    }
    return {
        "seed": seed,
        "catalog_size": len(catalog_global),
        "sample_count": len(samples),
        "train_count": len(train),
        "validation_count": len(validation),
        "feature_order": ["semantic", "category", "trust", "freshness"],
        "selected_weights": selected_weights,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "baselines": baselines,
        "personas": [asdict(persona) for persona in PERSONAS],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="FINAL_ANALYSIS/artifacts/optimization/persona_benchmark.json")
    args = parser.parse_args()
    result = run_benchmark(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "catalog_size", "sample_count", "train_count", "validation_count",
        "selected_weights", "validation_metrics", "baselines",
    )}, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
