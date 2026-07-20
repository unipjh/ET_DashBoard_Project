from __future__ import annotations

import random

from backend.training.dataset import ArticleFeatures, InteractionSample
from backend.services import repo


# 8개 페르소나: 카테고리 선호도 + 신뢰도 민감도
PERSONAS: list[dict] = [
    {
        "name": "IT마니아",
        "weights": {"IT/과학": 0.70, "경제": 0.15, "세계": 0.10},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "경제독자",
        "weights": {"경제": 0.70, "정치": 0.15, "IT/과학": 0.10},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "정치독자",
        "weights": {"정치": 0.70, "사회": 0.15, "세계": 0.10},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "사회독자",
        "weights": {"사회": 0.60, "생활/문화": 0.20, "정치": 0.15},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "국제독자",
        "weights": {"세계": 0.70, "정치": 0.15, "경제": 0.10},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "라이프독자",
        "weights": {"생활/문화": 0.60, "연예": 0.25, "사회": 0.10},
        "default": 0.03,
        "trust_weight": 0.0,
    },
    {
        "name": "종합독자",
        "weights": {},
        "default": 0.125,
        "trust_weight": 0.0,
    },
    {
        "name": "신뢰독자",
        "weights": {},
        "default": 0.10,
        "trust_weight": 0.35,  # trust_score(0~1)에 0.35를 곱해 가산
    },
]


def _load_trust_scores() -> dict[str, float]:
    """articles 테이블에서 trust_score 로드. 실패 시 빈 dict 반환."""
    try:
        con = repo._get_con()
        cur = con.cursor()
        try:
            cur.execute("SELECT article_id, trust_score FROM articles WHERE trust_score IS NOT NULL")
            rows = cur.fetchall()
        finally:
            cur.close()
            con.close()
        result: dict[str, float] = {}
        for article_id, score in rows:
            raw = float(score or 0)
            # DB 값이 0~100 범위면 0~1로 정규화
            result[str(article_id)] = raw / 100.0 if raw > 1.0 else raw
        return result
    except Exception:
        return {}


def _score(
    article_id: str,
    features: ArticleFeatures,
    persona: dict,
    trust_scores: dict[str, float],
    noise: float,
    rng: random.Random,
) -> float:
    category_score = persona["weights"].get(features.category, persona["default"])
    trust_bonus = persona["trust_weight"] * trust_scores.get(article_id, 0.5)
    return category_score + trust_bonus + rng.gauss(0.0, noise)


def _simulate_persona(
    persona: dict,
    article_ids: list[str],
    article_features: dict[str, ArticleFeatures],
    trust_scores: dict[str, float],
    sessions: int,
    history_size: int,
    negative_count: int,
    impression_size: int,
    noise: float,
    rng: random.Random,
) -> list[InteractionSample]:
    samples: list[InteractionSample] = []
    history: list[str] = []

    for session_idx in range(sessions):
        # 이미 최근 히스토리에 있는 기사는 제외 (현실적인 노출 풀 구성)
        recent = set(history[-history_size:])
        available = [aid for aid in article_ids if aid not in recent]
        if len(available) < impression_size + negative_count:
            available = article_ids  # 기사 수 부족 시 전체 풀 사용

        pool = rng.sample(available, min(impression_size, len(available)))
        scored = sorted(
            pool,
            key=lambda aid: _score(aid, article_features[aid], persona, trust_scores, noise, rng),
            reverse=True,
        )

        positive_id = scored[0]
        not_clicked = scored[1:]

        if history:
            negatives = list(not_clicked[:negative_count])
            # negative가 부족하면 전체 풀에서 보충
            if len(negatives) < negative_count:
                clicked_set = set(history) | {positive_id}
                extra = [aid for aid in article_ids if aid not in clicked_set and aid not in negatives]
                rng.shuffle(extra)
                negatives.extend(extra[: negative_count - len(negatives)])

            samples.append(
                InteractionSample(
                    subject_id=f"synthetic_{persona['name']}_{session_idx}",
                    history_article_ids=list(history[-history_size:]),
                    positive_article_id=positive_id,
                    negative_article_ids=negatives[:negative_count],
                )
            )

        history.append(positive_id)

    return samples


def generate_synthetic_samples(
    article_features: dict[str, ArticleFeatures],
    sessions_per_persona: int = 80,
    history_size: int = 20,
    negative_count: int = 4,
    impression_size: int = 15,
    noise: float = 0.15,
    seed: int = 42,
) -> list[InteractionSample]:
    """
    페르소나 기반 합성 인터랙션 샘플 생성.

    각 페르소나는 카테고리 선호도에 따라 노출된 기사 중 가장 적합한 기사를 클릭한다.
    생성된 샘플은 train_user_encoder.py의 --synthetic 플래그로 사용된다.
    """
    min_required = impression_size + negative_count + 2
    if len(article_features) < min_required:
        raise RuntimeError(
            f"기사 수 부족: {len(article_features)}개. "
            f"합성 데이터 생성에 최소 {min_required}개 필요합니다."
        )

    rng = random.Random(seed)
    trust_scores = _load_trust_scores()
    article_ids = list(article_features.keys())

    all_samples: list[InteractionSample] = []
    for persona in PERSONAS:
        persona_samples = _simulate_persona(
            persona=persona,
            article_ids=article_ids,
            article_features=article_features,
            trust_scores=trust_scores,
            sessions=sessions_per_persona,
            history_size=history_size,
            negative_count=negative_count,
            impression_size=impression_size,
            noise=noise,
            rng=rng,
        )
        all_samples.extend(persona_samples)
        print(f"  [{persona['name']}] {len(persona_samples)}개 샘플 생성")

    rng.shuffle(all_samples)
    print(f"  총 {len(all_samples)}개 합성 샘플 (8개 페르소나 × ~{sessions_per_persona}세션)")
    return all_samples
