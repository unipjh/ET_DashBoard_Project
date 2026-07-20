from __future__ import annotations

import math

from backend.services import encoder_inference, repo


MIN_HISTORY_FOR_PERSONALIZATION = 1
EMBED_DIM = 768

# 단기(세션) 의도를 우선하되 장기(계정) 취향을 보조로 섞는 비율
SHORT_TERM_WEIGHT = 0.65
# 히스토리 recency 반감기: 최근 5개 이전의 기사는 가중치가 절반으로 줄어든다
RECENCY_HALF_LIFE = 5.0
# 프로필-기사 코사인 유사도 하한. 미달 시 "뜬금없는" 추천 대신 카테고리 폴백
RELEVANCE_FLOOR = 0.35
# 같은 카테고리가 추천 목록을 도배하지 않도록 하는 비율 상한
MAX_CATEGORY_SHARE = 0.6
# limit이 이 값 이상일 때 마지막 슬롯 1개를 탐색(explore) 추천에 배정
EXPLORE_SLOT_MIN_LIMIT = 5
# trust_verdict 'likely_false' 경계(40점)와 동일 — 이보다 낮으면 목록 후순위로 강등
LOW_TRUST_THRESHOLD = 40
# profile 후보 재정렬: semantic/category/trust/freshness.
# 사람 직관 페르소나 8종 × 5 seeds와 시간 순서 실로그 리플레이로 선택한 보수적 조합.
PROFILE_RERANK_WEIGHTS = (0.10, 0.60, 0.10, 0.20)


def _chronological(history_articles: list[dict]) -> list[dict]:
    """repo.get_recent_user_history는 최신순(DESC)이므로 시간순(과거→최근)으로 변환한다."""
    return list(reversed(history_articles))


def build_profile_vector(history_articles: list[dict]) -> list[float]:
    """raw Gemini 공간의 시간 가중 프로필.

    이 벡터는 ``article_chunks.embedding`` 검색에 사용되므로 변환된
    ``learned_embedding``을 섞지 않는다. Attention 경로는 encode_user와
    learned_embedding 검색을 별도로 사용한다.
    """
    recent = history_articles[-20:]
    vectors: list[list[float]] = []
    weights: list[float] = []
    for index, article in enumerate(recent):
        vector = repo._parse_vector(article.get("embed_summary"))
        if vector:
            age = len(recent) - 1 - index  # 0 = 가장 최근 기사
            vectors.append(vector)
            weights.append(0.5 ** (age / RECENCY_HALF_LIFE))
    if not vectors:
        return [0.0] * EMBED_DIM

    total_weight = sum(weights)
    return [
        sum(vector[dim] * weight for vector, weight in zip(vectors, weights)) / total_weight
        for dim in range(EMBED_DIM)
    ]


def _blend_vectors(primary: list[float], secondary: list[float], primary_weight: float) -> list[float]:
    if not any(secondary):
        return primary
    if not any(primary):
        return secondary
    return [
        p * primary_weight + s * (1.0 - primary_weight)
        for p, s in zip(primary, secondary)
    ]


def _merge_histories(session_history: list[dict], user_history: list[dict]) -> list[dict]:
    """세션/계정 히스토리를 최신순으로 병합하고 기사 단위로 중복 제거한다."""
    merged: list[dict] = []
    seen: set[str] = set()
    for article in [*session_history, *user_history]:
        article_id = article.get("article_id")
        if article_id and article_id not in seen:
            seen.add(article_id)
            merged.append(article)
    return merged


def _tag(records: list[dict], source: str) -> list[dict]:
    for record in records:
        record["rec_source"] = source
        if record.get("score") is not None and record.get("rec_score") is None:
            try:
                record["rec_score"] = round(float(record["score"]), 4)
            except (TypeError, ValueError):
                pass
    return records


def _fallback_latest(history_articles: list[dict], limit: int) -> list[dict]:
    exclude_ids = [article.get("article_id") for article in history_articles if article.get("article_id")]
    return repo.get_latest_articles(limit=limit, exclude_article_ids=exclude_ids)


def _fallback_by_history_categories(history_articles: list[dict], limit: int) -> list[dict]:
    categories = []
    for article in history_articles:
        category = article.get("category")
        if category and category not in categories:
            categories.append(category)

    if not categories:
        return []

    exclude_ids = [article.get("article_id") for article in history_articles if article.get("article_id")]
    return repo.get_latest_articles_by_categories(
        categories=categories,
        limit=limit,
        exclude_article_ids=exclude_ids,
    )


def _as_article_response(record: dict) -> dict:
    return {
        "article_id": record.get("article_id", ""),
        "title": record.get("title", ""),
        "source": record.get("source", ""),
        "url": record.get("url", ""),
        "published_at": record.get("published_at", ""),
        "summary_text": record.get("summary_text", ""),
        "keywords": record.get("keywords", "[]"),
        "trust_score": int(record.get("trust_score") or 0),
        "trust_verdict": record.get("trust_verdict", ""),
        "category": record.get("category", ""),
        "score": record.get("score"),
    }


def _fill_to_limit(
    results: list[dict],
    limit: int,
    exclude_article_ids: list[str] | None = None,
) -> list[dict]:
    """최적화 후보가 부족하면 최신 기사로 채워 응답 개수를 보장한다."""
    if len(results) >= limit:
        return results[:limit]
    excluded = set(exclude_article_ids or [])
    existing_ids = {record.get("article_id") for record in results}
    filler = repo.get_latest_articles(
        limit=limit * 2,
        exclude_article_ids=list(existing_ids | excluded),
    )
    for article in filler:
        if len(results) >= limit:
            break
        article_id = article.get("article_id")
        if article_id and article_id not in existing_ids and article_id not in excluded:
            record = _as_article_response(article) if "trust_score" not in article else article
            record.setdefault("rec_source", "latest")
            results.append(record)
            existing_ids.add(article_id)
    return results[:limit]


def _diversify_by_category(records: list[dict], limit: int) -> list[dict]:
    """유사도 순서를 유지하되 한 카테고리가 limit의 일정 비율을 넘지 않게 제한한다."""
    if limit <= 2:
        return records
    max_per_category = max(2, math.ceil(limit * MAX_CATEGORY_SHARE))
    picked: list[dict] = []
    overflow: list[dict] = []
    counts: dict[str, int] = {}
    for record in records:
        category = record.get("category") or "unknown"
        if counts.get(category, 0) < max_per_category:
            counts[category] = counts.get(category, 0) + 1
            picked.append(record)
        else:
            overflow.append(record)
    return picked + overflow


def _apply_trust_guardrail(records: list[dict]) -> list[dict]:
    """분석 결과 신뢰도가 낮은(likely_false 구간) 기사는 목록 뒤로 강등한다."""
    def is_low_trust(record: dict) -> bool:
        score = int(record.get("trust_score") or 0)
        return 0 < score < LOW_TRUST_THRESHOLD

    trusted = [record for record in records if not is_low_trust(record)]
    demoted = [record for record in records if is_low_trust(record)]
    return trusted + demoted


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _rerank_profile_candidates(records: list[dict], history: list[dict]) -> list[dict]:
    """임베딩 검색 후보를 관심 카테고리·신뢰도·최신성으로 가볍게 재정렬한다."""
    if len(records) < 2:
        return records

    category_weights: dict[str, float] = {}
    for age, article in enumerate(history[:20]):  # history는 최신순
        category = article.get("category") or "unknown"
        category_weights[category] = category_weights.get(category, 0.0) + 0.5 ** (age / RECENCY_HALF_LIFE)

    semantic = _minmax([float(record.get("score") or 0.0) for record in records])
    category = _minmax([
        category_weights.get(record.get("category") or "unknown", 0.0)
        for record in records
    ])
    trust = _minmax([float(record.get("trust_score") or 0.0) / 100.0 for record in records])
    date_values = [str(record.get("published_at") or "") for record in records]
    date_rank = {value: index for index, value in enumerate(sorted(set(date_values)))}
    freshness = _minmax([float(date_rank[value]) for value in date_values])

    for index, record in enumerate(records):
        components = (semantic[index], category[index], trust[index], freshness[index])
        record["_rank_score"] = sum(
            weight * component
            for weight, component in zip(PROFILE_RERANK_WEIGHTS, components)
        )
    return sorted(
        records,
        key=lambda record: (record.get("_rank_score", 0.0), record.get("article_id", "")),
        reverse=True,
    )


def _add_explore_slot(records: list[dict], history: list[dict], limit: int) -> list[dict]:
    """마지막 슬롯 1개를 프로필 밖 카테고리의 최신 기사로 교체해 필터버블을 완화한다."""
    if limit < EXPLORE_SLOT_MIN_LIMIT or len(records) < limit:
        return records

    history_categories = {
        article.get("category") for article in history if article.get("category")
    }
    known_ids = {record.get("article_id") for record in records} | {
        article.get("article_id") for article in history
    }
    try:
        candidates = repo.get_latest_articles(limit=10, exclude_article_ids=list(known_ids))
    except Exception as e:
        print(f"[RECOMMEND WARN] explore slot fetch failed: {type(e).__name__}: {e}")
        return records

    for candidate in candidates:
        if (candidate.get("category") or "unknown") not in history_categories:
            candidate["rec_source"] = "explore"
            return records[: limit - 1] + [candidate]
    return records


def get_recommendations(
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 10,
) -> list[dict]:
    session_history = (
        repo.get_recent_user_history(session_id=session_id, limit=20) if session_id else []
    )
    user_history = (
        repo.get_recent_user_history(user_id=user_id, limit=20) if user_id else []
    )
    history = _merge_histories(session_history, user_history)

    if len(history) < MIN_HISTORY_FOR_PERSONALIZATION:
        return _tag(_fallback_latest(history, limit), "latest")

    # 원격 main의 서비스 정책 유지: 직전 5개는 반드시 제외하고 오래된 이력은 재노출 허용.
    exclude_ids = [article.get("article_id") for article in history[:5] if article.get("article_id")]
    results: list[dict] = []

    # Stage 1 — 어텐션 인코더 (torch 가용 환경에서만)
    if encoder_inference.is_model_ready():
        user_vector = encoder_inference.encode_user(_chronological(history))
        recommendations = repo.search_articles_by_learned_embedding(
            user_vector,
            limit=limit * 2,
            exclude_article_ids=exclude_ids,
        )
        if recommendations:
            results = _tag(recommendations, "encoder")

    # Stage 2 — 임베딩 프로필(장/단기 블렌딩) 유사도
    if not results:
        session_profile = build_profile_vector(_chronological(session_history))
        user_profile = build_profile_vector(_chronological(user_history))
        profile_vector = _blend_vectors(session_profile, user_profile, SHORT_TERM_WEIGHT)
        if any(profile_vector):
            try:
                df = repo.search_similar_chunks(
                    profile_vector,
                    limit=limit * 3,
                    min_score=RELEVANCE_FLOOR,
                    dedupe_per_article=True,
                )
                if not df.empty:
                    records = [
                        _as_article_response(record)
                        for record in df.to_dict(orient="records")
                        if record.get("article_id") not in set(exclude_ids)
                    ]
                    if records:
                        results = _tag(_rerank_profile_candidates(records, history), "profile")
            except Exception as e:
                print(f"[RECOMMEND WARN] fallback vector search failed: {type(e).__name__}: {e}")

    if results:
        results = _diversify_by_category(results, limit)
        results = _apply_trust_guardrail(results)[:limit]
        results = _fill_to_limit(results, limit, exclude_ids)
        results = _add_explore_slot(results, history, limit)
        return results[:limit]

    # Stage 3 — 카테고리 폴백
    category_recommendations = _fallback_by_history_categories(history, limit)
    if category_recommendations:
        tagged = _tag(_apply_trust_guardrail(category_recommendations), "category")
        return _fill_to_limit(tagged, limit, exclude_ids)

    # Stage 4 — 최신 기사 폴백
    return _tag(_fallback_latest(history, limit), "latest")
