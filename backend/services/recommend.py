from __future__ import annotations

from backend.services import encoder_inference, repo


MIN_HISTORY_FOR_PERSONALIZATION = 1


def build_profile_vector(history_articles: list[dict]) -> list[float]:
    vectors = []
    weights = []
    for index, article in enumerate(history_articles[-20:]):
        vector = (
            repo._parse_vector(article.get("learned_embedding"))
            or repo._parse_vector(article.get("embed_summary"))
        )
        if vector:
            vectors.append(vector)
            weights.append(1.0 + index / max(len(history_articles), 1))
    if not vectors:
        return [0.0] * 768

    total_weight = sum(weights)
    return [
        sum(vector[dim] * weight for vector, weight in zip(vectors, weights)) / total_weight
        for dim in range(768)
    ]


def _fallback_latest(history_articles: list[dict], limit: int) -> list[dict]:
    exclude_ids = [article.get("article_id") for article in history_articles[:5] if article.get("article_id")]
    return repo.get_latest_articles(limit=limit, exclude_article_ids=exclude_ids)


def _fallback_by_history_categories(history_articles: list[dict], limit: int) -> list[dict]:
    categories = []
    for article in history_articles:
        category = article.get("category")
        if category and category not in categories:
            categories.append(category)

    if not categories:
        return []

    exclude_ids = [article.get("article_id") for article in history_articles[:5] if article.get("article_id")]
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
    }


def _fill_to_limit(results: list[dict], limit: int) -> list[dict]:
    """결과가 limit보다 적으면 최신 기사로 채워 항상 limit개 반환"""
    if len(results) >= limit:
        return results[:limit]
    existing_ids = {r.get("article_id") for r in results}
    filler = repo.get_latest_articles(limit=limit * 2, exclude_article_ids=list(existing_ids))
    for article in filler:
        if len(results) >= limit:
            break
        aid = article.get("article_id")
        if aid and aid not in existing_ids:
            results.append(_as_article_response(article) if "trust_score" not in article else article)
            existing_ids.add(aid)
    return results[:limit]


def get_recommendations(
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 10,
) -> list[dict]:
    history = repo.get_recent_user_history(user_id=user_id, session_id=session_id, limit=20)
    if len(history) < MIN_HISTORY_FOR_PERSONALIZATION:
        return _fill_to_limit(_fallback_latest(history, limit), limit)

    # 직전에 읽은 5개만 제외 — AI 추천 로직은 그대로, 항상 limit개 보장
    exclude_ids = [article.get("article_id") for article in history[:5] if article.get("article_id")]

    if encoder_inference.is_model_ready():
        user_vector = encoder_inference.encode_user(history)
        recommendations = repo.search_articles_by_learned_embedding(
            user_vector,
            limit=limit,
            exclude_article_ids=exclude_ids,
        )
        if recommendations:
            return _fill_to_limit(recommendations, limit)

    profile_vector = build_profile_vector(history)
    try:
        df = repo.search_similar_chunks(
            profile_vector,
            limit=limit * 3,
            min_score=0.0,
            dedupe_per_article=True,
        )
        if not df.empty:
            records = [
                _as_article_response(record) for record in df.to_dict(orient="records")
                if record.get("article_id") not in set(exclude_ids)
            ]
            if records:
                return _fill_to_limit(records, limit)
    except Exception as e:
        print(f"[RECOMMEND WARN] fallback vector search failed: {type(e).__name__}: {e}")

    category_recommendations = _fallback_by_history_categories(history, limit)
    if category_recommendations:
        return _fill_to_limit(category_recommendations, limit)

    return _fill_to_limit(_fallback_latest(history, limit), limit)
