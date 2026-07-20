"""오프라인 리플레이 평가 — 실제 event_logs를 재생하며 추천 전략별 랭킹 품질을 측정한다.

각 클릭 시점에 "그 이전까지의 히스토리"만으로 후보(클릭 기사 1 + 네거티브 N)를
랭킹했을 때, 실제 클릭 기사를 얼마나 상위에 올리는지를 전략별로 비교한다.

지표: HitRate@K (top-K 안에 정답 포함 비율), MRR, NDCG@K
전략: profile(임베딩 recency-decay 프로필) / hybrid(경량 재정렬)
      / category(히스토리 카테고리 매칭) / latest(최신순) / random(무작위 기준선)

torch 불필요 — 서빙 환경과 동일한 조건에서 실행 가능.

사용법:
    python -m backend.training.evaluate_offline [--k 5] [--negative-count 9] [--min-history 1]
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from collections import Counter

from backend.services import recommend, repo
from backend.services.recommend import RECENCY_HALF_LIFE, build_profile_vector
from backend.training.samples import InteractionSample, load_samples_from_db


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_article_meta() -> dict[str, dict]:
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT article_id, category, published_at, trust_score, embed_summary FROM articles"
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        con.close()

    meta: dict[str, dict] = {}
    for article_id, category, published_at, trust_score, embed_summary in rows:
        meta[str(article_id)] = {
            "article_id": str(article_id),
            "category": str(category or "unknown"),
            "published_at": str(published_at or ""),
            "trust_score": int(trust_score or 0),
            "embedding": repo._parse_vector(embed_summary),
        }
    return meta


def rank_by_profile(sample: InteractionSample, meta: dict[str, dict], candidates: list[str]) -> list[str]:
    history_articles = [
        {"embed_summary": meta[article_id]["embedding"]}
        for article_id in sample.history_article_ids
        if article_id in meta and meta[article_id]["embedding"]
    ]
    profile = build_profile_vector(history_articles)
    if not any(profile):
        return list(candidates)
    scored = [
        (
            _cosine(profile, meta[cid]["embedding"]) if meta.get(cid, {}).get("embedding") else -1.0,
            cid,
        )
        for cid in candidates
    ]
    return [cid for _, cid in sorted(scored, key=lambda item: item[0], reverse=True)]


def rank_by_category(sample: InteractionSample, meta: dict[str, dict], candidates: list[str]) -> list[str]:
    weights: Counter = Counter()
    history = sample.history_article_ids
    for index, article_id in enumerate(history):
        category = meta.get(article_id, {}).get("category", "unknown")
        age = len(history) - 1 - index
        weights[category] += 0.5 ** (age / RECENCY_HALF_LIFE)

    def key(cid: str):
        info = meta.get(cid, {})
        return (
            weights.get(info.get("category", "unknown"), 0.0),
            info.get("published_at", ""),
            cid,
        )

    return sorted(candidates, key=key, reverse=True)


def rank_by_hybrid(sample: InteractionSample, meta: dict[str, dict], candidates: list[str]) -> list[str]:
    history_articles = [
        {"embed_summary": meta[article_id]["embedding"]}
        for article_id in sample.history_article_ids
        if article_id in meta and meta[article_id]["embedding"]
    ]
    profile = build_profile_vector(history_articles)
    records = []
    for article_id in candidates:
        info = meta.get(article_id, {})
        records.append({
            "article_id": article_id,
            "score": _cosine(profile, info.get("embedding") or []) if profile else -1.0,
            "category": info.get("category", "unknown"),
            "published_at": info.get("published_at", ""),
            "trust_score": info.get("trust_score", 0),
        })
    history_for_rerank = [
        {"category": meta.get(article_id, {}).get("category", "unknown")}
        for article_id in reversed(sample.history_article_ids)
    ]
    ranked = recommend._rerank_profile_candidates(records, history_for_rerank)
    return [record["article_id"] for record in ranked]


def rank_by_latest(_: InteractionSample, meta: dict[str, dict], candidates: list[str]) -> list[str]:
    return sorted(
        candidates,
        key=lambda cid: (meta.get(cid, {}).get("published_at", ""), cid),
        reverse=True,
    )


def rank_by_random(sample: InteractionSample, _: dict[str, dict], candidates: list[str]) -> list[str]:
    material = f"42|{sample.subject_id}|{sample.positive_article_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


STRATEGIES = {
    "profile": rank_by_profile,
    "hybrid": rank_by_hybrid,
    "category": rank_by_category,
    "latest": rank_by_latest,
    "random": rank_by_random,
}


def evaluate(samples: list[InteractionSample], meta: dict[str, dict], k: int) -> dict[str, dict]:
    results = {name: {"hit": 0, "mrr": 0.0, "ndcg": 0.0, "n": 0} for name in STRATEGIES}
    for sample in samples:
        candidates = [sample.positive_article_id, *sample.negative_article_ids]
        if len(candidates) < 2:
            continue
        for name, rank_fn in STRATEGIES.items():
            ranked = rank_fn(sample, meta, candidates)
            rank = ranked.index(sample.positive_article_id) + 1
            bucket = results[name]
            bucket["n"] += 1
            bucket["hit"] += 1 if rank <= k else 0
            bucket["mrr"] += 1.0 / rank
            bucket["ndcg"] += 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--negative-count", type=int, default=3,
                        help="정답 1개당 과거 impression 기반 네거티브 수 (기본 3)")
    parser.add_argument("--min-history", type=int, default=1,
                        help="평가에 포함할 최소 히스토리 길이")
    args = parser.parse_args()

    print("event_logs에서 상호작용 샘플 로딩 중...")
    samples = load_samples_from_db(
        history_size=20,
        negative_count=args.negative_count,
        sampling_mode="temporal",
    )
    samples = [s for s in samples if len(s.history_article_ids) >= args.min_history]
    if not samples:
        print("평가할 샘플이 없습니다. 클릭/노출 로그가 더 쌓인 뒤 다시 실행하세요.")
        print("  (최소 조건: 같은 세션·유저에서 클릭 2회 이상)")
        return

    print(f"샘플 {len(samples)}개 · 후보 {1 + args.negative_count}개 중 정답 1개 랭킹 평가\n")
    meta = load_article_meta()
    embedded = sum(1 for m in meta.values() if m["embedding"])
    print(f"기사 {len(meta)}개 중 임베딩 보유 {embedded}개\n")

    results = evaluate(samples, meta, args.k)

    header = f"{'strategy':<10} {'n':>5} {'HitRate@' + str(args.k):>11} {'MRR':>7} {'NDCG@' + str(args.k):>8}"
    print(header)
    print("-" * len(header))
    for name, bucket in sorted(results.items(), key=lambda item: -item[1]["mrr"]):
        n = bucket["n"] or 1
        print(
            f"{name:<10} {bucket['n']:>5} "
            f"{bucket['hit'] / n:>11.4f} {bucket['mrr'] / n:>7.4f} {bucket['ndcg'] / n:>8.4f}"
        )
    print("\n해석: profile이 latest/random보다 유의미하게 높아야 임베딩 개인화가 동작하는 것.")
    print("      category ≈ profile이면 임베딩이 카테고리 이상의 정보를 주지 못하는 상태.")


if __name__ == "__main__":
    main()
