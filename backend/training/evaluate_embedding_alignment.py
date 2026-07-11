"""Measure learned/raw embedding compatibility and profile replay impact."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.services import repo
from backend.training.samples import load_samples_from_db


def _vector(value):
    parsed = repo._parse_vector(value)
    if not parsed:
        return None
    vector = np.asarray(parsed, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def load_spaces():
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT article_id, embed_summary, learned_embedding FROM articles")
        article_rows = cur.fetchall()
        cur.execute("SELECT article_id, embedding FROM article_chunks WHERE embedding IS NOT NULL")
        chunk_rows = cur.fetchall()
    finally:
        cur.close()
        con.close()
    raw = {str(article_id): _vector(value) for article_id, value, _ in article_rows}
    learned = {str(article_id): _vector(value) for article_id, _, value in article_rows}
    raw = {key: value for key, value in raw.items() if value is not None}
    learned = {key: value for key, value in learned.items() if value is not None}
    chunks = defaultdict(list)
    for article_id, value in chunk_rows:
        vector = _vector(value)
        if vector is not None:
            chunks[str(article_id)].append(vector)
    return raw, learned, dict(chunks)


def _retrieval_metrics(queries, candidates, score_fn):
    ranks = []
    for article_id, query in queries.items():
        if article_id not in candidates:
            continue
        ordered = [
            candidate_id
            for _, candidate_id in sorted(
                ((score_fn(query, value), candidate_id) for candidate_id, value in candidates.items()),
                reverse=True,
            )
        ]
        ranks.append(ordered.index(article_id) + 1)
    return {
        "n": len(ranks),
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
    }


def _profile(history, raw, learned, use_mixed):
    vectors = []
    weights = []
    for index, article_id in enumerate(history[-20:]):
        vector = learned.get(article_id) if use_mixed else None
        vector = vector if vector is not None else raw.get(article_id)
        if vector is not None:
            age = len(history[-20:]) - 1 - index
            vectors.append(vector)
            weights.append(0.5 ** (age / 5.0))
    if not vectors:
        return None
    profile = np.average(np.stack(vectors), axis=0, weights=np.asarray(weights))
    norm = float(np.linalg.norm(profile))
    return profile / norm if norm > 0 else None


def _replay(samples, raw, learned, use_mixed, k=3):
    ranks = []
    for sample in samples:
        candidates = [sample.positive_article_id, *sample.negative_article_ids]
        if any(article_id not in raw for article_id in candidates):
            continue
        profile = _profile(sample.history_article_ids, raw, learned, use_mixed)
        if profile is None:
            continue
        scores = [(float(profile @ raw[article_id]), article_id) for article_id in candidates]
        ordered = [article_id for _, article_id in sorted(scores, reverse=True)]
        ranks.append(ordered.index(sample.positive_article_id) + 1)
    return {
        "n": len(ranks),
        f"hit_rate_at_{k}": sum(rank <= k for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        f"ndcg_at_{k}": sum(1 / math.log2(rank + 1) if rank <= k else 0 for rank in ranks) / len(ranks),
    }


def evaluate():
    raw, learned, chunks = load_spaces()
    paired_ids = sorted(set(raw) & set(learned))
    paired_cosines = [float(raw[article_id] @ learned[article_id]) for article_id in paired_ids]
    chunk_candidates = {
        article_id: vectors
        for article_id, vectors in chunks.items()
        if vectors
    }
    learned_to_chunks = _retrieval_metrics(
        {article_id: learned[article_id] for article_id in paired_ids},
        chunk_candidates,
        lambda query, vectors: max(float(query @ vector) for vector in vectors),
    )
    raw_to_chunks = _retrieval_metrics(
        {article_id: raw[article_id] for article_id in paired_ids},
        chunk_candidates,
        lambda query, vectors: max(float(query @ vector) for vector in vectors),
    )
    samples = load_samples_from_db(history_size=20, negative_count=3, sampling_mode="temporal")
    return {
        "raw_articles": len(raw),
        "learned_articles": len(learned),
        "paired_articles": len(paired_ids),
        "paired_cosine_mean": float(np.mean(paired_cosines)),
        "paired_cosine_median": float(np.median(paired_cosines)),
        "learned_to_chunks": learned_to_chunks,
        "raw_to_chunks": raw_to_chunks,
        "legacy_mixed_profile_replay": _replay(samples, raw, learned, True),
        "raw_profile_replay": _replay(samples, raw, learned, False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="FINAL_ANALYSIS/artifacts/optimization/embedding_alignment.json")
    args = parser.parse_args()
    result = evaluate()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
