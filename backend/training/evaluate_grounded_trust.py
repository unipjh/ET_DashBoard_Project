"""Evaluate optional Google Search grounding as a trust-score correction.

The SNU label applies to ``_fc_title`` (a fact-check claim), not to the whole
article.  This script therefore grounds that claim and treats the article body
only as context.  It is an offline experiment and does not change production
scoring unless a separate adoption decision is made.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash"
W2 = {
    "source_credibility": 0.20,
    "evidence_support": 0.20,
    "style_neutrality": 0.25,
    "logical_consistency": 0.15,
    "clickbait_risk": -0.20,
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    plain = cleaned.replace("*", "")
    score_match = re.search(r"(?:사실성\s*점수|factuality_score)\s*[:：]\s*(\d{1,2})", plain, re.I)
    status_match = re.search(
        r"(?:판단|status)\s*[:：]\s*(supported|mostly_supported|mixed|mostly_unsupported|unsupported|unverifiable)",
        plain,
        re.I,
    )
    if score_match and status_match:
        return {
            "factuality_score": int(score_match.group(1)),
            "status": status_match.group(1).lower(),
            "reason": re.sub(r"\s+", " ", cleaned)[:1000],
        }
    raise ValueError(f"JSON or fallback fields not found: {text[:200]}")


def ground_claim(client: genai.Client, item: dict) -> dict:
    claim = item.get("_fc_title") or item.get("title") or item.get("text", "")[:300]
    context = item.get("text", "")[:5000]
    prompt = f"""당신은 한국어 팩트체커다. Google 검색으로 아래 '검증 주장'의 사실성을
현재 공개된 독립적이고 신뢰 가능한 자료와 교차검증하라. 기사 본문은 주장 이해용 문맥일
뿐 근거 그 자체로 세지 않는다. 검색으로 확인되지 않으면 추측하지 않는다.

[검증 주장]
{claim}

[기사 문맥]
{context}

반드시 JSON 한 개만 출력하라:
{{"factuality_score": 0부터 10 사이 정수,
  "status": "supported|mostly_supported|mixed|mostly_unsupported|unsupported|unverifiable",
  "reason": "핵심 근거와 반증을 2문장 이내로 요약"}}

점수 기준: 0=명백히 거짓, 2=대부분 거짓, 5=절반/맥락 의존,
8=대부분 사실, 10=명백히 사실. 자료 부족은 5가 아니라 unverifiable과 4점으로 둔다."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        ),
    )
    parsed = _json_object(response.text or "")
    score = int(parsed["factuality_score"])
    if not 0 <= score <= 10:
        raise ValueError(f"factuality_score out of range: {score}")
    sources = []
    candidate = response.candidates[0] if response.candidates else None
    metadata = getattr(candidate, "grounding_metadata", None)
    for chunk in (getattr(metadata, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and getattr(web, "uri", None):
            sources.append({"title": getattr(web, "title", ""), "uri": web.uri})
    return {
        "id": str(item.get("_id", "")),
        "claim": claim,
        "label": float(item["label"]),
        "factuality_score": score,
        "status": parsed.get("status", ""),
        "reason": parsed.get("reason", ""),
        "sources": sources,
        "model": MODEL,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return float("nan")
    rank_sum = float(_rank(scores)[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    rho = float(np.corrcoef(_rank(scores), _rank(labels))[0, 1]) if len(scores) > 1 else float("nan")
    extreme = (labels <= 0.25) | (labels >= 0.75)
    extreme_labels = (labels[extreme] >= 0.75).astype(float)
    auc = _auc(scores[extreme], extreme_labels) if extreme.sum() else float("nan")
    return {"n": int(len(scores)), "extreme_n": int(extreme.sum()), "spearman_rho": rho, "extreme_auc": auc}


def _w2_score(result: dict) -> float:
    criteria = result["per_criteria"]
    raw = sum(criteria[key]["score"] * weight for key, weight in W2.items())
    score = raw / (sum(weight for weight in W2.values() if weight > 0) * 10) * 100
    evidence = criteria["evidence_support"]["score"]
    neutrality = criteria["style_neutrality"]["score"]
    if evidence < 4 and neutrality < 4:
        score -= 10
    if evidence < 4:
        score = min(score, 69)
    return float(max(0, min(100, int(score))))


def _stratified_folds(labels: np.ndarray, fold_count: int = 5) -> np.ndarray:
    folds = np.zeros(len(labels), dtype=int)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        for position, index in enumerate(indices):
            folds[index] = position % fold_count
    return folds


def _cross_validated_blend(base_scores: np.ndarray, fact_scores: np.ndarray, labels: np.ndarray) -> dict:
    alphas = np.linspace(0, 1, 11)
    folds = _stratified_folds(labels)
    predictions = np.empty(len(labels), dtype=float)
    selected = []
    for fold in range(5):
        validation = folds == fold
        train = ~validation
        base_train = metrics(base_scores[train], labels[train])
        eligible = []
        for alpha in alphas:
            train_scores = (1 - alpha) * base_scores[train] + alpha * fact_scores[train]
            result = metrics(train_scores, labels[train])
            if np.isnan(result["extreme_auc"]) or result["extreme_auc"] >= base_train["extreme_auc"]:
                eligible.append((result["spearman_rho"], -alpha, alpha))
        alpha = max(eligible)[2]
        predictions[validation] = (1 - alpha) * base_scores[validation] + alpha * fact_scores[validation]
        selected.append(float(alpha))
    return {
        "folds": 5,
        "selected_alphas": selected,
        "mean_alpha": float(np.mean(selected)),
        "out_of_fold": metrics(predictions, labels),
        "base_same_samples": metrics(base_scores, labels),
    }


def evaluate_corrections(data: list[dict], base: list[dict], grounded: list[dict]) -> dict:
    base_by_id = {
        str(source.get("_id", index)): result
        for index, (source, result) in enumerate(zip(data, base))
    }
    usable = [item for item in grounded if item["id"] in base_by_id]
    if not usable:
        return {"n": 0, "note": "사용 가능한 grounding 캐시가 없다."}
    labels = np.asarray([item["label"] for item in usable], dtype=float)
    base_scores = np.asarray([_w2_score(base_by_id[item["id"]]) for item in usable], dtype=float)
    fact_scores = np.asarray([item["factuality_score"] * 10 for item in usable], dtype=float)
    candidates = {}
    for alpha in np.linspace(0, 1, 11):
        combined = (1 - alpha) * base_scores + alpha * fact_scores
        candidates[f"alpha_{alpha:.1f}"] = metrics(combined, labels)
    return {
        "base": metrics(base_scores, labels),
        "grounded_only": metrics(fact_scores, labels),
        "blends": candidates,
        "five_fold_selection": _cross_validated_blend(base_scores, fact_scores, labels) if len(usable) >= 10 else None,
        "note": "전체 blends는 동일 데이터 진단값이다. five_fold_selection의 out_of_fold를 채택 판단에 사용한다.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/trust_eval_snu.jsonl")
    parser.add_argument("--base", default="FINAL_ANALYSIS/artifacts/optimization/trust_snu_w4_results.jsonl")
    parser.add_argument("--cache", default="FINAL_ANALYSIS/artifacts/optimization/trust_grounding.jsonl")
    parser.add_argument("--summary", default="FINAL_ANALYSIS/artifacts/optimization/trust_grounding_summary.json")
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    parser.add_argument("--live", action="store_true", help="Allow billable Gemini/Search calls")
    args = parser.parse_args()

    load_dotenv()
    data = load_jsonl(Path(args.data))
    base = load_jsonl(Path(args.base))
    cache_path = Path(args.cache)
    cached = load_jsonl(cache_path) if cache_path.exists() else []
    by_id = {item["id"]: item for item in cached}
    selected = data[: args.limit or None]

    missing = [item for item in selected if str(item.get("_id", "")) not in by_id]
    if missing and not args.live:
        print(f"cached={len(by_id)} missing={len(missing)}; use --live to allow external calls")
    elif missing:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        for index, item in enumerate(missing, 1):
            print(f"grounding {index}/{len(missing)} id={item.get('_id')}", flush=True)
            try:
                result = ground_claim(client, item)
            except Exception as exc:
                print(f"failed id={item.get('_id')}: {exc}", flush=True)
                continue
            by_id[result["id"]] = result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as handle:
                for record in by_id.values():
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    selected_ids = {str(item.get("_id", "")) for item in selected}
    summary = evaluate_corrections(
        data,
        base,
        [item for key, item in by_id.items() if key in selected_ids],
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
