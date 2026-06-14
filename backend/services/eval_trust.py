"""
backend/services/eval_trust.py
trust.py 점수의 타당성 검증 스크립트.

=== 모드 1: 레이블 기반 평가 (기존) ===
    python -m backend.services.eval_trust --data data/trust_eval_samples.jsonl

=== 모드 2: 불변 조건 쌍 테스트 + LLM 캐시 저장 ===
    python -m backend.services.eval_trust --data data/invariant_pairs.jsonl --save-raw
    → data/eval_cache.jsonl 생성 (pair_id + per_criteria 캐시)

=== 모드 3: 캐시 재활용 + 가중치 교체 (LLM 호출 0회) ===
    python -m backend.services.eval_trust --from-raw data/eval_cache.jsonl --weights W0
    python -m backend.services.eval_trust --from-raw data/eval_cache.jsonl --weights W1
    python -m backend.services.eval_trust --from-raw data/eval_cache.jsonl --weights W2
    python -m backend.services.eval_trust --from-raw data/eval_cache.jsonl --weights W3

입력 형식 (모드 1, jsonl):
    {"text": "...", "source": "연합뉴스", "label": 1}

입력 형식 (모드 2, jsonl):
    pair  → {"pair_id": "P1", "type": "pair",   "a": {...}, "a_prime": {...}, "expected": "a_gt_a_prime"}
    single→ {"pair_id": "P9", "type": "single",  "a": {...}, "expected_range": [40, 69]}
"""

import argparse
import json
import sys
from pathlib import Path

# Windows 터미널 UTF-8 출력 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn.metrics import classification_report, confusion_matrix
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

from backend.services.trust import score_trust

# ============================================================
# 가중치 실험 후보군 (trust_dev3_plan.md §2-1)
# ============================================================
WEIGHT_VARIANTS: dict[str, dict] = {
    "W0": {  # 기준선 (현재)
        "source_credibility":  0.25,
        "evidence_support":    0.25,
        "style_neutrality":    0.20,
        "logical_consistency": 0.20,
        "clickbait_risk":     -0.10,
    },
    "W1": {  # 스타일 + 클릭베이트 강화
        "source_credibility":  0.20,
        "evidence_support":    0.25,
        "style_neutrality":    0.25,
        "logical_consistency": 0.15,
        "clickbait_risk":     -0.15,
    },
    "W2": {  # 클릭베이트 최대 페널티
        "source_credibility":  0.20,
        "evidence_support":    0.20,
        "style_neutrality":    0.25,
        "logical_consistency": 0.15,
        "clickbait_risk":     -0.20,
    },
    "W3": {  # 스타일만 올리고 나머지 소폭 조정
        "source_credibility":  0.20,
        "evidence_support":    0.25,
        "style_neutrality":    0.25,
        "logical_consistency": 0.20,
        "clickbait_risk":     -0.10,
    },
    "W4": {  # source 추가 하향 + evidence 상향 (P8 역전 해소 목적)
        "source_credibility":  0.15,
        "evidence_support":    0.30,
        "style_neutrality":    0.25,
        "logical_consistency": 0.20,
        "clickbait_risk":     -0.10,
    },
}


# ============================================================
# 유틸
# ============================================================
def _max_raw(weights: dict) -> float:
    return sum(w for w in weights.values() if w > 0) * 10


def _apply_weights(per_criteria: dict, weights: dict) -> int:
    """per_criteria + 지정 가중치로 0~100 점수 재계산 (교차 패널티 포함)."""
    raw = sum(per_criteria[k]["score"] * w for k, w in weights.items())
    score_100 = (raw / _max_raw(weights)) * 100
    
    # 교차 패널티: style_neutrality < 4 AND evidence_support < 4 → -10
    neutrality = per_criteria["style_neutrality"]["score"]
    evidence   = per_criteria["evidence_support"]["score"]
    if neutrality < 4 and evidence < 4:
        score_100 -= 10.0
        
    return int(max(0, min(100, score_100)))


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[eval] 라인 {lineno} 파싱 오류: {e}", file=sys.stderr)
    return items


# ============================================================
# 모드 1: 레이블 기반 평가 (기존)
# ============================================================
def run_label_eval(samples: list[dict]) -> list[dict]:
    results = []
    for i, s in enumerate(samples, 1):
        text   = s.get("text", "")
        source = s.get("source", None)
        label  = s.get("label", 0.5)
        print(f"[eval] ({i}/{len(samples)}) source={source} label={label} ...", end=" ", flush=True)
        try:
            result = score_trust(text, source, s.get("title"))
            result["label"] = label
            results.append(result)
            print(f"score={result['score']} verdict={result['verdict']}")
        except Exception as e:
            print(f"실패: {e}", file=sys.stderr)
    return results


def print_label_stats(results: list[dict]) -> None:
    if not _DEPS_OK:
        print("\n[eval] scipy / sklearn / matplotlib 미설치 — 기본 통계만 출력")
        scores = [r["score"] for r in results]
        print(f"  샘플 수: {len(results)}")
        print(f"  점수 평균: {sum(scores)/len(scores):.1f}")
        trust_scores    = [r["score"] for r in results if r["label"] == 1]
        distrust_scores = [r["score"] for r in results if r["label"] == 0]
        if trust_scores:
            print(f"  신뢰 그룹 평균: {sum(trust_scores)/len(trust_scores):.1f}  (목표 ≥ 65)")
        if distrust_scores:
            print(f"  불신 그룹 평균: {sum(distrust_scores)/len(distrust_scores):.1f}  (목표 ≤ 45)")
        return

    scores = np.array([r["score"] for r in results])
    labels = np.array([r["label"] for r in results])

    rho, pval = stats.spearmanr(scores, labels)
    print(f"\n[eval] Spearman ρ = {rho:.3f}  (p={pval:.4f})  목표 ≥ 0.5")

    trust_mask    = labels == 1
    distrust_mask = labels == 0
    if trust_mask.any():
        print(f"[eval] 신뢰 그룹 평균: {scores[trust_mask].mean():.1f}  목표 ≥ 65")
    if distrust_mask.any():
        print(f"[eval] 불신 그룹 평균: {scores[distrust_mask].mean():.1f}  목표 ≤ 45")

    cb_scores   = np.array([r["per_criteria"]["clickbait_risk"]["score"] for r in results])
    cb_pred     = (cb_scores <= 5).astype(int)
    true_labels = (labels == 0).astype(int)
    print("\n[eval] clickbait_risk 분류 리포트:")
    print(classification_report(true_labels, 1 - cb_pred, target_names=["신뢰", "낚시성"]))

    verdict_map = {"likely_true": 1, "uncertain": 0.5, "likely_false": 0}
    verdict_scores = np.array([verdict_map.get(r["verdict"], 0.5) for r in results])
    print("[eval] verdict vs label 혼동 행렬:")
    print(confusion_matrix(labels >= 0.75, verdict_scores >= 0.75))

    criteria_keys = ["source_credibility", "evidence_support", "style_neutrality",
                     "logical_consistency", "clickbait_risk"]
    data = [[r["per_criteria"][k]["score"] for r in results] for k in criteria_keys]
    _, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=criteria_keys, vert=True)
    ax.set_title("기준별 점수 분포")
    ax.set_ylabel("점수 (0~10)")
    plt.tight_layout()
    plt.savefig("eval_boxplot.png")
    print("[eval] boxplot 저장: eval_boxplot.png")


# ============================================================
# 모드 2: 불변 조건 쌍 테스트 + --save-raw
# ============================================================
def run_pair_eval_with_save(pairs: list[dict], cache_path: str) -> list[dict]:
    """
    LLM을 호출하여 각 샘플의 per_criteria를 계산하고
    cache_path(jsonl)에 저장한다.
    반환: 캐시 레코드 목록
    """
    cache_records = []
    total = sum(2 if p["type"] == "pair" else 1 for p in pairs)
    idx = 0

    for p in pairs:
        pair_id = p["pair_id"]
        samples = [("a", p["a"])]
        if p["type"] == "pair":
            samples.append(("a_prime", p["a_prime"]))

        for side, sample in samples:
            idx += 1
            source = sample.get("source")
            text   = sample.get("text", "")
            print(f"[eval] ({idx}/{total}) {pair_id}-{side} source={source} ...", end=" ", flush=True)
            try:
                result = score_trust(text, source, sample.get("title"))
                record = {
                    "pair_id":      pair_id,
                    "side":         side,
                    "score":        result["score"],
                    "verdict":      result["verdict"],
                    "per_criteria": result["per_criteria"],
                }
                cache_records.append(record)
                print(f"score={result['score']}")
            except Exception as e:
                print(f"실패: {e}", file=sys.stderr)

    # 캐시 저장
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for rec in cache_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[eval] 캐시 저장 완료: {cache_path} ({len(cache_records)}건)")
    return cache_records


# ============================================================
# 모드 3: 캐시 재활용 + 가중치 교체
# ============================================================
def run_from_cache(cache_records: list[dict], pairs: list[dict], weights: dict, weight_name: str) -> None:
    """
    캐시에서 per_criteria를 로드하고 지정 가중치로 점수를 재계산한다.
    LLM 호출 없음.
    """
    # pair_id + side → per_criteria 인덱스
    cache_index: dict[tuple, dict] = {}
    for rec in cache_records:
        cache_index[(rec["pair_id"], rec["side"])] = rec

    passed = 0
    failed = 0
    uncertain_count = 0

    print(f"\n{'='*55}")
    print(f"  가중치 조합: {weight_name}")
    print(f"{'='*55}")

    for p in pairs:
        pair_id  = p["pair_id"]
        ptype    = p["type"]
        signal   = p.get("signal", "")

        if ptype == "pair":
            rec_a      = cache_index.get((pair_id, "a"))
            rec_prime  = cache_index.get((pair_id, "a_prime"))
            if not rec_a or not rec_prime:
                print(f"  {pair_id} {signal}: 캐시 누락 — SKIP")
                continue

            score_a      = _apply_weights(rec_a["per_criteria"],     weights)
            score_prime  = _apply_weights(rec_prime["per_criteria"],  weights)
            expected     = p.get("expected", "a_gt_a_prime")

            if expected == "a_gt_a_prime":
                result = "PASS ✅" if score_a > score_prime else "FAIL ❌"
                ok = score_a > score_prime
            else:  # a_prime_gt_a (P8 역방향)
                result = "PASS ✅" if score_prime > score_a else "FAIL ❌"
                ok = score_prime > score_a

            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  {pair_id} {signal}: score(A)={score_a} vs score(A')={score_prime} → {result}")

        else:  # single
            rec_a = cache_index.get((pair_id, "a"))
            if not rec_a:
                print(f"  {pair_id} {signal}: 캐시 누락 — SKIP")
                continue

            score_a = _apply_weights(rec_a["per_criteria"], weights)
            lo, hi  = p.get("expected_range", [0, 100])
            ok      = lo <= score_a <= hi
            result  = "PASS ✅" if ok else "FAIL ❌"

            if ok:
                passed += 1
                uncertain_count += 1
            else:
                failed += 1

            print(f"  {pair_id} {signal}: score={score_a}  기대범위=[{lo},{hi}] → {result}")

    total = passed + failed
    print(f"\n  결과: {passed}/{total} 통과")
    if total > 0:
        rate = passed / total
        if rate >= 0.8:
            print(f"  → ✅ 채택 기준 충족 (≥80%)")
        elif rate >= 0.6:
            print(f"  → ⚠️  일부 기준 재검토 필요 (60~79%)")
        else:
            print(f"  → ❌ 가중치 조합 기각 (<60%)")


# ============================================================
# 진입점
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="trust.py 점수 검증")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data",     help="JSONL 샘플 파일 (레이블 기반 또는 불변 쌍)")
    group.add_argument("--from-raw", dest="from_raw", help="캐시 JSONL 경로 (LLM 호출 없음)")

    parser.add_argument("--save-raw",  action="store_true",
                        help="--data 모드에서 per_criteria를 캐시 파일로 저장")
    parser.add_argument("--cache-out", default="data/eval_cache.jsonl",
                        help="--save-raw 저장 경로 (기본: data/eval_cache.jsonl)")
    parser.add_argument("--weights",   default=None,
                        choices=list(WEIGHT_VARIANTS.keys()),
                        help="--from-raw 모드에서 사용할 가중치 조합 (W0~W3)")
    parser.add_argument("--pairs-ref", default=None,
                        help="--from-raw 모드에서 쌍 정의 파일 (기본: data/invariant_pairs.jsonl)")

    args = parser.parse_args()

    # ── 모드 3: 캐시 재활용
    if args.from_raw:
        cache_path = Path(args.from_raw)
        if not cache_path.exists():
            print(f"[eval] 캐시 파일 없음: {cache_path}", file=sys.stderr)
            sys.exit(1)

        cache_records = load_jsonl(str(cache_path))
        print(f"[eval] 캐시 로드: {len(cache_records)}건 ({cache_path})")

        # 쌍 정의 파일
        pairs_ref_path = args.pairs_ref or "data/invariant_pairs.jsonl"
        if not Path(pairs_ref_path).exists():
            print(f"[eval] 쌍 정의 파일 없음: {pairs_ref_path}", file=sys.stderr)
            sys.exit(1)
        pairs = load_jsonl(pairs_ref_path)

        # 가중치 지정 없으면 전체 실험
        targets = [args.weights] if args.weights else list(WEIGHT_VARIANTS.keys())
        for wname in targets:
            run_from_cache(cache_records, pairs, WEIGHT_VARIANTS[wname], wname)
        return

    # ── 모드 1 / 2: --data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[eval] 파일 없음: {data_path}", file=sys.stderr)
        sys.exit(1)

    items = load_jsonl(str(data_path))
    print(f"[eval] 데이터 로드: {len(items)}건 ({data_path})")

    # type 필드 있으면 불변 조건 쌍 파일로 판단
    is_pair_file = any("type" in item for item in items)

    if is_pair_file:
        # 모드 2: 불변 조건 쌍 테스트
        cache_records = run_pair_eval_with_save(items, args.cache_out)

        if args.save_raw:
            print("[eval] --save-raw 완료. --from-raw 로 가중치 실험을 이어서 실행하세요.")
            print(f"  예: python -m backend.services.eval_trust --from-raw {args.cache_out} --weights W1")
        else:
            # 캐시 저장 후 W0으로 즉시 평가
            print("\n[eval] W0(기준선) 즉시 평가:")
            run_from_cache(cache_records, items, WEIGHT_VARIANTS["W0"], "W0")
    else:
        # 모드 1: 레이블 기반 평가
        results = run_label_eval(items)
        print_label_stats(results)


if __name__ == "__main__":
    main()