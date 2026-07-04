"""
crawl_exp/04_build_eval.py
snu_crawled.jsonl → trust_eval_samples.jsonl 형식으로 변환.

실행:
    python crawl_exp/04_build_eval.py [--per-class N] [--out PATH]

옵션:
    --per-class N   레이블 그룹별 최대 샘플 수 (기본: 10)
    --min-len N     최소 본문 길이 (기본: 300자)
    --out PATH      출력 경로 (기본: data/trust_eval_snu.jsonl)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

IN_PATH  = Path("data/snu_crawled.jsonl")
OUT_PATH = Path("data/trust_eval_snu.jsonl")

# label 범위 → 그룹
def label_group(label: float) -> str:
    if label >= 0.75:
        return "trust"       # 사실 / 대체로 사실
    elif label <= 0.25:
        return "distrust"    # 전혀 사실 아님 / 대체로 사실 아님
    else:
        return "neutral"     # 절반의 사실 / 판단 유보


def run(per_class: int, min_len: int, min_sim: float, out_path: Path) -> None:
    if not IN_PATH.exists():
        print(f"[build] {IN_PATH} 없음 — 03_search_and_crawl.py 먼저 실행하세요.")
        return

    # 그룹별 풀 구성
    pool: dict[str, list[dict]] = defaultdict(list)
    skipped_sim = 0
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            if len(rec.get("text", "")) < min_len:
                continue
            if rec.get("sim") is not None and rec["sim"] < min_sim:
                skipped_sim += 1
                continue
            grp = label_group(rec["label"])
            pool[grp].append(rec)

    if skipped_sim:
        print(f"[build] sim < {min_sim} 제외: {skipped_sim}건")

    print("[build] 그룹별 풀:")
    for g, items in pool.items():
        print(f"  {g}: {len(items)}건")

    # 그룹별 샘플링
    random.seed(42)
    samples = []
    for grp, items in pool.items():
        n = min(per_class, len(items))
        sampled = random.sample(items, n)
        for rec in sampled:
            samples.append({
                "text":   rec["text"],
                "source": rec.get("source") or None,
                "label":  rec["label"],
                # 디버그용 메타
                "_id":         rec.get("id"),
                "_fc_title":   rec.get("fc_title"),
                "_judge":      rec.get("judge"),
                "_article_url": rec.get("article_url"),
            })

    random.shuffle(samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[build] 저장 완료: {out_path}  ({len(samples)}건)")

    # 레이블 분포
    from collections import Counter
    dist = Counter(label_group(s["label"]) for s in samples)
    for g, cnt in dist.items():
        print(f"  {g}: {cnt}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int,   default=10)
    parser.add_argument("--min-len",   type=int,   default=300)
    parser.add_argument("--min-sim",   type=float, default=0.3)
    parser.add_argument("--out",       default=str(OUT_PATH))
    args = parser.parse_args()

    run(
        per_class=args.per_class,
        min_len=args.min_len,
        min_sim=args.min_sim,
        out_path=Path(args.out),
    )
