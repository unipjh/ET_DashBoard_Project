"""
crawl_exp/01_fetch_data.py
GitHub에서 SNU 팩트체크 CSV를 다운로드하고 data/ 에 저장.

실행:
    python crawl_exp/01_fetch_data.py
"""

import csv
import io
import sys
from pathlib import Path

import requests

CSV_URL = "https://raw.githubusercontent.com/startedourmission/False-Information-Dataset-And-Analysis/main/fact_checks_final.csv"
OUT_PATH = Path("data/snu_factcheck.csv")


def fetch() -> None:
    print(f"[fetch] 다운로드: {CSV_URL}")
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()

    # BOM 제거 후 저장
    text = r.content.decode("utf-8-sig")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")

    # 기본 통계
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    from collections import Counter
    labels = Counter(r["judge"] for r in rows)
    print(f"[fetch] 저장 완료: {OUT_PATH}  ({len(rows)}건)")
    print("[fetch] judge 분포:")
    for k, v in labels.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    fetch()
