"""
crawl_exp/02_parse_sources.py
source 컬럼을 파싱해 크롤링 가능 여부 및 검색 쿼리를 결정.

실행:
    python crawl_exp/02_parse_sources.py

출력:
    data/snu_classified.csv  — 원본 컬럼 + type / outlet / article_hint 추가
"""

import csv
import re
import sys
from pathlib import Path

IN_PATH  = Path("data/snu_factcheck.csv")
OUT_PATH = Path("data/snu_classified.csv")

# ============================================================
# 알려진 언론사명 패턴 (부분 일치)
# ============================================================
KNOWN_OUTLETS = [
    "연합뉴스", "KBS", "MBC", "SBS", "JTBC", "YTN", "TV조선", "채널A", "MBN",
    "조선일보", "중앙일보", "동아일보", "한겨레", "경향신문",
    "한국일보", "국민일보", "서울신문", "세계일보", "문화일보",
    "서울경제", "한국경제", "매일경제", "머니투데이", "뉴시스", "뉴스1",
    "헤럴드경제", "이데일리", "파이낸셜뉴스", "아주경제", "데일리안",
    "뉴스핌", "뉴스위크", "시사저널", "주간조선",
    "의학신문", "데일리메디", "청년의사",
    "천지일보", "미디어오늘", "오마이뉴스", "프레시안",
    "대한민국 정책브리핑", "정책브리핑",
]

# 크롤링 불가 키워드 (source에 포함되면 SKIP)
SKIP_KEYWORDS = [
    "인터넷 커뮤니티", "커뮤니티 게시물",
    "페이스북", "트위터", "인스타그램", "유튜브",
    "디시인사이드", "네이트판", "인스티즈", "클리앙", "개드립", "시보드",
    "카카오", "밴드",
]

# 발언류 키워드 (기사 검색으로 대체 가능 → TYPE_SPEECH)
SPEECH_KEYWORDS = [
    "국정감사", "국회 발언", "의원총회", "유세 발언", "대통령", "청와대",
    "브리핑", "위원회", "모두발언", "유세", "인터뷰", "포럼", "발표",
]


# ============================================================
# source 필드 파싱
# ============================================================

def extract_outlet(source: str) -> str | None:
    """source 문자열에서 언론사명 추출. 없으면 None."""
    for name in KNOWN_OUTLETS:
        if name in source:
            return name
    return None


def extract_article_hint(source: str) -> str | None:
    """
    source에 포함된 기사 제목 힌트 추출.
    쌍따옴표 또는 꺾쇠 안의 텍스트를 우선.
    """
    # "..." 또는 '...' 형태
    m = re.search(r'["""](.{10,80})["""]', source)
    if m:
        return m.group(1).strip()
    # <...> 형태
    m = re.search(r'[<＜](.{5,60})[>＞]', source)
    if m:
        return m.group(1).strip()
    return None


def classify(row: dict) -> dict:
    """
    한 행의 source를 분석해 type, outlet, article_hint, search_query 결정.

    type 분류:
      TYPE_A  — 언론사명 + 기사 제목 힌트 모두 있음  (검색 품질 최상)
      TYPE_B  — 언론사명만 있음, 제목 힌트 없음      (팩트체크 제목으로 검색)
      TYPE_C  — 언론사 자체 문제제기                 (팩트체크 제목으로 검색)
      TYPE_D  — 발언/성명류 (연설·국정감사 등)        (관련 보도 검색)
      SKIP    — SNS·커뮤니티·유튜브 등               (크롤링 불가)
    """
    source  = row.get("source", "").strip()
    title   = row.get("title", "").strip()
    date    = row.get("date", "").strip()  # YYYY.MM.DD

    # 1. SKIP 판정
    for kw in SKIP_KEYWORDS:
        if kw in source:
            return {**row, "type": "SKIP", "outlet": "", "article_hint": "", "search_query": ""}

    # 2. 언론사 자체 문제제기
    if "언론사 자체 문제제기" in source or "언론사 자체 문제 제기" in source \
            or "다수 언론" in source:
        return {
            **row,
            "type": "TYPE_C",
            "outlet": "",
            "article_hint": "",
            "search_query": title,  # 팩트체크 주장 자체로 검색
        }

    # 3. 언론사 추출
    outlet = extract_outlet(source)
    hint   = extract_article_hint(source)

    # 4. 발언류 판정 (언론사 없거나 있어도 발언이 주)
    if outlet is None:
        for kw in SPEECH_KEYWORDS:
            if kw in source:
                return {
                    **row,
                    "type": "TYPE_D",
                    "outlet": "",
                    "article_hint": "",
                    "search_query": title,  # 관련 보도 검색
                }
        # 언론사도 없고 발언류도 아님 → SKIP
        return {**row, "type": "SKIP", "outlet": "", "article_hint": "", "search_query": ""}

    # 5. 언론사 있음
    if hint:
        # TYPE_A: 언론사 + 기사 힌트
        return {
            **row,
            "type": "TYPE_A",
            "outlet": outlet,
            "article_hint": hint,
            "search_query": hint,       # 기사 제목 힌트로 정밀 검색
        }
    else:
        # TYPE_B: 언론사만
        return {
            **row,
            "type": "TYPE_B",
            "outlet": outlet,
            "article_hint": "",
            "search_query": title,      # 팩트체크 주장으로 검색
        }


# ============================================================
# 실행
# ============================================================
def run() -> None:
    if not IN_PATH.exists():
        print(f"[parse] {IN_PATH} 없음 — 01_fetch_data.py 먼저 실행하세요.")
        sys.exit(1)

    rows = []
    with open(IN_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(classify(row))

    # 통계
    from collections import Counter
    type_counts = Counter(r["type"] for r in rows)
    print("[parse] source 분류 결과:")
    for t, cnt in sorted(type_counts.items()):
        print(f"  {t}: {cnt}건")
    crawlable = sum(v for t, v in type_counts.items() if t != "SKIP")
    print(f"  → 크롤링 시도 대상: {crawlable}건 / 전체 {len(rows)}건")

    # 저장
    fieldnames = list(rows[0].keys())
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[parse] 저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    run()
