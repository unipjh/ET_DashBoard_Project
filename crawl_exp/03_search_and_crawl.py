"""
crawl_exp/03_search_and_crawl.py
Naver 뉴스 검색(공식 API) → 최적 기사 URL 선택 → 본문 크롤링.

실행:
    python crawl_exp/03_search_and_crawl.py [--limit N] [--types A,B,C,D]

옵션:
    --limit N       처리할 최대 행 수 (기본: 전체)
    --types X,Y     처리할 type 목록 (기본: TYPE_A,TYPE_B,TYPE_C,TYPE_D)
    --delay F       요청 간 딜레이 초 (기본: 0.5)

출력:
    data/snu_crawled.jsonl  — 크롤링 성공 행 누적 저장
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.environ.get("CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", "")

IN_PATH  = Path("data/snu_classified.csv")
OUT_PATH = Path("data/snu_crawled.jsonl")

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"

CRAWL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Referer": "https://news.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ============================================================
# 유틸
# ============================================================
def jaccard(a: str, b: str) -> float:
    s1 = set(re.findall(r"[가-힣a-zA-Z0-9]+", a))
    s2 = set(re.findall(r"[가-힣a-zA-Z0-9]+", b))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0


def strip_html(text: str) -> str:
    """Naver API 응답의 <b> 태그 등 제거."""
    return re.sub(r"<[^>]+>", "", text)


# ============================================================
# Naver 뉴스 검색 API → URL 목록
# ============================================================
def search_naver(query: str, max_results: int = 10) -> list[dict]:
    """
    Naver 뉴스 검색 API로 기사 URL + 제목 반환.
    API Docs: https://developers.naver.com/docs/serviceapi/search/news/news.md
    """
    if not CLIENT_ID or not CLIENT_SECRET:
        print("  [search] CLIENT_ID/SECRET 없음 — .env 확인")
        return []

    params = {
        "query":  query,
        "display": max_results,
        "sort":   "sim",   # 관련도순
    }
    headers = {
        "X-Naver-Client-Id":     CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }

    try:
        r = requests.get(NAVER_API_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  [search] API 요청 실패: {e}")
        return []

    data = r.json()
    results = []
    for item in data.get("items", []):
        title = strip_html(item.get("title", ""))
        url   = item.get("link", "")
        if url.startswith("http"):
            results.append({"url": url, "title": title})

    return results


# ============================================================
# 최적 기사 선택
# ============================================================
def pick_best(candidates: list[dict], query: str, outlet: str) -> dict | None:
    """
    후보 기사 중 query 제목과 가장 유사하고, outlet과 일치하는 기사 선택.
    """
    if not candidates:
        return None

    scored = []
    for c in candidates:
        score = jaccard(query, c["title"])
        # outlet 일치 보너스
        if outlet and outlet in c.get("title", "") + c.get("url", ""):
            score += 0.2
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]

    # 유사도 0.1 미만이면 관련 없는 기사로 판단
    if best_score < 0.10:
        return None

    return best


# ============================================================
# 네이버 기사 본문 크롤링
# ============================================================
def crawl_naver_article(url: str) -> str | None:
    """
    n.news.naver.com 기사 본문 크롤링.
    200자 미만이면 None 반환.
    """
    try:
        r = requests.get(url, headers=CRAWL_HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  [crawl] 요청 실패: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    content_div = soup.select_one("div._article_content, #dic_area, div#articleBodyContents")
    if not content_div:
        return None

    for tag in content_div.select("script, style, .ad_area, .img_desc, figure"):
        tag.decompose()

    text = content_div.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text if len(text) >= 200 else None


def crawl_any_article(url: str) -> str | None:
    """
    네이버 외 언론사 직링크 본문 크롤링 (fallback).
    """
    try:
        r = requests.get(url, headers=CRAWL_HEADERS, timeout=10)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    for selector in ["article", "div#article-view-content-div", "div.article_txt",
                     "div#newsct_article", "div.news_cnt_detail_wrap", "div#content"]:
        div = soup.select_one(selector)
        if div:
            for tag in div.select("script, style, figure, .ad"):
                tag.decompose()
            text = re.sub(r"\s+", " ", div.get_text(" ", strip=True)).strip()
            if len(text) >= 200:
                return text

    return None


# ============================================================
# 메인
# ============================================================
def run(limit: int, types: set[str], delay: float) -> None:
    if not IN_PATH.exists():
        print(f"[crawl] {IN_PATH} 없음 — 02_parse_sources.py 먼저 실행하세요.")
        sys.exit(1)

    if not CLIENT_ID or not CLIENT_SECRET:
        print("[crawl] 오류: .env에 CLIENT_ID / CLIENT_SECRET 없음")
        sys.exit(1)

    # 이미 처리된 id 로드 (중단 후 재시작 지원)
    done_ids: set[str] = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[crawl] 기존 완료 {len(done_ids)}건 스킵")

    rows = []
    with open(IN_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] in types and row["id"] not in done_ids:
                rows.append(row)

    if limit:
        rows = rows[:limit]

    print(f"[crawl] 처리 대상: {len(rows)}건 (type={types})")

    ok, fail, skip = 0, 0, 0
    out_f = open(OUT_PATH, "a", encoding="utf-8")

    for i, row in enumerate(rows, 1):
        query  = row["search_query"].strip()
        outlet = row["outlet"].strip()
        title  = row["title"].strip()
        date   = row["date"].strip()
        judge  = row["judge"].strip()

        print(f"[{i}/{len(rows)}] {title[:50]} ({row['type']})", end=" ... ")

        if not query:
            print("쿼리 없음 — SKIP")
            skip += 1
            continue

        # 1. Naver API 검색
        candidates = search_naver(query)
        time.sleep(delay)

        if not candidates:
            print("검색 결과 없음")
            fail += 1
            continue

        # 2. 최적 기사 선택
        best = pick_best(candidates, query, outlet)
        if not best:
            print("유사도 미달")
            fail += 1
            continue

        # 3. 본문 크롤링
        url = best["url"]
        if "n.news.naver.com" in url or "news.naver.com" in url:
            text = crawl_naver_article(url)
        else:
            text = crawl_any_article(url)
        time.sleep(delay * 0.5)

        if not text:
            print("본문 없음")
            fail += 1
            continue

        # 4. 저장
        sim_score = jaccard(query, best["title"])
        record = {
            "id":          row["id"],
            "date":        date,
            "fc_title":    title,
            "judge":       judge,
            "outlet":      outlet or "미상",
            "type":        row["type"],
            "article_url": url,
            "article_title": best["title"],
            "sim":         round(sim_score, 3),
            "text":        text,
            "source":      outlet or None,
            "label":       _judge_to_label(judge),
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        ok += 1
        print(f"OK (sim={jaccard(query, best['title']):.2f}, len={len(text)})")

    out_f.close()
    print(f"\n[crawl] 완료 — 성공 {ok} / 실패 {fail} / 스킵 {skip}")


def _judge_to_label(judge: str) -> float:
    mapping = {
        "사실":          1.0,
        "대체로 사실":   0.75,
        "절반의 사실":   0.5,
        "판단 유보":     0.5,
        "대체로 사실 아님": 0.25,
        "전혀 사실 아님": 0.0,
    }
    return mapping.get(judge.strip(), 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="처리 최대 행 수 (0=전체)")
    parser.add_argument("--types", default="TYPE_A,TYPE_B,TYPE_C,TYPE_D",
                        help="처리할 type 목록 (콤마 구분)")
    parser.add_argument("--delay", type=float, default=0.5, help="요청 간 딜레이(초)")
    args = parser.parse_args()

    run(
        limit=args.limit,
        types=set(args.types.split(",")),
        delay=args.delay,
    )
