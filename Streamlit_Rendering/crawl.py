# Streamlit_Rendering/crawl.py
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone


DEFAULT_HEADERS = {
    # 너무 과하지 않게, 기본 UA 정도만
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def fetch_article_from_url(url: str, source: str = "manual", timeout_sec: int = 10) -> pd.DataFrame:
    """
    URL 1개를 실제로 GET 요청해서 title/full_text를 뽑아오는 '최소 크롤러'
    - 반환 스키마(필수):
      article_id, title, source, url, published_at, full_text
    """
    now = datetime.now(timezone.utc).isoformat()

    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_sec)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # title: <title> 태그를 1순위로 사용
    title = _clean_text(soup.title.get_text()) if soup.title else "Untitled"

    # 본문: 사이트별로 다르므로 '일단' 가장 단순하게 전체 텍스트를 가져오되,
    # 너무 길면 일부만 저장(데모 목적)
    body_text = _clean_text(soup.get_text(" "))

    if len(body_text) > 4000:
        body_text = body_text[:4000] + "..."

    # 지금 단계에서는 article_id는 "timestamp 기반"으로 생성해서
    # 같은 url을 눌러도 '추가'는 되지만,
    # 중복 필터링을 켜면 스킵됩니다(아래 admin_pipeline에서 처리).
    article_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    df = pd.DataFrame([{
        "article_id": article_id,
        "title": title,
        "source": source,
        "url": url,
        "published_at": now,
        "full_text": body_text,
    }])
    return df
