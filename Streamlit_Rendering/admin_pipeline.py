# Streamlit_Rendering/admin_pipeline.py
import json
import pandas as pd

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
from Streamlit_Rendering.summary import summarize_text_dummy
from Streamlit_Rendering.trust import score_trust_dummy
ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    """
    더미 크롤링 함수
    URL 1개 → 크롤링 → (중복 필터링) → DB 적재
    반환: {"status": "inserted"/"skipped"/"error", "message": "...", "url": "..."}
    """
    try:
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다. (중복 스킵)", "url": url}

        df_raw = fetch_article_from_url(url=url, source=source)
        df_ready = build_ready_rows(df_raw)

        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}
    

def run_summary(full_text: str) -> str:
    return summarize_text_dummy(full_text, max_chars=50)

def run_keywords(full_text: str) -> list[str]:
    return []

def run_embedding(text: str) -> list[float]:
    return []

def run_trust(full_text: str, source: str) -> dict:
    return score_trust_dummy(full_text, source=source, low=30, high=100)

def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        source = str(r["source"])

        summary_text = summarize_text_dummy(full_text, max_chars=50)
        trust = score_trust_dummy(full_text, source=source, low=30, high=100)

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": source,
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,

            # 모델링 포맷 확정 전: 더미/빈 값
            "summary_text": summary_text,
            "keywords": json.dumps([], ensure_ascii=False),
            "embed_full": json.dumps([]),
            "embed_summary": json.dumps([]),

            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(trust.get("per_criteria", {}), ensure_ascii=False),

            "status": "ready",
        })

    df_ready = pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
    return df_ready
