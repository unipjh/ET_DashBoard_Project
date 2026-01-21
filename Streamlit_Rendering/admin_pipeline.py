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

    # [수정] 크롤러가 반환하는 실제 컬럼명으로 변경
    required_cols = ["date", "category", "title", "reporter", "comment_cnt", "like_cnt", "link", "content"]
    for c in required_cols:
        if c not in df_raw.columns:
            raise ValueError(f"df_raw에 필수 컬럼이 없습니다: {c}")

    for _, r in df_raw.iterrows():
        # [수정] 요청하신 매핑 규칙 적용
        reporter = str(r["reporter"])   
        title = str(r["title"])
        category = str(r["category"])     
        link = str(r["link"])            
        date = str(r["date"])   
        content = str(r["content"])    
        category = str(r["category"])
        comment_cnt = int(r["comment_cnt"])
        like_cnt = int(r["like_cnt"])

        # 기존 분석 로직 유지
        summary_text = run_summary(full_text)
        keywords = run_keywords(full_text)
        embed_full = run_embedding(full_text)
        embed_summary = run_embedding(summary_text)
        trust = run_trust(full_text, source)

        rows.append({
            "reporter": reporter,
            "title": title,
            "category": category,
            "link": link,
            "date": date,
            "content": content,
            "comment_cnt": comment_cnt,
            "like_cnt": like_cnt,

            "summary_text": summary_text,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "embed_full": json.dumps(embed_full),
            "embed_summary": json.dumps(embed_summary),

            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(
                trust.get("per_criteria", {}), ensure_ascii=False
            ),

            "status": "pending",
        })

    return pd.DataFrame(rows)

