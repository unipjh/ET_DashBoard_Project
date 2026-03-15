from fastapi import APIRouter, BackgroundTasks
from backend.schemas import AdminStats
from backend.services import repo
from backend.services.admin_pipeline import build_ready_rows_from_naver
from backend.services.crawl import fetch_articles_from_naver
from backend.services.trust import score_trust
from pydantic import BaseModel
import json

router = APIRouter(prefix="/api/admin", tags=["admin"])


class CrawlRequest(BaseModel):
    max_articles_per_category: int = 10


@router.get("/stats", response_model=AdminStats)
def get_stats():
    df = repo.load_articles()
    if df.empty:
        return AdminStats(total_articles=0, sources=[], unanalyzed_count=0)
    sources = df["source"].dropna().unique().tolist()
    unanalyzed = int((df["trust_score"] == 0).sum())
    return AdminStats(
        total_articles=len(df),
        sources=sources,
        unanalyzed_count=unanalyzed,
    )


def _run_crawl(max_articles: int):
    df_raw = fetch_articles_from_naver(max_articles_per_category=max_articles)
    if not df_raw.empty:
        build_ready_rows_from_naver(df_raw)


@router.post("/crawl")
def start_crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_crawl, req.max_articles_per_category)
    return {"status": "started", "max_articles_per_category": req.max_articles_per_category}


def _run_analyze():
    """trust_score == 0인 기사를 일괄 신뢰도 분석 (백그라운드)."""
    df = repo.load_articles_without_trust()
    if df.empty:
        print("✅ 미분석 기사 없음")
        return
    total = len(df)
    print(f"🔍 미분석 기사 {total}개 신뢰도 분석 시작")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        aid = str(row["article_id"])
        text = str(row.get("full_text", "") or "")
        source = str(row.get("source", "미상") or "미상")
        print(f"  [{i}/{total}] {row.get('title', '')[:40]}")
        result = score_trust(text, source)
        repo.update_article_trust(
            article_id=aid,
            score=result["score"],
            verdict=result["verdict"],
            reason=result["reason"],
            per_criteria=json.dumps(result["per_criteria"], ensure_ascii=False),
        )
    print(f"✅ {total}개 기사 신뢰도 분석 완료")


@router.post("/analyze")
def start_analyze(background_tasks: BackgroundTasks):
    """미분석 기사(trust_score=0) 일괄 신뢰도 분석."""
    df = repo.load_articles_without_trust()
    count = len(df)
    if count == 0:
        return {"status": "nothing_to_analyze", "count": 0}
    background_tasks.add_task(_run_analyze)
    return {"status": "started", "count": count}
