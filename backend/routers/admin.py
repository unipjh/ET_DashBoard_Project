from fastapi import APIRouter, BackgroundTasks
from backend.schemas import AdminStats
from backend.services import repo
from backend.services.admin_pipeline import build_ready_rows_from_naver
from backend.services.crawl import fetch_articles_from_naver
from pydantic import BaseModel

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
