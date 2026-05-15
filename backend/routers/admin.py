from fastapi import APIRouter, BackgroundTasks
from backend.schemas import AdminStats, CategoryStat, ApiUsage
from backend.services import repo
from backend.services.admin_pipeline import build_ready_rows_from_naver, run_gemini_summary_and_keywords
from backend.services.crawl import fetch_articles_from_naver, compute_jaccard_similarity, NEWS_CATEGORY
from backend.services.trust import score_trust
from pydantic import BaseModel
import json
import time
import pandas as pd
from datetime import datetime
from backend.services.process_status import update_status, reset_status, STATUS
import random

router = APIRouter(prefix="/api/admin", tags=["admin"])

# 임시 API 상태 모니터링 (서버 메모리 유지)
_MOCK_API_STATS = {
    "error_count": 0,
    "last_success": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

class CrawlRequest(BaseModel):
    max_articles_per_category: int = 10
    categories: list[str] | None = None
    total_articles: int | None = None


@router.get("/stats")
def get_stats():
    df = repo.load_articles()
    
    api_usage = ApiUsage(
        last_success_time=_MOCK_API_STATS["last_success"],
        error_count=_MOCK_API_STATS["error_count"],
        quota_percent=round(random.uniform(5.0, 15.0), 1) if df.empty else min(99.9, len(df) * 0.5)
    )
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    cat_stats = []
    
    if df.empty:
        all_cat_names = ["IT/과학", "경제", "사회", "생활/문화", "세계", "정치", "연예", "스포츠"]
        cat_stats = [{"category": c, "total": 0, "unanalyzed": 0, "today_articles": 0} for c in all_cat_names]
        return {
            "total_articles": 0, "sources": [], "unanalyzed_count": 0,
            "category_stats": cat_stats, "api_usage": api_usage
        }
        
    sources = df["source"].dropna().unique().tolist()
    unanalyzed_count = int((df["trust_score"] == 0).sum())
    total_articles = len(df)
    
    # 실제 카테고리별 통계 집계
    cat_stats = []
    if 'category' in df.columns:
        valid_df = df[df['category'].notna() & (df['category'] != "미분류") & (df['category'] != "일반")]
        valid_df = valid_df.copy()
        valid_df['is_today'] = valid_df['published_at'].astype(str).str.startswith(today_str)
        
        category_stats_df = valid_df.groupby('category').agg(
            total=('article_id', 'count'),
            unanalyzed=('trust_score', lambda x: (x == 0).sum()),
            today_articles=('is_today', 'sum')
        ).reset_index()

        cat_stats = [
            {"category": row['category'], "total": row['total'], "unanalyzed": int(row['unanalyzed']), "today_articles": int(row['today_articles'])}
            for _, row in category_stats_df.iterrows()
        ]
        
    # DB에 없는 카테고리도 0으로 표시
    ordered_cats = ["IT/과학", "경제", "사회", "생활/문화", "세계", "정치", "연예", "스포츠"]
    existing_cats = {cs["category"] for cs in cat_stats}
    for cat_name in set(ordered_cats) - existing_cats:
        cat_stats.append({"category": cat_name, "total": 0, "unanalyzed": 0, "today_articles": 0})

    # 지정된 순서대로 정렬
    cat_stats.sort(key=lambda x: ordered_cats.index(x["category"]) if x["category"] in ordered_cats else 999)

    return {
        "total_articles": total_articles,
        "sources": sources,
        "unanalyzed_count": unanalyzed_count,
        "category_stats": cat_stats,
        "api_usage": api_usage
    }


def _run_crawl(max_articles: int, categories: list[str] = None, total_articles: int = None):
    STATUS["process_name"] = "crawl"
    try:
        df_raw = fetch_articles_from_naver(max_articles_per_category=max_articles, categories=categories)
        if total_articles is not None and not df_raw.empty:
            df_raw = df_raw.head(total_articles)
        if not df_raw.empty:
            build_ready_rows_from_naver(df_raw)
    finally:
        time.sleep(2)  # 프론트엔드가 마지막 로그("✅ 작업 완료...")를 읽어갈 수 있도록 대기
        reset_status()


@router.post("/crawl")
def start_crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    STATUS["process_name"] = "crawl"
    update_status("System", "크롤링 작업 시작 중...")
    background_tasks.add_task(_run_crawl, req.max_articles_per_category, req.categories, req.total_articles)
    return {"status": "started", "max_articles_per_category": req.max_articles_per_category, "categories": req.categories, "total_articles": req.total_articles}


@router.get("/process-status")
def get_process_status():
    return STATUS

def _run_analyze():
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
    df = repo.load_articles_without_trust()
    count = len(df)
    if count == 0:
        return {"status": "nothing_to_analyze", "count": 0}
    background_tasks.add_task(_run_analyze)
    return {"status": "started", "count": count}


def _run_dedupe():
    """DB 내 중복 기사 제거 (제목 Jaccard 유사도 80% 이상)"""
    df = repo.load_articles()
    if df.empty:
        print("✅ DB가 비어있습니다.")
        return

    articles = df[["article_id", "title"]].to_dict("records")
    total = len(articles)
    to_delete = set()

    print(f"🔍 DB 내 중복 검사 시작 ({total}개 기사)")

    for i in range(total):
        if articles[i]["article_id"] in to_delete:
            continue
        for j in range(i + 1, total):
            if articles[j]["article_id"] in to_delete:
                continue
            sim = compute_jaccard_similarity(articles[i]["title"], articles[j]["title"])
            if sim >= 0.8:
                # 나중에 들어온 기사(j) 제거
                to_delete.add(articles[j]["article_id"])
                print(f"  🗑️ 중복 제거: {articles[j]['title'][:40]}")

    if not to_delete:
        print("✅ 중복 기사 없음")
        return

    print(f"🗑️ {len(to_delete)}개 중복 기사 삭제 중...")
    repo.delete_articles(list(to_delete))
    print(f"✅ {len(to_delete)}개 중복 기사 삭제 완료")


@router.post("/dedupe")
def start_dedupe(background_tasks: BackgroundTasks):
    """DB 내 중복 기사 일괄 제거"""
    df = repo.load_articles()
    if df.empty:
        return {"status": "empty_db", "count": 0}
    background_tasks.add_task(_run_dedupe)
    return {"status": "started", "total_articles": len(df)}


def _run_extract_keywords():
    """DB에 있는 기존 기사 중 키워드가 없는 기사들의 키워드를 일괄 추출"""
    df = repo.load_articles()
    if df.empty:
        print("✅ DB가 비어있습니다.")
        return

    def needs_extract(kw):
        if pd.isna(kw) or not str(kw).strip() or str(kw).strip() == "[]":
            return True
        try:
            parsed = json.loads(str(kw))
            if isinstance(parsed, list) and len(parsed) > 3:
                return True
            return False
        except:
            return True

    mask = df["keywords"].apply(needs_extract)
    df_missing = df[mask]

    total = len(df_missing)
    if total == 0:
        print("✅ 키워드 추출이 필요한 기사가 없습니다.")
        return

    print(f"🔍 미추출 기사 {total}개 키워드 추출 시작")
    updated_rows = []
    
    for i, (_, row) in enumerate(df_missing.iterrows(), 1):
        text = str(row.get("full_text", "") or "")
        print(f"  [{i}/{total}] {row.get('title', '')[:40]}")

        # 요약과 키워드를 동시에 추출하는 원래 함수 사용 (요약값은 _로 무시하고 키워드만 가져옴)
        _, keywords = run_gemini_summary_and_keywords(text)
        
        updated_row = row.copy()
        updated_row["keywords"] = keywords
        updated_rows.append(updated_row.to_dict())
        
        time.sleep(2)
        
        # 10개 단위로 중간 저장 (Rate limit나 에러 대비)
        if len(updated_rows) >= 10:
            repo.upsert_articles(pd.DataFrame(updated_rows))
            updated_rows = []

    # 남은 데이터 저장
    if updated_rows:
        repo.upsert_articles(pd.DataFrame(updated_rows))

    print(f"✅ {total}개 기사 키워드 일괄 추출 완료")


@router.post("/delete-no-keywords")
def delete_no_keywords():
    count = repo.delete_articles_without_keywords()
    return {"status": "done", "deleted_count": count}


@router.post("/keywords")
def start_keywords(background_tasks: BackgroundTasks):
    """키워드 일괄 추출 백그라운드 작업 시작"""
    df = repo.load_articles()
    if df.empty:
        return {"status": "empty_db", "count": 0}
        
    def needs_extract(kw):
        if pd.isna(kw) or not str(kw).strip() or str(kw).strip() == "[]":
            return True
        try:
            parsed = json.loads(str(kw))
            if isinstance(parsed, list) and len(parsed) > 3:
                return True
            return False
        except:
            return True
            
    mask = df["keywords"].apply(needs_extract)
    count = int(mask.sum())
    
    if count == 0:
        return {"status": "nothing_to_extract", "count": 0}
        
    background_tasks.add_task(_run_extract_keywords)
    return {"status": "started", "count": count}