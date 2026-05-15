from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import RedirectResponse
from backend.services import repo
from backend.schemas import ArticleOut, ArticleDetail, SearchResult, PaginatedArticlesResponse
from backend.services.config import get_gemini_api_key
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

router = APIRouter(prefix="/api/articles", tags=["articles"])

genai.configure(api_key=get_gemini_api_key())
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _embed(text: str) -> list:
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query",
        )
        vec = result["embedding"]
        if len(vec) > 768:
            vec = vec[:768]
        elif len(vec) < 768:
            vec = vec + [0.0] * (768 - len(vec))
        return vec
    except Exception:
        return [0.0] * 768


@router.get("", response_model=PaginatedArticlesResponse)
def get_articles(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    category: str | None = Query(None),
):
    if category:
        articles = repo.get_articles_paginated_by_category(page=page, size=size, category=category)
    else:
        articles = repo.get_articles_paginated(page=page, size=size)
    total_count = repo.get_articles_total_count(category=category)
    return {"articles": articles, "total_count": total_count}


@router.get("/{article_id}")
def get_article(article_id: str):
    # DB에서 특정 ID의 기사만 직접 조회합니다.
    article = repo.get_article_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.get("/{article_id}/related", response_model=list[SearchResult])
def get_related(article_id: str, limit: int = Query(5, ge=1, le=20)):
    article = repo.get_article_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    # 제목 + 요약으로 쿼리 벡터 생성 (embed_summary는 "[]" placeholder이므로 재임베딩)
    query_text = f"{article.get('title', '')} {article.get('summary_text', '')}".strip()
    vec = _embed(query_text)
    related_df = repo.search_similar_chunks_excluding(
        query_vector=vec,
        exclude_article_id=article_id,
        limit=limit,
        min_score=0.65,
    )
    if related_df.empty:
        return []
    return related_df.to_dict(orient="records")

@router.get("/{article_id}/thumbnail")
def get_article_thumbnail(article_id: str):
    article = repo.get_article_by_id(article_id)
    if not article or not article.get("url"):
        return RedirectResponse(f"https://picsum.photos/seed/{article_id}/800/600")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(article.get("url"), headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        og_img = soup.select_one('meta[property="og:image"]')
        img_url = og_img.get("content") if og_img else None
        
        if img_url:
            img_res = requests.get(img_url, headers=headers, timeout=5)
            return Response(content=img_res.content, media_type=img_res.headers.get("Content-Type", "image/jpeg"))
    except Exception:
        pass
    
    return RedirectResponse(f"https://picsum.photos/seed/{article_id}/800/600")