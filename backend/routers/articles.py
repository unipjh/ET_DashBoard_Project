from fastapi import APIRouter, HTTPException, Query
from backend.services import repo
from backend.schemas import ArticleOut, ArticleDetail, SearchResult
from backend.services.config import get_gemini_api_key
import google.generativeai as genai

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


@router.get("", response_model=list[ArticleOut])
def get_articles(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100)):
    df = repo.load_articles()
    if df.empty:
        return []
    start = (page - 1) * size
    end = start + size
    rows = df.iloc[start:end]
    return rows.to_dict(orient="records")


@router.get("/{article_id}/related", response_model=list[SearchResult])
def get_related(article_id: str, limit: int = Query(5, ge=1, le=20)):
    df = repo.load_articles()
    row = df[df["article_id"] == article_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    r = row.iloc[0]
    # 제목 + 요약으로 쿼리 벡터 생성 (embed_summary는 "[]" placeholder이므로 재임베딩)
    query_text = f"{r.get('title', '')} {r.get('summary_text', '')}".strip()
    vec = _embed(query_text)
    related_df = repo.search_similar_chunks_excluding(
        query_vector=vec,
        exclude_article_id=article_id,
        limit=limit,
        min_score=0.5,
    )
    if related_df.empty:
        return []
    return [
        SearchResult(
            article_id=str(row["article_id"]),
            title=str(row["title"]),
            source=str(row["source"]),
            published_at=str(row["published_at"]),
            score=float(row["score"]),
            chunk_text=str(row["chunk_text"]),
        )
        for _, row in related_df.iterrows()
    ]


@router.get("/{article_id}", response_model=ArticleDetail)
def get_article(article_id: str):
    df = repo.load_articles()
    row = df[df["article_id"] == article_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    return row.iloc[0].to_dict()
