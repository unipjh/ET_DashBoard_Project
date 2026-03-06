from fastapi import APIRouter, HTTPException, Query
from backend.services import repo
from backend.schemas import ArticleOut, ArticleDetail

router = APIRouter(prefix="/api/articles", tags=["articles"])


@router.get("", response_model=list[ArticleOut])
def get_articles(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100)):
    df = repo.load_articles()
    if df.empty:
        return []
    start = (page - 1) * size
    end = start + size
    rows = df.iloc[start:end]
    return rows.to_dict(orient="records")


@router.get("/{article_id}", response_model=ArticleDetail)
def get_article(article_id: str):
    df = repo.load_articles()
    row = df[df["article_id"] == article_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    return row.iloc[0].to_dict()
