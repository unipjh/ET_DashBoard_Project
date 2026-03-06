from fastapi import APIRouter
from backend.schemas import SearchRequest, SearchResult
from backend.services import repo
from backend.services.config import get_gemini_api_key
import google.generativeai as genai

router = APIRouter(prefix="/api/search", tags=["search"])

genai.configure(api_key=get_gemini_api_key())
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _embed_query(text: str) -> list:
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        vec = result["embedding"]
        if len(vec) > 768:
            vec = vec[:768]
        elif len(vec) < 768:
            vec = vec + [0.0] * (768 - len(vec))
        return vec
    except Exception:
        return [0.0] * 768


@router.post("", response_model=list[SearchResult])
def search_articles(req: SearchRequest):
    vec = _embed_query(req.query)
    df = repo.search_similar_chunks(query_vector=vec, limit=req.limit, min_score=0.5)
    if df.empty:
        return []
    results = []
    for _, row in df.iterrows():
        results.append(SearchResult(
            article_id=str(row["article_id"]),
            title=str(row["title"]),
            source=str(row["source"]),
            published_at=str(row["published_at"]),
            score=float(row["score"]),
            chunk_text=str(row["chunk_text"]),
        ))
    return results
