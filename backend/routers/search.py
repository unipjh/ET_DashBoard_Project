from fastapi import APIRouter
from backend.schemas import SearchRequest, SearchResult
from backend.services import repo
from backend.services.config import get_gemini_api_key
from google import genai
from google.genai import types

router = APIRouter(prefix="/api/search", tags=["search"])

_client = genai.Client(api_key=get_gemini_api_key())
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _embed_query(text: str) -> list:
    try:
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query"),
        )
        vec = list(result.embeddings[0].values)
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

    # 하이브리드 검색 (시맨틱 + BM25 RRF)
    df = repo.search_hybrid(
        query=req.query,
        query_vector=vec,
        limit=req.limit,
        min_semantic_score=0.5,
    )
    # 정규화된 RRF score만으로는 부족 — 시맨틱 또는 BM25 절대값 기준도 함께 검증
    # (다중 단어 질의는 BM25 최고점을 독점하는 기사 하나 때문에 상대 점수가
    #  낮게 나오는 경우가 많아 기준을 다소 완화함)
    df = df[
        (df["score"] >= 0.6) &
        ((df["semantic_score"] >= 0.55) | (df["bm25_score"] >= 0.5))
    ]
    if df.empty:
        return []

    return df.to_dict(orient="records")