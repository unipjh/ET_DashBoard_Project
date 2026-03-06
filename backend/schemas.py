from pydantic import BaseModel
from typing import Optional


class ArticleOut(BaseModel):
    article_id: str
    title: str
    source: str
    url: str
    published_at: str
    summary_text: Optional[str] = ""
    trust_score: Optional[int] = 0
    trust_verdict: Optional[str] = ""


class ArticleDetail(ArticleOut):
    full_text: Optional[str] = ""
    trust_reason: Optional[str] = ""
    trust_per_criteria: Optional[str] = ""  # JSON string


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    article_id: str
    title: str
    source: str
    published_at: str
    score: float
    chunk_text: str


class AdminStats(BaseModel):
    total_articles: int
    sources: list[str]
    unanalyzed_count: int
