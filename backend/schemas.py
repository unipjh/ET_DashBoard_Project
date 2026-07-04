from pydantic import BaseModel
from typing import Optional, List


class CategoryStat(BaseModel):
    category: str
    total: int
    unanalyzed: int
    today_articles: int = 0


class ApiUsage(BaseModel):
    last_success_time: str
    error_count: int
    quota_percent: float


class ArticleOut(BaseModel):
    article_id: str
    title: str
    source: str
    url: str
    published_at: str
    summary_text: Optional[str] = ""
    keywords: Optional[str] = "[]"
    trust_score: Optional[int] = 0
    trust_verdict: Optional[str] = ""
    category: Optional[str] = ""


class PaginatedArticlesResponse(BaseModel):
    articles: List[ArticleOut]
    total_count: int


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
    summary_text: Optional[str] = ""
    keywords: Optional[str] = "[]"
    trust_score: Optional[int] = 0
    trust_verdict: Optional[str] = ""


class AdminStats(BaseModel):
    total_articles: int
    sources: list[str]
    unanalyzed_count: int
    category_stats: Optional[List[CategoryStat]] = []
    api_usage: Optional[ApiUsage] = None
