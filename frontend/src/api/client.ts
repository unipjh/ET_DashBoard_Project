import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
})

export interface Article {
  article_id: string
  title: string
  source: string
  url: string
  published_at: string
  summary_text: string
  trust_score: number
  trust_verdict: string
}

export interface ArticleDetail extends Article {
  full_text: string
  trust_reason: string
  trust_per_criteria: string
}

export interface SearchResult {
  article_id: string
  title: string
  source: string
  published_at: string
  score: number
  chunk_text: string
}

export interface AdminStats {
  total_articles: number
  sources: string[]
  unanalyzed_count: number
}

export const fetchArticles = (page = 1, size = 10) =>
  api.get<Article[]>('/api/articles', { params: { page, size } }).then(r => r.data)

export const fetchArticle = (id: string) =>
  api.get<ArticleDetail>(`/api/articles/${id}`).then(r => r.data)

export const searchArticles = (query: string, limit = 10) =>
  api.post<SearchResult[]>('/api/search', { query, limit }).then(r => r.data)

export const fetchRelatedArticles = (id: string, limit = 5) =>
  api.get<SearchResult[]>(`/api/articles/${id}/related`, { params: { limit } }).then(r => r.data)

export const fetchStats = () =>
  api.get<AdminStats>('/api/admin/stats').then(r => r.data)

export const startCrawl = (max_articles_per_category: number) =>
  api.post('/api/admin/crawl', { max_articles_per_category }).then(r => r.data)

export const startAnalyze = () =>
  api.post<{ status: string; count: number }>('/api/admin/analyze').then(r => r.data)

export default api
