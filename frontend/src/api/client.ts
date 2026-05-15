import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
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
  category: string
}

export interface PaginatedArticlesResponse {
  articles: Article[];
  total_count: number;
}

export interface ArticleDetail extends Article {
  full_text: string
  trust_reason: string
  trust_per_criteria: string
  keywords?: string
}

export interface SearchResult {
  article_id: string
  title: string
  source: string
  published_at: string
  score: number
  chunk_text: string
}

export interface CategoryStat {
  category: string
  total: number
  unanalyzed: number
}

export interface ApiUsage {
  last_success_time: string
  error_count: number
  quota_percent: number
}

export interface AdminStats {
  total_articles: number
  sources: string[]
  unanalyzed_count: number
  category_stats: CategoryStat[]
  api_usage: ApiUsage
}

export const fetchArticles = (page = 1, size = 10, category?: string) => {
  const params: { page: number; size: number; category?: string } = { page, size };
  if (category) { params.category = category; }
  return api.get<PaginatedArticlesResponse>('/api/articles', { params }).then(r => r.data);
}

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

export const startAnalyze = () => api.post<{ status: string; count: number }>('/api/admin/analyze').then(r => r.data)

export const startDedupe = () => api.post('/api/admin/dedupe').then(r => r.data)

export const startKeywords = () => api.post('/api/admin/keywords').then(r => r.data)

export interface FeedbackLog {
  feedback_id: string;
  article_id: string;
  article_title: string;
  feedback_type: 'like' | 'dislike';
  created_at: string;
}

export const fetchFeedback = () => api.get<FeedbackLog[]>('/api/feedback').then(r => r.data)

export const postFeedback = (articleId: string, feedbackType: 'like' | 'dislike') =>
  api.post('/api/feedback', { article_id: articleId, feedback_type: feedbackType }).then(r => r.data)

export default api
