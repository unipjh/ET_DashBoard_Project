import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 10000,
})

export const getApiAssetUrl = (path: string) =>
  `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`

export interface Article {
  article_id: string
  title: string
  source: string
  url: string
  published_at: string
  summary_text: string
  chunk_text?: string
  keywords?: string
  trust_score: number
  trust_verdict: string
  category: string
  rec_source?: string | null
  rec_score?: number | null
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
  today_articles?: number
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

export const fetchRecommendations = (sessionId?: string | null, userId?: string | null, limit = 10) =>
  api.get<Article[]>('/api/recommendations', {
    params: { session_id: sessionId || undefined, user_id: userId || undefined, limit },
  }).then(r => r.data)

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

export const postFeedback = (articleId: string, feedbackType: 'like' | 'dislike', userId: string = 'guest') =>
  api.post('/api/feedback', { article_id: articleId, feedback_type: feedbackType, user_id: userId }).then(r => r.data)

export interface StockItem {
  price: number | null
  change_pct: number | null
  error?: string
}
export type StocksData = Record<string, StockItem>

export const fetchStocks = () =>
  api.get<StocksData>('/api/stocks').then(r => r.data)

export const signup = (email: string, password: string) =>
  api.post<{ ok: boolean; user_id: string }>('/api/auth/signup', { email, password }).then(r => r.data)

export const login = (email: string, password: string) =>
  api.post<{ ok: boolean; user_id: string }>('/api/auth/login', { email, password }).then(r => r.data)

export const fetchUserInfo = (userId: string) =>
  api.get<{ user_id: string; created_at: string }>('/api/auth/me', { params: { user_id: userId } }).then(r => r.data)

export const deleteAccount = (userId: string, password: string) =>
  api.delete<{ ok: boolean }>('/api/auth/me', { data: { user_id: userId, password } }).then(r => r.data)

export const adminHeaders = (password: string) => ({
  'X-Admin-Password': password,
})

export const validateAdmin = (password: string) =>
  api.post<{ status: string }>('/api/admin/session', null, {
    headers: adminHeaders(password),
  }).then(r => r.data)

export const fetchAdminStats = (password: string) =>
  api.get<AdminStats>('/api/admin/stats', {
    headers: adminHeaders(password),
  }).then(r => r.data)

export default api
