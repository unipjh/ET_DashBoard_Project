import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchArticles, searchArticles } from '../api/client'
import ArticleCard from '../components/ArticleCard'

export default function MainPage() {
  const [query, setQuery] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [page, setPage] = useState(1)
  const navigate = useNavigate()

  const { data: articles = [], isLoading } = useQuery({
    queryKey: ['articles', page],
    queryFn: () => fetchArticles(page, 10),
  })

  const { data: searchResults, isLoading: isSearching } = useQuery({
    queryKey: ['search', submittedQuery],
    queryFn: () => searchArticles(submittedQuery),
    enabled: submittedQuery.length > 0,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) setSubmittedQuery(query.trim())
  }

  const clearSearch = () => {
    setSubmittedQuery('')
    setQuery('')
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-900">ET · 뉴스 신뢰도 분석</h1>
        <button
          onClick={() => navigate('/admin')}
          className="text-sm text-slate-500 hover:text-slate-800 border border-slate-200 px-3 py-1.5 rounded-lg"
        >
          Admin
        </button>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="궁금한 키워드를 입력하세요"
            className="flex-1 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700"
          >
            검색
          </button>
        </form>

        {submittedQuery && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-700">
                "{submittedQuery}" 검색 결과
              </h2>
              <button onClick={clearSearch} className="text-xs text-slate-400 hover:text-slate-600">
                닫기
              </button>
            </div>
            {isSearching ? (
              <p className="text-sm text-slate-400">검색 중...</p>
            ) : searchResults?.length === 0 ? (
              <p className="text-sm text-slate-400">결과가 없습니다.</p>
            ) : (
              <div className="space-y-3">
                {searchResults?.map(r => (
                  <div
                    key={r.article_id}
                    onClick={() => navigate(`/article/${r.article_id}`)}
                    className="bg-white rounded-xl border border-slate-100 p-4 cursor-pointer hover:shadow-md transition-shadow"
                  >
                    <h3 className="font-semibold text-slate-900 text-sm">{r.title}</h3>
                    <p className="text-xs text-slate-500 mt-1 line-clamp-2">{r.chunk_text}</p>
                    <div className="flex gap-2 text-xs text-slate-400 mt-2">
                      <span>{r.source}</span>
                      <span>·</span>
                      <span>유사도 {Math.round(r.score * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div>
          <h2 className="text-sm font-semibold text-slate-700 mb-3">최신 기사</h2>
          {isLoading ? (
            <p className="text-sm text-slate-400">불러오는 중...</p>
          ) : (
            <div className="space-y-3">
              {articles.map(a => <ArticleCard key={a.article_id} article={a} />)}
            </div>
          )}
        </div>

        <div className="flex justify-center gap-3">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="text-sm px-4 py-2 border border-slate-200 rounded-lg disabled:opacity-40 hover:bg-slate-100"
          >
            이전
          </button>
          <span className="text-sm text-slate-500 flex items-center px-2">{page}페이지</span>
          <button
            onClick={() => setPage(p => p + 1)}
            disabled={articles.length < 10}
            className="text-sm px-4 py-2 border border-slate-200 rounded-lg disabled:opacity-40 hover:bg-slate-100"
          >
            다음
          </button>
        </div>
      </main>
    </div>
  )
}
