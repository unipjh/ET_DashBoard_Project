import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchArticle } from '../api/client'
import TrustGauge from '../components/TrustGauge'

export default function DetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: article, isLoading, isError } = useQuery({
    queryKey: ['article', id],
    queryFn: () => fetchArticle(id!),
    enabled: !!id,
  })

  if (isLoading) return <div className="p-8 text-slate-400">불러오는 중...</div>
  if (isError || !article) return <div className="p-8 text-red-500">기사를 찾을 수 없습니다.</div>

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <button onClick={() => navigate(-1)} className="text-sm text-slate-500 hover:text-slate-800">
          ← 뒤로
        </button>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        <div>
          <h1 className="text-xl font-bold text-slate-900 leading-snug">{article.title}</h1>
          <div className="flex gap-2 text-sm text-slate-400 mt-2">
            <span>{article.source}</span>
            <span>·</span>
            <span>{article.published_at}</span>
            <span>·</span>
            <a href={article.url} target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">
              원문 보기
            </a>
          </div>
        </div>

        {article.summary_text && (
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-sm text-slate-700">
            <p className="font-semibold text-blue-700 mb-1 text-xs">AI 요약</p>
            {article.summary_text}
          </div>
        )}

        {article.trust_score > 0 && (
          <TrustGauge
            score={article.trust_score}
            verdict={article.trust_verdict}
            reason={article.trust_reason}
            perCriteriaJson={article.trust_per_criteria}
          />
        )}

        <div className="bg-white rounded-xl border border-slate-100 p-5">
          <p className="font-semibold text-slate-700 text-sm mb-3">기사 본문</p>
          <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{article.full_text}</p>
        </div>
      </main>
    </div>
  )
}
