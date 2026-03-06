import { useNavigate } from 'react-router-dom'
import type { Article } from '../api/client'

const verdictStyle: Record<string, string> = {
  likely_true: 'bg-green-100 text-green-800',
  uncertain: 'bg-yellow-100 text-yellow-800',
  likely_false: 'bg-red-100 text-red-800',
}

const verdictLabel: Record<string, string> = {
  likely_true: '신뢰',
  uncertain: '불확실',
  likely_false: '비신뢰',
}

export default function ArticleCard({ article }: { article: Article }) {
  const navigate = useNavigate()
  const verdict = article.trust_verdict || 'uncertain'

  return (
    <div
      onClick={() => navigate(`/article/${article.article_id}`)}
      className="bg-white rounded-xl shadow-sm border border-slate-100 p-5 cursor-pointer hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold text-slate-900 leading-snug line-clamp-2 flex-1">
          {article.title}
        </h3>
        {article.trust_score > 0 && (
          <span className={`text-xs font-medium px-2 py-1 rounded-full whitespace-nowrap ${verdictStyle[verdict] ?? verdictStyle.uncertain}`}>
            {verdictLabel[verdict] ?? verdict} {article.trust_score}점
          </span>
        )}
      </div>
      <p className="mt-2 text-sm text-slate-500 line-clamp-2">{article.summary_text}</p>
      <div className="mt-3 flex gap-2 text-xs text-slate-400">
        <span>{article.source}</span>
        <span>·</span>
        <span>{article.published_at}</span>
      </div>
    </div>
  )
}
