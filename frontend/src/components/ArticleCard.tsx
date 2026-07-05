import { useNavigate } from 'react-router-dom'
import type { Article, SearchResult } from '../api/client'

const getHue = (s: number) => {
  if (s <= 30) return (s / 30) * 15
  if (s <= 60) return 15 + ((s - 30) / 30) * 35
  return 50 + ((s - 60) / 40) * 90
}

export default function ArticleCard({ article, isRelated = false }: { article: Article | SearchResult; isRelated?: boolean }) {
  const navigate = useNavigate()
  const trustScore = 'trust_score' in article ? article.trust_score : 0
  const summary = 'summary_text' in article ? article.summary_text : ''

  return (
    <div
      onClick={() => navigate(`/article/${article.article_id}`)}
      className="flex flex-col bg-white rounded-lg border border-gray-200 p-5 cursor-pointer hover:border-navy-300 transition-colors duration-150"
    >
      <div className="flex items-center gap-2 mb-2 text-[12px] font-medium text-gray-500 tracking-tight">
        <span className="font-semibold text-gray-800">{article.source}</span>
        {article.published_at && (
          <>
            <span className="text-gray-300">|</span>
            <span>{article.published_at.substring(0, 10)}</span>
          </>
        )}
      </div>

      <div className="flex items-start justify-between gap-3">
        <h3 className="text-[16px] font-bold text-gray-900 tracking-tight leading-snug line-clamp-2 flex-1 break-keep">
          {article.title}
        </h3>
        {!isRelated && trustScore > 0 && (
          <span
            className="text-xs font-bold whitespace-nowrap shrink-0 pt-0.5"
            style={{ color: `hsl(${getHue(trustScore)}, 70%, 38%)` }}
          >
            신뢰도 {trustScore}점
          </span>
        )}
      </div>

      <p className="mt-2 text-[13.5px] font-medium text-gray-500 line-clamp-1 leading-relaxed tracking-tight">
        {summary || article.chunk_text || '요약 데이터가 없습니다.'}
      </p>
    </div>
  )
}
