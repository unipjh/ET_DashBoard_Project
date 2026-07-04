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

  let keywordList: string[] = []
  try {
    const parsed: unknown = JSON.parse('keywords' in article ? article.keywords || '[]' : '[]')
    keywordList = (Array.isArray(parsed) ? parsed : [])
      .filter((kw): kw is string => typeof kw === 'string')
      .map((kw) => kw.split('>').pop()?.trim() || kw)
  } catch {
    keywordList = []
  }

  const displayKeywords = keywordList.slice(0, 2)

  return (
    <div
      onClick={() => navigate(`/article/${article.article_id}`)}
      className="flex flex-col bg-white rounded-xl shadow-sm border border-gray-200 p-5 cursor-pointer hover:shadow-md hover:border-gray-300 hover:-translate-y-0.5 transition-all duration-200"
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
            className="text-xs font-extrabold px-2 py-1 rounded-md whitespace-nowrap border shadow-sm shrink-0"
            style={{
              color: `hsl(${getHue(trustScore)}, 85%, 35%)`,
              backgroundColor: `hsl(${getHue(trustScore)}, 100%, 94%)`,
              borderColor: `hsl(${getHue(trustScore)}, 85%, 85%)`
            }}
          >
            신뢰도 {trustScore}점
          </span>
        )}
      </div>

      <p className="mt-2 text-[13.5px] font-medium text-gray-500 line-clamp-1 leading-relaxed tracking-tight">
        {summary || article.chunk_text || '요약 데이터가 없습니다.'}
      </p>

      {!isRelated && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {displayKeywords.length > 0 ? (
            displayKeywords.map((kw, idx) => (
              <span key={idx} className="text-xs font-medium text-blue-600 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded-md">
                #{kw}
              </span>
            ))
          ) : (
            <span className="text-xs font-medium text-gray-400 bg-gray-100 px-2 py-0.5 rounded-md">
              #키워드없음
            </span>
          )}
        </div>
      )}
    </div>
  )
}
