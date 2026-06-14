import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { fetchArticle, fetchRelatedArticles, postFeedback, fetchStocks } from '../api/client'
import { useEffect, useState } from 'react'
import ArticleCard from '../components/ArticleCard'
import { trackEvent } from '../api/logger'


function splitSentences(fullText: string): string[] {
  const result: string[] = []
  const lines = fullText.split('\n')
  for (let li = 0; li < lines.length; li++) {
    if (li > 0) result.push('\n')
    const line = lines[li]
    if (!line) continue
    let buf = ''
    for (let ci = 0; ci < line.length; ci++) {
      buf += line[ci]
      if ('.!?。！？'.includes(line[ci]) && line[ci + 1] === ' ' && /[가-힣A-Z]/.test(line[ci + 2] ?? '')) {
        buf += ' '
        ci++
        result.push(buf)
        buf = ''
      }
    }
    if (buf) result.push(buf)
  }
  return result
}

function buildHighlightNodes(fullText: string, summaryText: string): React.ReactNode {
  if (!summaryText || !fullText) return <>{fullText}</>

  const stopWords = new Set([
    '이', '가', '을', '를', '은', '는', '의', '에', '에서', '로', '으로',
    '과', '와', '것', '수', '등', '및', '또한', '그리고', '그러나', '하지만',
    '때문에', '통해', '대해', '위해', '따라', '관련', '이후', '이전', '현재',
    '이번', '있는', '없는', '하는', '되는', '했다', '한다', '된다', '있다', '없다',
  ])

  const summaryKeywords = new Set(
    summaryText
      .split(/[\s,，、。.!?！？\n]+/)
      .map(w => w.replace(/[^가-힣a-zA-Z0-9]/g, ''))
      .filter(w => w.length >= 3 && !stopWords.has(w))
  )

  if (summaryKeywords.size === 0) return <>{fullText}</>

  const segments = splitSentences(fullText)

  // 1단계: 각 문장의 키워드 매칭 점수 계산
  const scored = segments.map((seg, idx) => {
    if (seg === '\n' || !seg.trim()) return { idx, score: 0, density: 0 }
    const words = seg
      .split(/\s+/)
      .map(w => w.replace(/[^가-힣a-zA-Z0-9]/g, ''))
      .filter(w => w.length >= 3 && !stopWords.has(w))
    const matches = words.filter(w => summaryKeywords.has(w)).length
    const density = words.length > 0 ? matches / words.length : 0
    return { idx, score: matches, density }
  })

  // 2단계: 점수 기준 상위 최대 2문장 선택 (score >= 2 이상인 것만)
  const topTwo = new Set(
    scored
      .filter(s => s.score >= 2)
      .sort((a, b) => b.score - a.score || b.density - a.density)
      .slice(0, 2)
      .map(s => s.idx)
  )

  return (
    <>
      {segments.map((seg, idx) =>
        topTwo.has(idx) ? (
          <mark key={idx} className="bg-yellow-200 text-gray-900 font-medium rounded-sm px-0.5">
            {seg}
          </mark>
        ) : (
          seg
        )
      )}
    </>
  )
}

const CRITERIA_LABELS: Record<string, string> = {
  source_credibility: '출처 신뢰성',
  evidence_support: '근거 지지도',
  style_neutrality: '문체 중립성',
  logical_consistency: '논리 일관성',
  clickbait_risk: '어뷰징 위험도',
}

export default function DetailPage() {
  const params = useParams<{ id?: string; article_id?: string; articleId?: string }>()
  const id = params.id || params.article_id || params.articleId

  const navigate = useNavigate()
  const [isTrustModalOpen, setIsTrustModalOpen] = useState(false)
  const [isPageLoaded, setIsPageLoaded] = useState(true)
  const [feedback, setFeedback] = useState<'like' | 'dislike' | null>(null)
  const [toastMessage, setToastMessage] = useState<string | null>(null)
  const [isToastVisible, setIsToastVisible] = useState(false)

  const userId = localStorage.getItem('et_user') || 'guest'
  const isValidId = !!id && id !== 'undefined'

  const { data: article, isLoading, isError } = useQuery({
    queryKey: ['article', id],
    queryFn: () => fetchArticle(id!),
    enabled: isValidId,
  })

  const { data: related = [] } = useQuery({
    queryKey: ['related', id],
    queryFn: () => fetchRelatedArticles(id!),
    enabled: isValidId && !!article,
  })

  const feedbackMutation = useMutation({
    mutationFn: (variables: { articleId: string; feedback: 'like' | 'dislike' | null }) => {
      if (!variables.feedback) return Promise.resolve({ status: 'cancelled' })
      return postFeedback(variables.articleId, variables.feedback, userId)
    },
    onSuccess: (_data, variables) => {
      if (variables.feedback) {
        setToastMessage('피드백 주셔서 감사합니다!')
        setIsToastVisible(true)
        setTimeout(() => {
          setIsToastVisible(false)
          setTimeout(() => setToastMessage(null), 500)
        }, 3000)
      }
    },
    onError: () => {
      setToastMessage('피드백 제출에 실패했습니다.')
      setIsToastVisible(true)
      setTimeout(() => {
        setIsToastVisible(false)
        setTimeout(() => setToastMessage(null), 500)
      }, 3000)
    },
  })

  useEffect(() => {
    if (!id) return
    const storageKey = `feedbacks_${userId}`
    const userFeedbacks = JSON.parse(localStorage.getItem(storageKey) || '[]')
    const existing = userFeedbacks.find((f: any) => f.articleId === id)
    setFeedback(existing ? existing.type : null)
  }, [id, userId])

  const handleFeedback = (newFeedback: 'like' | 'dislike') => {
    if (!id) return
    const storageKey = `feedbacks_${userId}`
    let userFeedbacks = JSON.parse(localStorage.getItem(storageKey) || '[]')

    // 1. 피드백 취소 시
    if (feedback === newFeedback) {
      setFeedback(null)
      userFeedbacks = userFeedbacks.filter((f: any) => f.articleId !== id)
      localStorage.setItem(storageKey, JSON.stringify(userFeedbacks))
      feedbackMutation.mutate({ articleId: id, feedback: null })
      
      // 👈 여기에 '피드백 취소' 로그 추가
      trackEvent('cancel_feedback', id, { type: newFeedback }) 

      setToastMessage('피드백이 취소되었습니다.')
      setIsToastVisible(true)
      setTimeout(() => setIsToastVisible(false), 3000)
      return
    }

    // 2. 새로운 피드백 등록 및 변경 시
    setFeedback(newFeedback)
    userFeedbacks = userFeedbacks.filter((f: any) => f.articleId !== id)
    userFeedbacks.push({ articleId: id, title: article?.title || '알 수 없는 기사', type: newFeedback, timestamp: Date.now() })
    localStorage.setItem(storageKey, JSON.stringify(userFeedbacks))
    
    // 👈 여기에 '피드백 제출' 로그 추가
    trackEvent('click_feedback', id, { type: newFeedback })

    feedbackMutation.mutate({ articleId: id, feedback: newFeedback })
  }

  useEffect(() => {
    if (id) trackEvent('view_article_detail', id)
    window.scrollTo(0, 0)
  }, [id])

  useEffect(() => {
    if (!isLoading && article) {
      setIsPageLoaded(false)
      const timer = setTimeout(() => setIsPageLoaded(true), 50)
      return () => clearTimeout(timer)
    }
  }, [id, isLoading, article])

  const score = article?.trust_score || 0
  const radius = 88
  const circumference = 2 * Math.PI * radius

  const getHue = (s: number) => {
    if (s <= 30) return (s / 30) * 15
    if (s <= 60) return 15 + ((s - 30) / 30) * 35
    return 50 + ((s - 60) / 40) * 90
  }

  const [currentOffset, setCurrentOffset] = useState(circumference)
  const [currentScore, setCurrentScore] = useState(0)

  const { data: stocks } = useQuery({
    queryKey: ['stocks'],
    queryFn: fetchStocks,
    refetchInterval: 3 * 60 * 1000,
    staleTime: 2 * 60 * 1000,
  })

  useEffect(() => {
    if (!article || score === 0) return
    setCurrentOffset(circumference)
    setCurrentScore(0)
    const timeoutId = setTimeout(() => {
      setCurrentOffset(circumference - (score / 100) * circumference)
    }, 50)
    let startTimestamp: number | null = null
    const duration = 1000
    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp
      const progress = Math.min((timestamp - startTimestamp) / duration, 1)
      const easeOutProgress = 1 - Math.pow(1 - progress, 3)
      setCurrentScore(Math.floor(easeOutProgress * score))
      if (progress < 1) window.requestAnimationFrame(step)
    }
    const animationId = window.requestAnimationFrame(step)
    return () => {
      clearTimeout(timeoutId)
      window.cancelAnimationFrame(animationId)
    }
  }, [id, score, circumference, article])

  if (!isValidId) return <div className="p-8 text-amber-600 font-medium">잘못된 접근입니다 (기사 ID 누락). URL을 확인해주세요.</div>
  if (isLoading) return (
    <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
      <div className="flex gap-2">
        <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '-0.3s' }} />
        <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.15s' }} />
        <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" />
      </div>
    </div>
  )
  if (isError || !article) {
    return (
      <div className="p-10 text-center space-y-3 mt-10">
        <div className="text-red-500 font-bold text-xl">기사를 찾을 수 없습니다.</div>
        <p className="text-slate-600">요청한 기사 ID: <span className="font-mono text-blue-500">{id}</span></p>
        <p className="text-sm text-slate-400">백엔드 서버 API 주소가 정확한지, 백엔드가 켜져 있는지 확인해주세요.</p>
      </div>
    )
  }

  let criteriaData: Record<string, any> = {}
  try { criteriaData = JSON.parse(article.trust_per_criteria || '{}') } catch (e) {}

  const criteriaKeys = Object.keys(CRITERIA_LABELS)
  const radarCenterX = 170
  const radarCenterY = 150
  const radarMaxR = 100

  const getRadarPoint = (val: number, i: number) => {
    const angle = i * (2 * Math.PI / 5) - (Math.PI / 2)
    const r = (val / 10) * radarMaxR
    return { x: radarCenterX + r * Math.cos(angle), y: radarCenterY + r * Math.sin(angle), angle }
  }

  const polygonPoints = criteriaKeys
    .map((k, i) => {
      let s = criteriaData[k]?.score || 0
      if (k === 'clickbait_risk') s = 10 - s
      const p = getRadarPoint(s, i)
      return `${p.x},${p.y}`
    })
    .join(' ')

  let keywordList: string[] = []
  try {
    const parsed = JSON.parse(article.keywords || '[]')
    keywordList = (Array.isArray(parsed) ? parsed : []).map((kw: string) => kw.split('>').pop()?.trim() || kw)
  } catch (e) {}

  const contentLen = (article.full_text || article.summary_text || '').length
  const displayKeywords = contentLen < 500 ? keywordList.slice(0, 3) : keywordList.slice(0, 5)

  return (
    <div className="min-h-screen bg-neutral-50 font-sans antialiased text-gray-900">
      <header className="w-full px-6 py-4 flex items-center justify-start bg-neutral-50/90 backdrop-blur-md sticky top-0 z-10">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center justify-center w-12 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105"
          title="뒤로가기"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="#2563eb" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
        </button>

        {/* 주가 지수 */}
        {stocks && (
          <div className="ml-auto flex items-center gap-3 bg-blue-50 border border-blue-200 rounded-2xl px-4 py-2 shadow-sm">
            {(['KOSPI', 'KOSDAQ', 'DOW'] as const).map((name, i) => {
              const item = (stocks as any)[name]
              if (!item || item.price === null) return null
              const pct = item.change_pct ?? 0
              const isUp = pct >= 0
              const color = isUp ? 'text-red-500' : 'text-blue-500'
              const arrow = isUp ? '▲' : '▼'
              const priceStr = name === 'DOW'
                ? item.price.toLocaleString('en-US', { maximumFractionDigits: 0 })
                : item.price.toLocaleString('ko-KR', { maximumFractionDigits: 2 })
              const urls: Record<string, string> = {
                KOSPI:  'https://finance.naver.com/sise/sise_index.naver?code=KOSPI',
                KOSDAQ: 'https://finance.naver.com/sise/sise_index.naver?code=KOSDAQ',
                DOW:    'https://finance.naver.com/world/sise.naver?symbol=DJI',
              }
              return (
                <div key={name} className="flex items-center gap-3">
                  <a
                    href={urls[name]}
                    target="_blank"
                    rel="noreferrer"
                    className="flex flex-col items-end leading-tight cursor-pointer hover:opacity-70 transition-opacity"
                  >
                    <span className="text-[11px] font-bold text-blue-400 tracking-wider">{name}</span>
                    <span className="text-[13px] font-extrabold text-gray-900 tracking-tight">{priceStr}</span>
                    <span className={`text-[11px] font-bold ${color}`}>{arrow} {Math.abs(pct).toFixed(2)}%</span>
                  </a>
                  {i < 2 && <div className="w-px h-8 bg-blue-200" />}
                </div>
              )
            })}
          </div>
        )}
      </header>

      <main className={`max-w-[1500px] mx-auto px-4 py-8 transition-all duration-700 ease-out transform ${isPageLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

          {/* 좌측 패널: 핵심 키워드 → 신뢰도 점수 */}
          <div className="order-2 lg:order-1 lg:col-span-3 space-y-4 lg:sticky lg:top-24">
            {keywordList.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm flex flex-col text-center">
                <h2 className="text-[17px] font-extrabold text-gray-900 tracking-tight mb-5">핵심 키워드</h2>
                <div className="flex flex-wrap items-center justify-center gap-2.5">
                  {displayKeywords.map((kw: string, idx: number) => (
                    <span
                      key={idx}
                      onClick={() => navigate(`/?q=${encodeURIComponent(kw)}`)}
                      className="text-[14px] font-medium text-blue-600 bg-blue-50 border border-blue-100 px-3 py-1.5 rounded-md transition-all hover:scale-105 hover:bg-blue-100 hover:border-blue-300 cursor-pointer"
                    >
                      #{kw}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm flex flex-col items-center text-center">
              <h2 className="text-[17px] font-extrabold text-gray-900 tracking-tight mb-5">신뢰도 점수</h2>

              {article.trust_score > 0 ? (
                <>
                  <div className="relative flex justify-center items-center my-6">
                    <svg className="w-56 h-56 transform -rotate-90 drop-shadow-lg">
                      <defs>
                        <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor={`hsl(${getHue(score)}, 95%, 65%)`} />
                          <stop offset="100%" stopColor={`hsl(${getHue(score)}, 100%, 45%)`} />
                        </linearGradient>
                      </defs>
                      <circle cx="112" cy="112" r={radius} stroke="currentColor" strokeWidth="36" fill="transparent" className="text-slate-200" />
                      <circle
                        cx="112" cy="112" r={radius} stroke="white" strokeWidth="42" fill="transparent"
                        strokeDasharray={circumference} strokeDashoffset={currentOffset} strokeLinecap="round"
                        className="transition-all duration-1000 ease-out"
                      />
                      <circle
                        cx="112" cy="112" r={radius} stroke="url(#scoreGradient)" strokeWidth="34" fill="transparent"
                        strokeDasharray={circumference} strokeDashoffset={currentOffset} strokeLinecap="round"
                        className="transition-all duration-1000 ease-out"
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center drop-shadow-sm">
                      <span
                        className="text-[60px] font-extrabold tracking-tighter transition-colors duration-75"
                        style={{ color: `hsl(${getHue(currentScore)}, 90%, 40%)` }}
                      >
                        {currentScore}
                      </span>
                      <span className="text-[12px] font-bold text-gray-400 uppercase tracking-widest mt-0.5">Score</span>
                    </div>
                  </div>

                  <p className="mt-6 text-[15px] text-gray-700 bg-gray-50 p-5 rounded-xl leading-relaxed text-left w-full border border-gray-200 font-medium break-keep tracking-tight">
                    <span className="font-extrabold text-gray-900 block mb-2 text-[15px]">AI 종합 평가</span>
                    {article.trust_reason || '분석된 이유가 없습니다.'}
                  </p>

                  <button
                    onClick={() => {
                      trackEvent('open_trust_modal', id)
                      setIsTrustModalOpen(true)
                    }}
                    className="mt-5 text-[14px] text-blue-600 font-bold hover:text-blue-700 hover:underline tracking-tight transition-colors"
                  >
                    상세 항목 보기
                  </button>
                </>
              ) : (
                <p className="text-[15px] font-bold text-gray-400 text-center py-6 tracking-tight">분석된 신뢰도 데이터가 없습니다.</p>
              )}
            </div>
          </div>

          {/* 중앙 패널: AI 요약 + 원문 */}
          <div className="order-1 lg:order-2 lg:col-span-6 bg-white rounded-xl border border-gray-200 p-6 sm:p-10 shadow-sm">
            <div className="mb-8 border-b border-gray-100 pb-8">
              <h1 className="text-3xl sm:text-[32px] font-extrabold text-gray-900 tracking-tighter leading-[1.3] mb-5 break-keep">
                {article.title}
              </h1>
              <div className="flex flex-wrap items-center justify-between gap-4 text-[14px] font-bold text-gray-500 tracking-tight">
                <div className="flex items-center gap-2.5">
                  <span className="text-blue-600">{article.source}</span>
                  <span>·</span>
                  <span>{article.published_at}</span>
                  <span>·</span>
                  <a href={article.url} target="_blank" rel="noreferrer" className="text-gray-400 hover:text-blue-600 hover:underline transition-colors">
                    원문 보기
                  </a>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 mr-2">이 분석이 유용했나요?</span>
                  <button
                    onClick={() => handleFeedback('like')}
                    title="유용해요"
                    className={`flex items-center justify-center w-10 h-10 rounded-full transition-all duration-200 transform ${
                      feedback === 'like'
                        ? 'bg-green-100 text-green-600 ring-2 ring-green-500 scale-110'
                        : 'bg-white text-gray-400 hover:text-green-500 hover:bg-green-50 border border-gray-200 hover:scale-110'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                      <path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/>
                    </svg>
                  </button>
                  <button
                    onClick={() => handleFeedback('dislike')}
                    title="개선이 필요해요"
                    className={`flex items-center justify-center w-10 h-10 rounded-full transition-all duration-200 transform ${
                      feedback === 'dislike'
                        ? 'bg-red-100 text-red-600 ring-2 ring-red-500 scale-110'
                        : 'bg-white text-gray-400 hover:text-red-500 hover:bg-red-50 border border-gray-200 hover:scale-110'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                      <path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>

            {article.summary_text && (
              <div className="bg-[#f4f7fb] rounded-2xl p-7 mb-10 border border-blue-100/50 shadow-sm">
                <h3 className="text-blue-900 text-[17px] font-extrabold mb-4 flex items-center gap-2 tracking-tight">
                  <span>✨</span> AI 요약
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(window.location.href)
                      setToastMessage('링크가 복사되었습니다.')
                      setIsToastVisible(true)
                      setTimeout(() => { setIsToastVisible(false); setTimeout(() => setToastMessage(null), 500) }, 3000)
                    }}
                    title="링크 복사"
                    className="flex items-center justify-center w-7 h-7 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-md transition-all duration-200 hover:scale-105 ml-1"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="#2563eb">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244" />
                    </svg>
                  </button>
                </h3>
                <p className="text-gray-800 text-[16px] leading-relaxed font-semibold break-keep tracking-tight">
                  {article.summary_text}
                </p>
              </div>
            )}

            <div className="text-gray-800 text-[17px] leading-[1.8] font-medium whitespace-pre-wrap break-keep tracking-[-0.01em]">
              {article.full_text
                ? buildHighlightNodes(article.full_text, article.summary_text || '')
                : <p className="text-gray-400 font-bold text-center py-10">본문 내용이 존재하지 않습니다.</p>}
            </div>
          </div>

          {/* 우측 패널: 관련 기사 */}
          <div className="order-3 lg:col-span-3 space-y-4 lg:sticky lg:top-24">
            {related.length > 0 ? (
              <div className="space-y-3">
                {related.map(r => (
                  <div
                    key={r.article_id}
                    className="transition-transform duration-300 ease-out hover:-translate-y-1"
                    onClickCapture={() => trackEvent('click_related', r.article_id, { source_article: id })}
                  >
                    <ArticleCard article={r} isRelated={true} />
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-gray-50 rounded-xl p-8 text-center border border-dashed border-gray-200">
                <p className="text-[15px] font-bold text-gray-500 tracking-tight">추천할 관련 기사가 없습니다.</p>
              </div>
            )}
          </div>

        </div>
      </main>

      {/* 신뢰도 상세 모달 */}
      {isTrustModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
          onClick={() => setIsTrustModalOpen(false)}
        >
          <div
            className="bg-white rounded-2xl p-6 w-full max-w-lg max-h-[85vh] overflow-y-auto shadow-xl"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6 border-b border-slate-100 pb-4">
              <h3 className="text-[19px] font-extrabold text-gray-900 tracking-tight">신뢰도 상세 항목</h3>
              <button
                onClick={() => setIsTrustModalOpen(false)}
                className="text-gray-400 hover:text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-full w-8 h-8 flex items-center justify-center transition-colors font-bold"
              >
                ✕
              </button>
            </div>
            <div className="flex items-center justify-center p-4">
              <svg width="340" height="320" className="overflow-visible">
                {[2, 4, 6, 8, 10].map(level => {
                  const pts = criteriaKeys.map((_, i) => {
                    const p = getRadarPoint(level, i)
                    return `${p.x},${p.y}`
                  }).join(' ')
                  return (
                    <polygon key={level} points={pts} fill="none" stroke="#e2e8f0" strokeWidth="1" strokeDasharray={level === 10 ? "" : "3,3"} />
                  )
                })}
                {criteriaKeys.map((_, i) => {
                  const p = getRadarPoint(10, i)
                  return <line key={i} x1={radarCenterX} y1={radarCenterY} x2={p.x} y2={p.y} stroke="#e2e8f0" strokeWidth="1" />
                })}
                <polygon points={polygonPoints} fill="rgba(59, 130, 246, 0.25)" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round" />
                {criteriaKeys.map((key, i) => {
                  const score = criteriaData[key]?.score || 0
                  const plotScore = key === 'clickbait_risk' ? 10 - score : score
                  const p = getRadarPoint(plotScore, i)
                  const labelP = getRadarPoint(11.5, i)

                  let anchor: "inherit" | "middle" | "start" | "end" = "middle"
                  if (Math.cos(labelP.angle) > 0.1) anchor = "start"
                  if (Math.cos(labelP.angle) < -0.1) anchor = "end"

                  let dy = 0
                  if (Math.sin(labelP.angle) > 0.1) dy = 10
                  if (Math.sin(labelP.angle) < -0.1) dy = -10

                  return (
                    <g key={key}>
                      <circle cx={p.x} cy={p.y} r="4" fill="#3b82f6" />
                      <text x={labelP.x} y={labelP.y + dy} textAnchor={anchor} dominantBaseline="middle" className="text-[13px] font-extrabold fill-gray-800 tracking-tight">
                        {CRITERIA_LABELS[key]}
                      </text>
                      <text x={labelP.x} y={labelP.y + dy + 16} textAnchor={anchor} dominantBaseline="middle">
                        <tspan className="text-sm font-semibold fill-blue-600">{score}점</tspan>
                      </text>
                    </g>
                  )
                })}
              </svg>
            </div>

            <div className="mt-2 space-y-3 px-2 pb-2">
              {Object.entries(CRITERIA_LABELS).map(([key, label]) => {
                const item = criteriaData[key]
                if (!item) return null
                const itemScore = item.score || 0
                const isClickbait = key === 'clickbait_risk'
                return (
                  <div key={key} className="bg-slate-50/80 rounded-xl p-4 border border-blue-100/50 hover:border-blue-200 transition-colors shadow-sm">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-extrabold text-gray-800 text-[15px] tracking-tight flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                        {label}
                      </span>
                      <div className="flex items-center gap-2">
                        {isClickbait && <span className="text-[11px] text-gray-500 font-bold bg-gray-200/70 px-2 py-0.5 rounded tracking-tight">낮을수록 좋음</span>}
                        <span className="font-extrabold text-blue-700 bg-blue-50 border border-blue-100 px-2.5 py-0.5 rounded-md text-[13px]">{itemScore} / 10</span>
                      </div>
                    </div>
                    <p className="text-[14px] text-gray-700 leading-relaxed font-medium break-keep">{(item.reason || '').replace(/\//g, '')}</p>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* 토스트 메시지 */}
      {toastMessage && (
        <div className={`fixed bottom-10 left-1/2 -translate-x-1/2 bg-gray-900/90 backdrop-blur-sm text-white px-6 py-3 rounded-full shadow-lg z-50 text-sm font-semibold transition-all duration-500 ease-out ${isToastVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-5 pointer-events-none'}`}>
          {toastMessage}
        </div>
      )}
    </div>
  )
}
