import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchStats } from '../api/client'
import type { AdminStats } from '../api/client'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function AdminPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  useEffect(() => {
    if (localStorage.getItem('et_user') !== 'etdashboard@naver.com') {
      navigate('/', { replace: true })
    }
  }, [])

  const [crawlMax, setCrawlMax] = useState(0)
  const ALL_CATEGORIES = ["IT/과학", "경제", "사회", "생활/문화", "세계", "정치", "연예", "스포츠"]

  const hasStartedRef = useRef(false)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const lastStartRef = useRef<number>(0)
  const [logs, setLogs] = useState<{ step: string; message: string; time: string }[]>([])

  const [crawlState, setCrawlState] = useState<{status: 'idle' | 'crawling' | 'success', prevCount: number | null, newCount?: number}>(() => {
    const stored = localStorage.getItem('crawlingState')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        if (parsed && typeof parsed === 'object' && 'status' in parsed && parsed.status === 'crawling') return parsed
      } catch(e) {}
    }
    return { status: 'idle', prevCount: null }
  })

  const { data: stats, isLoading: isLoadingStats } = useQuery<AdminStats>({
    queryKey: ['adminStats'],
    queryFn: fetchStats,
    refetchInterval: 5000,
  })

  const { data: processStatus } = useQuery({
    queryKey: ['processStatus'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/admin/process-status`)
      if (!res.ok) return null
      return res.json()
    },
    refetchInterval: crawlState.status === 'crawling' ? 200 : false,
    refetchOnWindowFocus: false,
    staleTime: 0,
  })

  const crawlMutation = useMutation({
    mutationFn: async (payload: { max: number, categories: string[] }) => {
      const perCat = Math.max(1, Math.ceil(payload.max / payload.categories.length))
      const res = await fetch(`${API_BASE}/api/admin/crawl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_articles_per_category: perCat, categories: payload.categories, total_articles: payload.max })
      })
      if (!res.ok) throw new Error('크롤링 서버 에러')
      return res.json()
    },
    onSuccess: () => {
      hasStartedRef.current = false
      lastStartRef.current = Date.now()
      setCrawlState({ status: 'crawling', prevCount: stats?.total_articles ?? 0 })
      queryClient.invalidateQueries({ queryKey: ['adminStats'] })
      queryClient.invalidateQueries({ queryKey: ['processStatus'] })
      setLogs([{
        step: 'System',
        message: '크롤링 작업을 서버에 요청했습니다. 프로세스 응답 대기 중...',
        time: new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
      }])
    },
    onError: (error: any) => {
      alert(`크롤링 작업 시작에 실패했습니다: ${error.message || '서버 에러'}`)
      setCrawlState({ status: 'idle', prevCount: null })
    },
  })

  useEffect(() => {
    localStorage.setItem('crawlingState', JSON.stringify(crawlState))
  }, [crawlState])

  useEffect(() => {
    if (crawlState.status === 'crawling') {
      if (processStatus?.process_name === 'crawl') hasStartedRef.current = true

      if (processStatus?.process_name === 'idle') {
        if (hasStartedRef.current) {
          setCrawlState(prev => ({ ...prev, status: 'success' }))
          hasStartedRef.current = false

          fetchStats().then(latestStats => {
            const newCount = (typeof latestStats?.total_articles === 'number' && typeof crawlState.prevCount === 'number')
              ? Math.max(0, latestStats.total_articles - crawlState.prevCount)
              : 0
            setCrawlState(prev => ({ ...prev, newCount, prevCount: latestStats?.total_articles ?? null }))
            queryClient.invalidateQueries({ queryKey: ['adminStats'] })
            setLogs(prev => {
              const finalMsg = newCount > 0
                ? `✅ 작업 완료! 총 ${newCount}개의 기사를 수집했습니다.`
                : `❌ 추가된 기사가 없습니다 (모두 기존 DB와 중복됨)`
              if (prev.length > 0 && (prev[prev.length - 1].message.includes('작업 완료') || prev[prev.length - 1].message.includes('기사가 없습니다'))) return prev
              return [...prev, { step: '완료', message: finalMsg, time: new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) }]
            })
          })
        } else if (Date.now() - lastStartRef.current > 3000) {
          setCrawlState({ status: 'idle', prevCount: null })
        }
      }
    }
  }, [processStatus?.process_name, crawlState.status, crawlState.prevCount, queryClient])

  useEffect(() => {
    if (processStatus && crawlState.status === 'crawling') {
      if (processStatus.last_message) {
        setLogs(prev => {
          if (prev.length > 0 && prev[prev.length - 1].message === processStatus.last_message) return prev
          return [...prev, {
            step: processStatus.current_step || '시스템',
            message: processStatus.last_message,
            time: new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
          }]
        })
      }
    }
  }, [processStatus?.last_message, processStatus?.current_step, crawlState.status])

  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <div className="min-h-screen bg-neutral-50 font-sans antialiased text-gray-900">
      <header className="w-full px-6 py-4 flex items-center justify-start bg-neutral-50/90 backdrop-blur-md sticky top-0 z-10">
        <button
          onClick={() => navigate('/')}
          className="flex items-center justify-center w-12 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105"
          title="메인으로 돌아가기"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="#2563eb" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
        </button>
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-12 pt-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">

          {/* DB 현황 */}
          <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm flex flex-col">
            <h2 className="text-[16px] font-extrabold text-gray-800 mb-5 tracking-tight">DB 현황</h2>
            {isLoadingStats ? (
              <div className="text-center py-10 text-gray-400">로딩 중...</div>
            ) : stats ? (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  { label: '총 기사 수', value: stats.total_articles },
                  { label: '미분석 기사', value: stats.unanalyzed_count, color: 'text-red-500' },
                  { label: '수집된 언론사', value: stats.sources?.length || 0 }
                ].map(item => (
                  <div key={item.label} className="bg-gray-50/80 p-4 rounded-xl border border-gray-200/80">
                    <p className="text-xs font-semibold text-gray-500 mb-1 tracking-wide">{item.label}</p>
                    <p className={`text-3xl font-bold mt-1 ${item.color || 'text-gray-900'}`}>{item.value}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 text-red-500 text-[14px] font-medium">통계 정보를 불러오지 못했습니다.</div>
            )}
          </section>

          {/* API 사용량 */}
          <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm flex flex-col">
            <h2 className="text-[16px] font-extrabold text-gray-800 mb-5 tracking-tight">API 사용량 및 상태 모니터링</h2>
            {isLoadingStats ? (
              <div className="text-center py-10 flex-grow flex items-center justify-center text-gray-400">로딩 중...</div>
            ) : stats?.api_usage ? (
              <div className="flex-grow flex flex-col justify-center space-y-3">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-medium text-gray-500">Gemini API 할당량</p>
                  <p className="text-lg font-bold text-blue-600">{stats.api_usage.quota_percent}%</p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${stats.api_usage.quota_percent}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-400 mt-1">마지막 성공: {stats.api_usage.last_success_time}</p>
              </div>
            ) : (
              <div className="text-center py-10 text-red-500 flex-grow flex items-center justify-center text-[14px] font-medium">API 정보를 불러오지 못했습니다.</div>
            )}
          </section>

          {/* 카테고리별 분석 상태 */}
          <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm flex flex-col">
            <div className="flex justify-between items-center mb-5">
              <h2 className="text-[16px] font-extrabold text-gray-800 tracking-tight">카테고리별 분석 상태</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="border-b border-gray-100">
                    <th scope="col" className="px-4 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50/80">카테고리</th>
                    <th scope="col" className="px-4 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50/80">총 기사</th>
                    <th scope="col" className="px-4 py-3 text-xs font-bold text-blue-600 uppercase tracking-wider bg-gray-50/80">오늘의 기사</th>
                    <th scope="col" className="px-4 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50/80">분석률</th>
                  </tr>
                </thead>
                <tbody>
                  {stats?.category_stats?.map(cat => {
                    const analyzedRate = cat.total > 0 ? ((cat.total - cat.unanalyzed) / cat.total) * 100 : 0
                    return (
                      <tr key={cat.category} className="border-b border-gray-50 last:border-b-0 hover:bg-gray-50/50 transition-colors">
                        <th scope="row" className="px-4 py-3.5 font-bold text-gray-900 whitespace-nowrap text-[13px]">{cat.category}</th>
                        <td className="px-4 py-3.5 text-gray-500 text-[13px]">{cat.total}</td>
                        <td className="px-4 py-3.5 font-bold text-blue-600 text-[13px]">{(cat as any).today_articles || 0}</td>
                        <td className="px-4 py-3.5">
                          <div className="flex items-center gap-2">
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div className="bg-green-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${analyzedRate}%` }}></div>
                            </div>
                            <span className="font-semibold text-gray-700 text-[12px] w-8 text-right">{Math.round(analyzedRate)}%</span>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* 자동 크롤링 설정 */}
          <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm flex flex-col h-full">
            <h2 className="text-[16px] font-extrabold text-gray-800 mb-5 tracking-tight">자동 크롤링 설정</h2>

            <div className="bg-gray-50/70 p-4 rounded-xl border border-gray-200/80 space-y-3 w-full mb-4">
              <h3 className="font-bold text-[15px] text-gray-800">기사 크롤링</h3>
              <p className="text-[13px] text-gray-500 leading-relaxed">네이버 뉴스에서 최신 기사를 수집하고 AI 분석을 위한 전처리를 수행합니다.</p>
              <div className="flex items-center justify-between pt-1">
                <label className="flex items-center gap-2 font-medium text-[13px] text-gray-600">
                  총
                  <input
                    type="number"
                    value={crawlMax === 0 ? '' : crawlMax}
                    onChange={e => setCrawlMax(Number(e.target.value))}
                    placeholder="0"
                    min="1"
                    max="100"
                    className="w-16 border border-gray-300 rounded-lg text-center font-bold text-base focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 transition-all py-1 bg-white"
                  />
                  개 수집
                </label>
                <button
                  onClick={() => crawlMutation.mutate({ max: crawlMax, categories: ALL_CATEGORIES })}
                  disabled={crawlMax <= 0 || crawlState.status === 'crawling' || crawlMutation.isPending || isLoadingStats}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg font-semibold text-[14px] disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed transition-all shadow-sm"
                >
                  {crawlState.status === 'crawling' || crawlMutation.isPending ? '처리 중...' : '크롤링 시작'}
                </button>
              </div>
            </div>

            {/* 상태 패널 */}
            <div className="flex-grow flex flex-col justify-center rounded-xl border border-gray-200 bg-white min-h-[220px] overflow-hidden">
              {crawlState.status === 'crawling' || crawlMutation.isPending ? (
                <div className="flex flex-col w-full h-full p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="flex gap-1.5">
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.3s' }}></div>
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.15s' }}></div>
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce"></div>
                    </div>
                    <p className="text-[14px] font-bold text-blue-700">
                      {processStatus?.current_step && processStatus.current_step !== '대기 중'
                        ? `${processStatus.current_step} 진행 중...`
                        : '크롤링 작업 준비 중...'}
                    </p>
                  </div>
                  <div className="w-full h-40 bg-[#1e1e1e] rounded-lg p-3 overflow-y-auto font-mono text-[12px] flex flex-col gap-1.5 shadow-inner">
                    {logs.length === 0 ? (
                      <p className="text-gray-400">로그 대기 중...</p>
                    ) : (
                      logs.map((log, idx) => (
                        <div key={idx} className="flex gap-2 items-start leading-relaxed">
                          <span className="text-gray-500 shrink-0">[{log.time}]</span>
                          <span className="text-green-400 shrink-0">[{log.step}]</span>
                          <span className="text-gray-200">{log.message}</span>
                        </div>
                      ))
                    )}
                    <div ref={logsEndRef} />
                  </div>
                </div>
              ) : crawlState.status === 'success' ? (
                <div className="flex flex-col w-full h-full p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xl">{crawlState.newCount === 0 ? '✨' : '🎉'}</span>
                    <p className={`text-[14px] font-bold ${crawlState.newCount === 0 ? 'text-blue-600' : 'text-green-600'}`}>
                      {typeof crawlState.newCount === 'number'
                        ? crawlState.newCount === 0
                          ? '추가된 새로운 기사가 없습니다 (모두 기존 DB와 중복됨)'
                          : `총 ${crawlState.newCount}개의 기사가 수집 및 분석되었습니다!`
                        : '수집 결과를 확인하는 중입니다...'}
                    </p>
                  </div>
                  <div className="w-full h-40 bg-[#1e1e1e] rounded-lg p-3 overflow-y-auto font-mono text-[12px] flex flex-col gap-1.5 shadow-inner opacity-80">
                    {logs.length === 0 ? (
                      <p className="text-gray-400">수집된 로그가 없습니다.</p>
                    ) : (
                      logs.map((log, idx) => (
                        <div key={idx} className="flex gap-2 items-start leading-relaxed">
                          <span className="text-gray-500 shrink-0">[{log.time}]</span>
                          <span className="text-green-400 shrink-0">[{log.step}]</span>
                          <span className="text-gray-200">{log.message}</span>
                        </div>
                      ))
                    )}
                    <div ref={logsEndRef} />
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center w-full h-full p-4">
                  <p className="text-[13px] font-medium text-gray-400 tracking-tight">새로운 크롤링 작업을 시작해보세요.</p>
                </div>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
