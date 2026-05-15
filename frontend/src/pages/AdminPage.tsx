import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchStats } from '../api/client'
import type { AdminStats } from '../api/client'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function AdminPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [crawlMax, setCrawlMax] = useState(0)
  const ALL_CATEGORIES = ["IT/과학", "경제", "사회", "생활/문화", "세계", "정치", "연예", "스포츠"]

  // 실시간 로그 누적 상태 관리
  const hasStartedRef = useRef(false)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const lastStartRef = useRef<number>(0)
  const [logs, setLogs] = useState<{ step: string; message: string; time: string }[]>([])

  // 로컬 스토리지에 상태를 저장하여 페이지를 나갔다 들어와도 알림 상태가 유지되게 함
  const [crawlState, setCrawlState] = useState<{status: 'idle' | 'crawling' | 'success', prevCount: number | null, newCount?: number}>(() => {
    const stored = localStorage.getItem('crawlingState')
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        // Don't persist 'success' state on reload, show the default idle message instead.
        if (parsed && typeof parsed === 'object' && 'status' in parsed && parsed.status === 'crawling') {
            return parsed;
        }
      } catch(e) {}
    }
    return { status: 'idle', prevCount: null }
  })

  const { data: stats, isLoading: isLoadingStats } = useQuery<AdminStats>({
    queryKey: ['adminStats'],
    queryFn: fetchStats,
    refetchInterval: 5000, // 5초마다 통계 자동 갱신
  })

  const { data: processStatus } = useQuery({
    queryKey: ['processStatus'],
    queryFn: async () => {
        const res = await fetch(`${API_BASE}/api/admin/process-status`);
        if (!res.ok) return null;
        return res.json();
    },
    // 백엔드의 빠른 작업 전환을 모두 캐치하기 위해 0.2초(200ms) 단위로 실시간 조회
    refetchInterval: crawlState.status === 'crawling' ? 200 : false,
    refetchOnWindowFocus: false,
    staleTime: 0, // 캐시를 사용하지 않고 무조건 최신 상태만 가져오도록 설정
  });

  // Crawl mutation is special as it takes an argument
  const crawlMutation = useMutation({
    mutationFn: async (payload: { max: number, categories: string[] }) => {
      // client.ts 수정 없이 바로 동작할 수 있도록 직접 API를 호출하여 카테고리를 전달합니다.
      const perCat = Math.max(1, Math.ceil(payload.max / payload.categories.length));
      const res = await fetch(`${API_BASE}/api/admin/crawl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_articles_per_category: perCat,
          categories: payload.categories,
          total_articles: payload.max
        })
      });
      if (!res.ok) throw new Error('크롤링 서버 에러');
      return res.json();
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

  // 상태가 바뀔 때마다 로컬 스토리지 업데이트
  useEffect(() => {
    localStorage.setItem('crawlingState', JSON.stringify(crawlState))
  }, [crawlState])

  // 총 기사 수(total_articles)가 변동되거나, 백엔드 프로세스가 'idle'로 돌아오면 완료로 간주
  useEffect(() => {
    if (crawlState.status === 'crawling') {
      if (processStatus?.process_name === 'crawl') {
        hasStartedRef.current = true;
      }

      if (processStatus?.process_name === 'idle') {
        if (hasStartedRef.current) {
          // 1. 중복 실행 방지를 위해 먼저 상태를 success로 전환 (newCount는 잠시 undefined 상태)
          setCrawlState(prev => ({ ...prev, status: 'success' }));
          hasStartedRef.current = false;
          
          // 2. 백엔드 작업 종료 직후, 실제 DB 최신 통계를 직접 한 번 더 가져와서 정확한 증가량을 계산합니다.
          fetchStats().then(latestStats => {
            const newCount = (typeof latestStats?.total_articles === 'number' && typeof crawlState.prevCount === 'number')
              ? Math.max(0, latestStats.total_articles - crawlState.prevCount)
              : 0;
            
            setCrawlState(prev => ({ ...prev, newCount, prevCount: latestStats?.total_articles ?? null }));
            queryClient.invalidateQueries({ queryKey: ['adminStats'] });

            // 통신 지연으로 마지막 완료 로그가 누락되었을 경우를 대비해 확실하게 완료 메시지를 로그창에 추가합니다.
            setLogs(prev => {
              const finalMsg = newCount > 0 
                ? `✅ 작업 완료! 총 ${newCount}개의 기사를 수집했습니다.` 
                : `❌ 추가된 기사가 없습니다 (모두 기존 DB와 중복됨)`;
            
              // 로그의 맨 마지막에 이미 완료 메시지가 찍혀있다면 중복해서 띄우지 않습니다.
              if (prev.length > 0 && (prev[prev.length - 1].message.includes('작업 완료') || prev[prev.length - 1].message.includes('기사가 없습니다'))) {
                return prev;
              }
              return [...prev, {
                step: '완료',
                message: finalMsg,
                time: new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
              }];
            });
          });
        } else if (Date.now() - lastStartRef.current > 3000) {
          // 💡 새로고침이나 탭 이동 등으로 프론트엔드 상태만 'crawling'으로 남고, 
          // 실제 백엔드는 이미 대기(idle) 중일 때 버튼 비활성화 멈춤 현상 강제 해제
          setCrawlState({ status: 'idle', prevCount: null });
        }
      }
    }
  }, [processStatus?.process_name, crawlState.status, crawlState.prevCount, queryClient])

  // 백엔드 상태 바뀔 때마다 로그 배열에 추가
  useEffect(() => {
    if (processStatus && crawlState.status === 'crawling') {
      if (processStatus.last_message) {
        setLogs(prev => {
          // 바로 직전 메시지와 같으면 중복 추가 방지
          if (prev.length > 0 && prev[prev.length - 1].message === processStatus.last_message) {
            return prev
          }
          return [...prev, {
            step: processStatus.current_step || '시스템',
            message: processStatus.last_message,
            time: new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
          }]
        })
      }
    }
  }, [processStatus?.last_message, processStatus?.current_step, crawlState.status]) // 객체 전체가 아닌 개별 메시지 내용에만 반응하도록 의존성 최적화

  // 로그 창 자동 스크롤
  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-900">
      <header className="w-full bg-gray-100 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-start">
          <button onClick={() => navigate('/')} className="flex items-center justify-center w-11 h-11 text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-xl transition-colors shadow-sm" title="메인으로 돌아가기">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
            </svg>
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-10 pt-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* 1. 좌측 상단: DB 현황 */}
          <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col">
              <h2 className="text-lg font-extrabold text-gray-800 mb-4">DB 현황</h2>
              {isLoadingStats ? (
                <div className="text-center py-10">로딩 중...</div>
              ) : stats ? (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {[{label: '총 기사 수', value: stats.total_articles}, {label: '미분석 기사', value: stats.unanalyzed_count, color: 'text-red-500'}, {label: '수집된 언론사', value: stats.sources?.length || 0}].map(item => (
                    <div key={item.label} className="bg-gray-50/80 p-4 rounded-lg border border-gray-200/80">
                      <p className="text-sm font-medium text-gray-500">{item.label}</p>
                      <p className={`text-3xl font-bold mt-1 ${item.color || ''}`}>{item.value}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-10 text-red-500">통계 정보를 불러오지 못했습니다.</div>
              )}
            </section>

          {/* 2. 우측 상단: API 사용량 및 상태 모니터링 */}
          <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col">
            <h2 className="text-lg font-extrabold text-gray-800 mb-4">API 사용량 및 상태 모니터링</h2>
            {isLoadingStats ? (
              <div className="text-center py-10 flex-grow flex items-center justify-center">로딩 중...</div>
            ) : stats?.api_usage ? (
              <div className="flex-grow flex flex-col justify-center">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-medium text-gray-500">Gemini API 할당량</p>
                  <p className="text-lg font-bold text-blue-600">{stats.api_usage.quota_percent}%</p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${stats.api_usage.quota_percent}%` }}></div>
                </div>
                <p className="text-xs text-gray-400 mt-3">마지막 성공: {stats.api_usage.last_success_time}</p>
              </div>
            ) : (
              <div className="text-center py-10 text-red-500 flex-grow flex items-center justify-center">API 정보를 불러오지 못했습니다.</div>
            )}
          </section>

          {/* 3. 좌측 하단: 카테고리별 분석 상태 */}
          <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-extrabold text-gray-800">카테고리별 분석 상태</h2>
              </div>
              <table className="w-full text-sm text-left text-gray-500">
                <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3">카테고리</th>
                    <th scope="col" className="px-6 py-3">총 기사</th>
                    <th scope="col" className="px-6 py-3 text-blue-600">오늘의 기사</th>
                    <th scope="col" className="px-6 py-3">분석률</th>
                  </tr>
                </thead>
                <tbody>
                  {stats?.category_stats?.map(cat => {
                    const analyzedRate = cat.total > 0 ? ((cat.total - cat.unanalyzed) / cat.total) * 100 : 0;
                    return (
                      <tr key={cat.category} className="border-b last:border-b-0">
                        <th scope="row" className="px-6 py-4 font-bold text-gray-900 whitespace-nowrap">{cat.category}</th>
                        <td className="px-6 py-4">{cat.total}</td>
                        <td className="px-6 py-4 font-bold text-blue-600">{(cat as any).today_articles || 0}</td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div className="bg-green-500 h-2.5 rounded-full" style={{ width: `${analyzedRate}%` }}></div>
                            </div>
                            <span className="font-medium text-gray-700">{Math.round(analyzedRate)}%</span>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </section>

          {/* 4. 우측 하단: 자동 크롤링 설정 */}
          <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-full">
              <h2 className="text-lg font-extrabold text-gray-800 mb-4">자동 크롤링 설정</h2>
            
            <div className="bg-gray-50/70 p-4 rounded-lg border border-gray-200/80 space-y-3 w-full mb-4">
                <h3 className="font-bold text-lg">기사 크롤링</h3>
                <p className="text-sm text-gray-600 mt-1">네이버 뉴스에서 최신 기사를 수집하고 AI 분석을 위한 전처리를 수행합니다.</p>
                <div className="flex flex-col gap-4 pt-2">
                  <div className="flex items-center justify-between mt-2">
                    <label className="flex items-center gap-2 font-medium text-sm text-gray-700">
                      총
                      <input type="number" value={crawlMax === 0 ? '' : crawlMax} onChange={e => setCrawlMax(Number(e.target.value))} placeholder="0" min="1" max="100" className="w-16 border-gray-300 rounded-md text-center font-bold text-base" />
                      개 수집
                    </label>
                    <button
                      onClick={() => {
                        crawlMutation.mutate({ max: crawlMax, categories: ALL_CATEGORIES })
                      }}
                      disabled={crawlMax <= 0 || crawlState.status === 'crawling' || crawlMutation.isPending || isLoadingStats}
                      className="bg-blue-600 text-white px-5 py-2.5 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 transition-all"
                    >
                      {crawlState.status === 'crawling' || crawlMutation.isPending ? '처리 중...' : '크롤링 시작'}
                    </button>
                  </div>
                </div>
              </div>

            {/* 남은 흰색 배경 영역에 띄우는 알림 패널 */}
            <div className="flex-grow flex flex-col justify-center rounded-xl border border-gray-200 bg-white min-h-[220px] overflow-hidden">
              {crawlState.status === 'crawling' || crawlMutation.isPending ? (
                <div className="flex flex-col w-full h-full p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="flex gap-1.5">
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.3s' }}></div>
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.15s' }}></div>
                      <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce"></div>
                    </div>
                    <p className="text-[15px] font-bold text-blue-700">
                      {processStatus?.current_step && processStatus.current_step !== '대기 중' 
                        ? `${processStatus.current_step} 진행 중...` 
                        : '크롤링 작업 준비 중...'}
                    </p>
                  </div>
                  
                  {/* 터미널 스타일 로그 뷰어 */}
                  <div className="w-full h-40 bg-[#1e1e1e] rounded-lg p-3 overflow-y-auto font-mono text-[13px] flex flex-col gap-1.5 shadow-inner">
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
                    <p className={`text-[15px] font-bold ${crawlState.newCount === 0 ? 'text-blue-600' : 'text-green-600'}`}>
                      {typeof crawlState.newCount === 'number'
                        ? crawlState.newCount === 0 
                          ? '추가된 새로운 기사가 없습니다 (모두 기존 DB와 중복됨)'
                          : `총 ${crawlState.newCount}개의 기사가 수집 및 분석되었습니다!`
                        : '수집 결과를 확인하는 중입니다...'}
                    </p>
                  </div>
                  
                  {/* 완료 후에도 로그 내역 유지 */}
                  <div className="w-full h-40 bg-[#1e1e1e] rounded-lg p-3 overflow-y-auto font-mono text-[13px] flex flex-col gap-1.5 shadow-inner opacity-80">
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
                  <p className="text-[14px] font-medium text-gray-400 tracking-tight">새로운 크롤링 작업을 시작해보세요.</p>
                </div>
              )}
            </div>
            </section>
        </div>
      </main>
    </div>
  )
}
