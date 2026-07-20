import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchDashboardStats } from '../api/client'
import type { FunnelStepData } from '../api/client'

const STEPS = [
  { key: 'visit_main', label: '1. 메인 방문' },
  { key: 'view_article_detail', label: '2. 기사 상세조회' },
  { key: 'open_trust_modal', label: '3. 신뢰도 모달 오픈' },
  { key: 'click_feedback', label: '4. 피드백 제출' },
] as const

export default function LogPage() {
  const navigate = useNavigate()
  const isAdmin = localStorage.getItem('et_user') === 'etdashboard@naver.com'

  useEffect(() => {
    if (!isAdmin) navigate('/')
  }, [isAdmin, navigate])

  const { data, isLoading } = useQuery({
    queryKey: ['dashboardStats'],
    queryFn: fetchDashboardStats,
    enabled: isAdmin,
    refetchInterval: 1 * 60 * 60 * 1000,
  })

  if (!isAdmin) return null

  const renderFunnelBars = (statData: FunnelStepData | undefined, colors: string[]) => {
    const maxVal = statData?.visit_main || 1
    return STEPS.map((step, idx) => {
      const count = statData?.[step.key as keyof FunnelStepData] || 0
      const widthPct = Math.max(Math.round((count / maxVal) * 100), 2)
      return (
        <div key={step.key} className="flex flex-col gap-1.5">
          <div className="flex justify-between items-end px-1">
            <span className="text-[14px] font-bold text-gray-700 tracking-tight">{step.label}</span>
            <span className="text-[12px] font-bold text-gray-500 bg-white/60 px-2 py-0.5 rounded-full shadow-sm">{count} ({widthPct}%)</span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-7 border border-black/5 overflow-hidden">
            <div className={`h-full flex items-center justify-end transition-all duration-1000 ${colors[idx]}`} style={{ width: `${widthPct}%` }} />
          </div>
        </div>
      )
    })
  }

  const maxSession = Math.max(...(data?.das?.map(d => d.sessions) || [10]), 10)
  const dasPoints = data?.das?.map((d, i, arr) => {
    const x = (i / (arr.length - 1 || 1)) * 100
    const y = 100 - (d.sessions / maxSession) * 100
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="min-h-screen bg-paper font-sans antialiased text-gray-900 pb-20">
      <header className="w-full px-6 py-4 flex items-center justify-start bg-paper/90 backdrop-blur-md sticky top-0 z-20">
        <button onClick={() => navigate(-1)} className="flex items-center justify-center w-11 h-11 bg-white hover:bg-gray-50 rounded-full transition-colors border border-gray-200 hover:border-navy-300">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="#2C4460" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
        </button>
        <div className="ml-4 flex flex-col">
          <h1 className="text-[18px] font-extrabold tracking-tight">ET DashBoard 통계 로그</h1>
          <span className="text-[12px] font-semibold text-gray-500">실시간 데이터 연동 (1시간 주기 갱신)</span>
        </div>
      </header>

      {isLoading && !data ? (
        <div className="flex justify-center py-32"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div></div>
      ) : (
        <main className="max-w-[1400px] mx-auto px-6 pt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            <section className="bg-white rounded-3xl border border-gray-200 p-8 shadow-sm flex flex-col justify-center">
              <h2 className="text-xl font-extrabold text-gray-900 mb-1">사용자 여정 퍼널 분석</h2>
              <p className="text-sm text-gray-500 mb-6 font-medium">메인 방문부터 피드백까지 전체 유저의 전환율입니다.</p>
              <div className="flex flex-col gap-4">
                {renderFunnelBars(data?.funnels?.overall, ['bg-blue-100', 'bg-blue-300', 'bg-blue-400', 'bg-blue-600'])}
              </div>
            </section>

            <section className="bg-white rounded-3xl border border-gray-200 p-8 shadow-sm">
              <h2 className="text-xl font-extrabold text-gray-900 mb-1">Daily Active Sessions</h2>
              <p className="text-sm text-gray-500 mb-8 font-medium">일자별 활성 고유 세션 수 추이입니다.</p>
                <div className="relative w-full h-48 mt-4 border-b border-l border-gray-200 pb-2 pl-2">
                  {data?.das && data.das.length > 0 && (
                    <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none">
                      <polyline fill="none" stroke="#3b82f6" strokeWidth="3" points={dasPoints} strokeLinejoin="round" />
                      {data.das.map((d, i, arr) => {
                        const x = (i / (arr.length - 1 || 1)) * 100
                        const y = 100 - (d.sessions / maxSession) * 100
                        return (
                          // 호버 영역(g)의 크기를 r="10"으로 키워 마우스 감지 범위를 넓힘
                          <g key={i} className="group cursor-pointer">
                            <circle cx={`${x}%`} cy={`${y}%`} r="6" fill="#3b82f6" stroke="white" strokeWidth="2" className="group-hover:fill-blue-800 transition-all" />
                            
                            {/* 호버 시 나타나는 툴팁 박스 */}
                            <g className="opacity-0 group-hover:opacity-100 transition-opacity">
                              <rect x={`${x}%`} y={`${y}%`} rx="4" width="60" height="26" transform="translate(-30, -50)" fill="#1e293b" />
                              <text x={`${x}%`} y={`${y}%`} dy="-32" textAnchor="middle" className="text-[11px] font-bold fill-white">
                                {d.sessions} 세션
                              </text>
                            </g>
                          </g>
                        )
                      })}
                    </svg>
                  )}
                {/* X축 날짜 라벨 */}
                <div className="absolute top-full left-0 w-full flex justify-between pt-3">
                  {data?.das?.map((d, i) => (
                    <span key={i} className="text-[11px] font-bold text-gray-400 -translate-x-1/2">{d.date}</span>
                  ))}
                </div>
              </div>
            </section>

            <section className="bg-white rounded-3xl border border-gray-200 p-8 shadow-sm">
              <h2 className="text-xl font-extrabold text-gray-900 mb-1">🔥 TOP 10 인기 기사 분석</h2>
              <p className="text-sm text-gray-500 mb-5 font-medium">상세 조회수가 높은 기사의 인게이지먼트 현황입니다.</p>
              
              <div className="overflow-x-auto rounded-xl border border-gray-100">
                <table className="w-full text-left border-collapse min-w-[500px]">
                  <thead className="bg-slate-50">
                    <tr className="border-b border-gray-200 text-gray-500 text-[13px]">
                      <th className="py-2.5 px-4 font-bold text-center">순위</th>
                      <th className="py-2.5 px-2 font-bold">기사 제목</th>
                      <th className="py-2.5 px-2 font-bold text-center">상세조회</th>
                      <th className="py-2.5 px-2 font-bold text-center">모달오픈</th>
                      <th className="py-2.5 px-2 font-bold text-center">피드백</th>
                    </tr>
                  </thead>
                  <tbody className="text-[14px]">
                    {data?.top_articles?.map((article, idx) => (
                      <tr key={idx} className="border-b border-gray-100 hover:bg-blue-50/30 transition-colors">
                        <td className="py-3 px-4 font-black text-blue-500 text-center">{idx + 1}</td>
                        <td className="py-3 px-2 font-bold text-gray-800"><div className="line-clamp-1 w-64">{article.title}</div></td>
                        <td className="py-3 px-2 font-semibold text-gray-600 text-center">{article.views}</td>
                        <td className="py-3 px-2 font-semibold text-gray-600 text-center">{article.trusts}</td>
                        <td className="py-3 px-2 font-extrabold text-blue-600 text-center">{article.feedbacks}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="bg-white rounded-3xl border border-gray-200 p-8 shadow-sm">
              <h2 className="text-xl font-extrabold text-gray-900 mb-1">유저 유형별 행동 퍼널 비교</h2>
              <p className="text-sm text-gray-500 mb-6 font-medium">회원(Member)과 비회원(Guest)의 전환율 차이입니다.</p>
              
              <div className="flex gap-4">
                <div className="flex-1 bg-blue-50/40 p-5 rounded-2xl border border-blue-100/70">
                  <h3 className="text-[14px] font-extrabold text-blue-900 mb-4 text-center">회원 (Members)</h3>
                  <div className="flex flex-col gap-3">
                    {renderFunnelBars(data?.funnels?.member, ['bg-blue-100', 'bg-blue-300', 'bg-blue-500', 'bg-blue-600'])}
                  </div>
                </div>
                <div className="flex-1 bg-slate-100/50 p-5 rounded-2xl border border-slate-200/70">
                  <h3 className="text-[14px] font-extrabold text-slate-700 mb-4 text-center">비회원 (Guests)</h3>
                  <div className="flex flex-col gap-3">
                    {renderFunnelBars(data?.funnels?.guest, ['bg-slate-200', 'bg-slate-300', 'bg-slate-400', 'bg-slate-500'])}
                  </div>
                </div>
              </div>
            </section>

          </div>
        </main>
      )}
    </div>
  )
}
