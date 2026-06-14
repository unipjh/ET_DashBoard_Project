import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export default function FeedbackPage() {
  const navigate = useNavigate()
  const [feedbacks, setFeedbacks] = useState<any[]>([])
  const [globalStats, setGlobalStats] = useState<Record<string, any>>({})
  const userId = 'guest'

  useEffect(() => {
    try {
      const storageKey = `feedbacks_${userId}`
      const storedFeedbacks = localStorage.getItem(storageKey)
      let userFeedbacks = storedFeedbacks && storedFeedbacks !== 'undefined' ? JSON.parse(storedFeedbacks) : []
      if (!Array.isArray(userFeedbacks)) userFeedbacks = []
      userFeedbacks.sort((a: any, b: any) => (b?.timestamp || 0) - (a?.timestamp || 0))
      setFeedbacks(userFeedbacks)

      const statsKey = 'global_feedback_stats'
      const storedStats = localStorage.getItem(statsKey)
      setGlobalStats(storedStats && storedStats !== 'undefined' ? JSON.parse(storedStats) : {})
    } catch (e) {
      console.error('데이터 파싱 에러:', e)
      setFeedbacks([])
      setGlobalStats({})
    }
  }, [])

  const safeFeedbacks = Array.isArray(feedbacks) ? feedbacks : []
  const likeCount = safeFeedbacks.filter(f => f?.type === 'like').length
  const dislikeCount = safeFeedbacks.filter(f => f?.type === 'dislike').length

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
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-12 pt-8 space-y-5">
        {/* 유저 통계 영역 */}
        <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm flex flex-col md:flex-row items-center gap-6 justify-between">
          <div>
            <h2 className="text-2xl font-extrabold text-gray-800 tracking-tight">
              <span className="text-blue-600">나</span>의 피드백 통계
            </h2>
            <p className="text-gray-500 mt-2 font-medium text-[14px]">
              지금까지 총 <span className="font-bold text-gray-700">{safeFeedbacks.length}</span>개의 기사에 의견을 남기셨습니다.
            </p>
          </div>
          <div className="flex gap-3">
            <div className="bg-green-50 border border-green-100 px-6 py-4 rounded-xl text-center shadow-sm">
              <p className="text-green-700 text-xs font-bold mb-1.5 tracking-wide">유용함 (좋아요)</p>
              <p className="text-3xl font-extrabold text-green-600">{likeCount}<span className="text-lg ml-0.5">개</span></p>
            </div>
            <div className="bg-red-50 border border-red-100 px-6 py-4 rounded-xl text-center shadow-sm">
              <p className="text-red-700 text-xs font-bold mb-1.5 tracking-wide">개선 필요 (싫어요)</p>
              <p className="text-3xl font-extrabold text-red-600">{dislikeCount}<span className="text-lg ml-0.5">개</span></p>
            </div>
          </div>
        </section>

        {/* 피드백 기사 목록 */}
        <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-[16px] font-extrabold text-gray-800 mb-5 tracking-tight">피드백 남긴 기사 목록</h2>
          {safeFeedbacks.length === 0 ? (
            <div className="text-center py-12 text-gray-500 font-medium">평가한 기사가 없습니다.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-gray-500">
                <thead>
                  <tr className="border-b border-gray-100">
                    <th scope="col" className="px-4 py-3 w-28 text-center text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50 rounded-tl-lg">나의 평가</th>
                    <th scope="col" className="px-4 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50">기사 제목</th>
                    <th scope="col" className="px-4 py-3 w-24 text-center text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50">좋아요</th>
                    <th scope="col" className="px-4 py-3 w-24 text-center text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50">싫어요</th>
                    <th scope="col" className="px-4 py-3 w-44 text-right text-xs font-bold text-gray-500 uppercase tracking-wider bg-gray-50 rounded-tr-lg">평가 시간</th>
                  </tr>
                </thead>
                <tbody>
                  {safeFeedbacks.map((log, index) => {
                    const stats = globalStats?.[log?.articleId] || { like: log?.type === 'like' ? 1 : 0, dislike: log?.type === 'dislike' ? 1 : 0 }
                    return (
                      <tr key={log?.timestamp || index} className="border-b border-gray-50 last:border-b-0 hover:bg-gray-50/60 transition-colors">
                        <td className="px-4 py-4 text-center">
                          {log?.type === 'like' ? (
                            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-green-100 text-green-600 ring-1 ring-green-400" title="유용함">
                              👍
                            </span>
                          ) : (
                            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-red-100 text-red-600 ring-1 ring-red-400" title="개선필요">
                              👎
                            </span>
                          )}
                        </td>
                        <td
                          onClick={() => navigate(`/article/${log?.articleId}`)}
                          className="px-4 py-4 font-bold text-gray-800 hover:text-blue-600 hover:underline cursor-pointer break-keep text-[14px] transition-colors"
                        >
                          {log?.title || '알 수 없는 기사'}
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className="text-green-600 font-extrabold bg-green-50 border border-green-100 px-2.5 py-1 rounded-full text-[12px]">{stats.like}개</span>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className="text-red-500 font-extrabold bg-red-50 border border-red-100 px-2.5 py-1 rounded-full text-[12px]">{stats.dislike}개</span>
                        </td>
                        <td className="px-4 py-4 font-mono text-gray-400 text-right text-xs">
                          {log?.timestamp ? new Date(log.timestamp).toLocaleString('ko-KR') : '-'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
