import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export default function FeedbackPage() {
  const navigate = useNavigate()
  const [feedbacks, setFeedbacks] = useState<any[]>([])
  const [globalStats, setGlobalStats] = useState<Record<string, any>>({})
  const userId = 'guest'
  
  useEffect(() => {
    try {
      const storageKey = `feedbacks_${userId}`;
      const storedFeedbacks = localStorage.getItem(storageKey);
      let userFeedbacks = storedFeedbacks && storedFeedbacks !== 'undefined' ? JSON.parse(storedFeedbacks) : [];
      if (!Array.isArray(userFeedbacks)) userFeedbacks = [];
      
      userFeedbacks.sort((a: any, b: any) => (b?.timestamp || 0) - (a?.timestamp || 0));
      setFeedbacks(userFeedbacks);

      const statsKey = 'global_feedback_stats';
      const storedStats = localStorage.getItem(statsKey);
      setGlobalStats(storedStats && storedStats !== 'undefined' ? JSON.parse(storedStats) : {});
    } catch (e) {
      console.error('데이터 파싱 에러:', e);
      setFeedbacks([]);
      setGlobalStats({});
    }
  }, []);

  const safeFeedbacks = Array.isArray(feedbacks) ? feedbacks : [];
  const likeCount = safeFeedbacks.filter(f => f?.type === 'like').length;
  const dislikeCount = safeFeedbacks.filter(f => f?.type === 'dislike').length;

  return (
    <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-900">
      <header className="w-full bg-gray-100 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <button onClick={() => navigate(-1)} className="flex items-center justify-center w-11 h-11 text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-xl transition-colors shadow-sm" title="뒤로가기">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
            </svg>
          </button>
          <h1 className="text-xl font-extrabold text-gray-800"></h1>
          <div className="w-11"></div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-10 pt-6 space-y-6">
        {/* 유저 통계 정보 영역 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col md:flex-row items-center gap-6 justify-between">
           <div>
             <h2 className="text-2xl font-extrabold text-gray-800"><span className="text-blue-600">나</span>의 피드백 통계</h2>
             <p className="text-gray-500 mt-2 font-medium">지금까지 총 <span className="font-bold text-gray-700">{safeFeedbacks.length}</span>개의 기사에 의견을 남기셨습니다.</p>
           </div>
           <div className="flex gap-4">
             <div className="bg-green-50 px-6 py-4 rounded-xl border border-green-100 text-center shadow-sm">
               <p className="text-green-700 text-sm font-bold mb-1">유용함 (좋아요)</p>
               <p className="text-3xl font-extrabold text-green-600">{likeCount}개</p>
             </div>
             <div className="bg-red-50 px-6 py-4 rounded-xl border border-red-100 text-center shadow-sm">
               <p className="text-red-700 text-sm font-bold mb-1">개선 필요 (싫어요)</p>
               <p className="text-3xl font-extrabold text-red-600">{dislikeCount}개</p>
             </div>
           </div>
        </section>

        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-extrabold text-gray-800 mb-4">피드백 남긴 기사 목록</h2>
          {safeFeedbacks.length === 0 ? (
            <div className="text-center py-10 text-gray-500 font-medium">평가한 기사가 없습니다.</div>
          ) : (
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 w-32 text-center">나의 평가</th>
                  <th scope="col" className="px-6 py-3">기사 제목</th>
                  <th scope="col" className="px-6 py-3 w-28 text-center">총 좋아요</th>
                  <th scope="col" className="px-6 py-3 w-28 text-center">총 싫어요</th>
                  <th scope="col" className="px-6 py-3 w-48 text-right">평가 시간</th>
                </tr>
              </thead>
              <tbody>
                {safeFeedbacks.map((log, index) => {
                  const stats = globalStats?.[log?.articleId] || { like: log?.type === 'like' ? 1 : 0, dislike: log?.type === 'dislike' ? 1 : 0 };
                  return (
                  <tr key={log?.timestamp || index} className="border-b last:border-b-0 hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 text-center">
                      {log?.type === 'like' ? (
                        <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-green-100 text-green-600 font-bold ring-1 ring-green-500" title="유용함">
                          👍
                        </span>
                      ) : (
                        <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-red-100 text-red-600 font-bold ring-1 ring-red-500" title="개선필요">
                          👎
                        </span>
                      )}
                    </td>
                    <td onClick={() => navigate(`/article/${log?.articleId}`)} className="px-6 py-4 font-bold text-gray-800 hover:text-blue-600 hover:underline cursor-pointer break-keep text-[15px]">
                      {log?.title || '알 수 없는 기사'}
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-green-600 font-extrabold bg-green-50 px-3 py-1.5 rounded-full text-[13px]">{stats.like}개</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-red-500 font-extrabold bg-red-50 px-3 py-1.5 rounded-full text-[13px]">{stats.dislike}개</span>
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-400 text-right text-xs">
                      {log?.timestamp ? new Date(log.timestamp).toLocaleString('ko-KR') : '-'}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
      </main>
    </div>
  )
}