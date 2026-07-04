import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export default function LogPage() {
  const navigate = useNavigate()
  const isAdmin = localStorage.getItem('et_user') === 'etdashboard@naver.com'

  useEffect(() => {
    if (!isAdmin) navigate('/')
  }, [isAdmin, navigate])

  if (!isAdmin) return null

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
        <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <h1 className="text-[20px] font-extrabold text-gray-900 tracking-tight mb-2">퍼널 분석 결과</h1>
          <p className="text-[14px] text-gray-500 font-medium leading-relaxed mb-6">
            노출 → 클릭 → 상세 조회 등 사용자 행동 퍼널 분석 데이터가 준비되는 대로 이곳에 표시됩니다.
          </p>
          <div className="rounded-xl border border-dashed border-gray-200 bg-gray-50/70 p-10 text-center">
            <p className="text-gray-900 font-bold tracking-tight mb-2">아직 표시할 데이터가 없습니다.</p>
            <p className="text-gray-500 text-[14px] font-medium">퍼널 분석 기능이 준비되는 대로 이 페이지에서 확인할 수 있습니다.</p>
          </div>
        </section>
      </main>
    </div>
  )
}
