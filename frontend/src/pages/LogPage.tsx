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
    <div className="min-h-screen bg-paper font-sans antialiased text-gray-900">
      <header className="w-full px-6 py-4 flex items-center justify-start bg-paper/90 backdrop-blur-md sticky top-0 z-10">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center justify-center w-11 h-11 bg-white hover:bg-gray-50 rounded-full transition-colors border border-gray-200 hover:border-navy-300"
          title="뒤로가기"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="#2C4460" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
        </button>
      </header>

      <main className="max-w-6xl mx-auto px-4 pb-12 pt-8">
        <section className="bg-white rounded-lg border border-gray-200 p-6">
          <h1 className="text-[20px] font-extrabold text-gray-900 tracking-tight mb-2">퍼널 분석 결과</h1>
          <p className="text-[14px] text-gray-500 font-medium leading-relaxed mb-6">
            노출 → 클릭 → 상세 조회 등 사용자 행동 퍼널 분석 데이터가 준비되는 대로 이곳에 표시됩니다.
          </p>
          <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50/70 p-10 text-center">
            <p className="text-gray-900 font-bold tracking-tight mb-2">아직 표시할 데이터가 없습니다.</p>
            <p className="text-gray-500 text-[14px] font-medium">퍼널 분석 기능이 준비되는 대로 이 페이지에서 확인할 수 있습니다.</p>
          </div>
        </section>
      </main>
    </div>
  )
}
