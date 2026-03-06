import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { fetchStats, startCrawl } from '../api/client'

export default function AdminPage() {
  const [maxArticles, setMaxArticles] = useState(10)
  const [crawlStatus, setCrawlStatus] = useState<string | null>(null)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: crawlStatus === 'started' ? 5000 : false,
  })

  const mutation = useMutation({
    mutationFn: () => startCrawl(maxArticles),
    onSuccess: () => {
      setCrawlStatus('started')
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    },
    onError: () => setCrawlStatus('error'),
  })

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-900">Admin</h1>
        <button onClick={() => navigate('/')} className="text-sm text-slate-500 hover:text-slate-800">
          ← 메인으로
        </button>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">
        <div className="bg-white rounded-xl border border-slate-100 p-6 space-y-4">
          <h2 className="font-semibold text-slate-800">DB 현황</h2>
          {isLoading ? (
            <p className="text-sm text-slate-400">불러오는 중...</p>
          ) : stats ? (
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-2xl font-bold text-slate-900">{stats.total_articles}</p>
                <p className="text-xs text-slate-400 mt-1">총 기사</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-2xl font-bold text-slate-900">{stats.sources.length}</p>
                <p className="text-xs text-slate-400 mt-1">언론사</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-2xl font-bold text-yellow-600">{stats.unanalyzed_count}</p>
                <p className="text-xs text-slate-400 mt-1">미분석</p>
              </div>
            </div>
          ) : null}
        </div>

        <div className="bg-white rounded-xl border border-slate-100 p-6 space-y-4">
          <h2 className="font-semibold text-slate-800">크롤링 실행</h2>
          <div className="flex items-center gap-3">
            <label className="text-sm text-slate-600 whitespace-nowrap">카테고리당 기사 수</label>
            <input
              type="number"
              min={1}
              max={50}
              value={maxArticles}
              onChange={e => setMaxArticles(Number(e.target.value))}
              className="w-24 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={() => { setCrawlStatus(null); mutation.mutate() }}
            disabled={mutation.isPending}
            className="w-full bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? '요청 중...' : '크롤링 시작'}
          </button>
          {crawlStatus === 'started' && (
            <p className="text-sm text-green-600">크롤링이 백그라운드에서 실행 중입니다. 완료되면 DB 현황이 자동 갱신됩니다.</p>
          )}
          {crawlStatus === 'error' && (
            <p className="text-sm text-red-500">크롤링 요청에 실패했습니다. 서버 상태를 확인하세요.</p>
          )}
        </div>
      </main>
    </div>
  )
}
