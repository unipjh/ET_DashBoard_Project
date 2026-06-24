import { useState, useEffect } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { fetchArticles, searchArticles, signup, login, getApiAssetUrl, fetchRecommendations } from '../api/client'
import type { Article, SearchResult } from '../api/client'
import ArticleCard from '../components/ArticleCard'
import { getSessionId, trackEvent, trackImpressions } from '../api/logger'

const CATEGORIES = ['전체', '정치', '경제', '사회', '생활/문화', '세계', 'IT/과학']

const getHue = (s: number) => {
  if (s <= 30) return (s / 30) * 15
  if (s <= 60) return 15 + ((s - 30) / 30) * 35
  return 50 + ((s - 60) / 40) * 90
}

const ArticleListSkeleton = () => (
  <div className="flex flex-col gap-3 w-full" aria-label="기사 목록 로딩 중">
    {Array.from({ length: 6 }).map((_, index) => (
      <div key={index} className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
        <div className="flex items-center gap-2 mb-3">
          <div className="h-3 w-16 rounded bg-gray-200 animate-pulse" />
          <div className="h-3 w-20 rounded bg-gray-100 animate-pulse" />
        </div>
        <div className="h-5 w-11/12 rounded bg-gray-200 animate-pulse mb-3" />
        <div className="h-4 w-3/5 rounded bg-gray-100 animate-pulse" />
      </div>
    ))}
  </div>
)

export default function MainPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const submittedQuery = searchParams.get('q') || ''
  const selectedCategory = searchParams.get('cat') || '전체'
  const page = parseInt(searchParams.get('page') || '1', 10)

  const [query, setQuery] = useState(submittedQuery)
  const location = useLocation()
  const navigate = useNavigate()
  const [isPageLoaded, setIsPageLoaded] = useState(true)

  const isSearchMode = !!submittedQuery

  useEffect(() => { trackEvent('visit_main') }, [])
  useEffect(() => { setQuery(submittedQuery) }, [submittedQuery])

  const {
    data: categoryFilteredData,
    isLoading: isLoadingCategoryArticles,
    isFetching: isFetchingCategoryArticles,
    isError: isCategoryError,
  } = useQuery({
    queryKey: ['articles', page, selectedCategory],
    queryFn: () => fetchArticles(page, 10, selectedCategory === '전체' ? undefined : selectedCategory),
    enabled: !isSearchMode,
    placeholderData: keepPreviousData,
    staleTime: 2 * 60 * 1000,
  })
  const categoryFilteredArticles = categoryFilteredData?.articles || []
  const categoryTotalCount = categoryFilteredData?.total_count || 0

  const { data: firstPageDataForTopPicks, isLoading: isLoadingFirstPageArticles } = useQuery({
    queryKey: ['articles', 1, selectedCategory],
    queryFn: () => fetchArticles(1, 10, selectedCategory === '전체' ? undefined : selectedCategory),
    enabled: !isSearchMode && selectedCategory === '전체',
    placeholderData: keepPreviousData,
    staleTime: 2 * 60 * 1000,
  })
  const firstPageArticlesForTopPicks = firstPageDataForTopPicks?.articles || []

  const sessionId = getSessionId()
  const currentUserId = localStorage.getItem('et_user')
  const { data: personalizedArticles = [], isLoading: isLoadingPersonalized } = useQuery({
    queryKey: ['recommendations', sessionId, currentUserId, selectedCategory, location.key],
    queryFn: () => fetchRecommendations(sessionId, currentUserId, 5),
    enabled: !isSearchMode && selectedCategory === '전체',
    staleTime: 0,
    refetchOnMount: 'always',
    refetchOnWindowFocus: true,
  })

  const {
    data: searchResults,
    isLoading: isSearching,
    isFetching: isFetchingSearch,
    isError: isSearchError,
  } = useQuery({
    queryKey: ['search', submittedQuery],
    queryFn: () => searchArticles(submittedQuery),
    enabled: isSearchMode,
    retry: 1,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const newParams = new URLSearchParams(searchParams)
    if (query.trim()) {
      trackEvent('execute_search', null, { query: query.trim() })
      newParams.set('q', query.trim())
      newParams.delete('cat')
    } else {
      newParams.delete('q')
    }
    newParams.set('page', '1')
    setSearchParams(newParams)
  }

  const handleCategoryClick = (cat: string) => {
    setQuery('')
    const newParams = new URLSearchParams(searchParams)
    newParams.delete('q')
    if (cat === '전체') newParams.delete('cat')
    else newParams.set('cat', cat)
    newParams.set('page', '1')
    setSearchParams(newParams)
  }

  const updatePage = (newPage: number) => {
    const newParams = new URLSearchParams(searchParams)
    newParams.set('page', newPage.toString())
    setSearchParams(newParams)
  }

  const isLoading = isSearchMode ? isSearching : isLoadingCategoryArticles
  const isRefreshing = isSearchMode ? isFetchingSearch : isFetchingCategoryArticles
  const hasPrimaryError = isSearchMode ? isSearchError : isCategoryError

  let articlesToRender: (Article | SearchResult)[] = []
  let totalCountForDisplay = 0

  if (isSearchMode) {
    articlesToRender = (searchResults || []).slice((page - 1) * 9, page * 9)
    totalCountForDisplay = (searchResults || []).length
  } else {
    const topPicks = selectedCategory === '전체'
      ? [...firstPageArticlesForTopPicks].sort((a, b) => (b.trust_score || 0) - (a.trust_score || 0)).slice(0, 1)
      : []
    articlesToRender = selectedCategory === '전체'
      ? categoryFilteredArticles.filter(a => !topPicks.find(t => t.article_id === a.article_id))
      : categoryFilteredArticles
    totalCountForDisplay = categoryTotalCount
  }

  useEffect(() => {
    if (!isLoading) {
      setIsPageLoaded(false)
      const timer = setTimeout(() => setIsPageLoaded(true), 50)
      return () => clearTimeout(timer)
    }
  }, [isLoading, isSearchMode, page, selectedCategory])

  const topPicks = !isSearchMode && selectedCategory === '전체'
    ? [...firstPageArticlesForTopPicks].sort((a, b) => (b.trust_score || 0) - (a.trust_score || 0)).slice(0, 1)
    : []

  const hasNextPage = isSearchMode ? page * 9 < totalCountForDisplay : page * 10 < totalCountForDisplay

  const impressionArticles = [...topPicks, ...articlesToRender]
    .filter((article) => !!article.article_id)
    .map((article, index) => ({
      article_id: article.article_id,
      position: index,
    }))
  const impressionSignature = impressionArticles
    .map((article) => `${article.position}:${article.article_id}`)
    .join('|')
  const impressionSource = isSearchMode ? 'search_results' : 'main_list'
  const impressionContextKey = [
    impressionSource,
    `page=${page}`,
    `category=${selectedCategory}`,
    `query=${submittedQuery || ''}`,
  ].join('|')

  useEffect(() => { window.scrollTo(0, 0) }, [location.search])

  useEffect(() => {
    if (isLoading || isRefreshing || isLoadingFirstPageArticles || hasPrimaryError) return
    trackEvent('view_article_list', null, {
      source: impressionSource,
      context_key: impressionContextKey,
      page,
      category: selectedCategory,
      query: submittedQuery || undefined,
      article_count: impressionArticles.length,
      total_count: totalCountForDisplay,
      article_ids: impressionArticles.map((article) => article.article_id),
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading, isRefreshing, isLoadingFirstPageArticles, hasPrimaryError, impressionSignature, impressionContextKey])

  useEffect(() => {
    if (isLoading || isRefreshing || isLoadingFirstPageArticles || hasPrimaryError) return
    if (impressionArticles.length === 0) return
    trackImpressions(impressionArticles, {
      source: impressionSource,
      context_key: impressionContextKey,
      page,
      category: selectedCategory,
      query: submittedQuery || undefined,
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading, isRefreshing, isLoadingFirstPageArticles, hasPrimaryError, impressionSignature, impressionContextKey])

  const [isLoginOpen, setIsLoginOpen] = useState(false)
  const [isSignupOpen, setIsSignupOpen] = useState(false)
  const [loginEmail, setLoginEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [isLoggedIn, setIsLoggedIn] = useState(() => !!localStorage.getItem('et_user'))
  const [loginError, setLoginError] = useState('')
  const [signupEmail, setSignupEmail] = useState('')
  const [signupPassword, setSignupPassword] = useState('')
  const [signupConfirm, setSignupConfirm] = useState('')
  const [signupError, setSignupError] = useState('')

  const [isAdminAuthOpen, setIsAdminAuthOpen] = useState(false)
  const [isFeedbackAuthOpen, setIsFeedbackAuthOpen] = useState(false)

  const handleFeedbackButtonClick = () => {
    if (isLoggedIn) {
      navigate('/feedback')
    } else {
      setIsFeedbackAuthOpen(true)
    }
  }

  const handleAdminButtonClick = () => {
    if (localStorage.getItem('et_user') === 'etdashboard@naver.com') {
      navigate('/admin')
    } else {
      setIsAdminAuthOpen(true)
    }
  }

  const closeLogin = () => { setIsLoginOpen(false); setLoginEmail(''); setLoginPassword(''); setLoginError('') }
  const closeSignup = () => { setIsSignupOpen(false); setSignupEmail(''); setSignupPassword(''); setSignupConfirm(''); setSignupError('') }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!loginEmail.trim() || !loginPassword.trim()) {
      setLoginError('이메일과 비밀번호를 입력해주세요.')
      return
    }
    if (loginEmail.trim() === 'etdashboard@naver.com' && loginPassword === 'sejong') {
      localStorage.setItem('et_user', 'etdashboard@naver.com')
      setIsLoggedIn(true)
      closeLogin()
      return
    }
    try {
      const res = await login(loginEmail.trim(), loginPassword)
      localStorage.setItem('et_user', res.user_id)
      setIsLoggedIn(true)
      closeLogin()
    } catch (err: unknown) {
      setLoginError((err as { response?: { data?: { detail?: string } } }).response?.data?.detail || '로그인에 실패했습니다.')
    }
  }

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!signupEmail.trim() || !signupPassword.trim() || !signupConfirm.trim()) {
      setSignupError('모든 항목을 입력해주세요.')
      return
    }
    if (signupPassword !== signupConfirm) {
      setSignupError('비밀번호가 일치하지 않습니다.')
      return
    }
    if (signupPassword.length < 6) {
      setSignupError('비밀번호는 6자 이상이어야 합니다.')
      return
    }
    try {
      const res = await signup(signupEmail.trim(), signupPassword)
      localStorage.setItem('et_user', res.user_id)
      setIsLoggedIn(true)
      closeSignup()
    } catch (err: unknown) {
      setSignupError((err as { response?: { data?: { detail?: string } } }).response?.data?.detail || '회원가입에 실패했습니다.')
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('et_user')
    setIsLoggedIn(false)
  }

  return (
    <div className="min-h-screen bg-neutral-50 font-sans antialiased text-gray-900">
      <header className="w-full px-6 py-4 flex items-center justify-start gap-2 bg-neutral-50/90 backdrop-blur-md sticky top-0 z-10">
        <button
          onClick={handleFeedbackButtonClick}
          className="flex items-center justify-center w-12 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105"
          title="피드백 (마이페이지)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="#2563eb" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V2.75a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 0 1-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 0 0-1.302 4.665c0 1.194.232 2.333.654 3.375Z" />
          </svg>
        </button>
        <button
          onClick={handleAdminButtonClick}
          className="flex items-center justify-center w-12 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105"
          title="Admin 설정"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="#2563eb" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.99l1.004.828c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
          </svg>
        </button>

        {/* 로그인 버튼 */}
        <div className="ml-auto flex items-center gap-2">
          {isLoggedIn ? (
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105 px-4"
              title="로그아웃"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="#2563eb" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15M12 9l-3 3m0 0 3 3m-3-3h12.75" />
              </svg>
              <span className="text-[13px] font-bold text-blue-600 hidden sm:inline">{localStorage.getItem('et_user')?.split('@')[0]}</span>
            </button>
          ) : (
            <button
              onClick={() => setIsLoginOpen(true)}
              className="flex items-center justify-center w-12 h-12 bg-blue-50 hover:bg-blue-100 rounded-2xl transition-all shadow-sm border border-blue-200 hover:border-blue-300 hover:scale-105"
              title="로그인"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="#2563eb" className="w-7 h-7">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
              </svg>
            </button>
          )}
        </div>
      </header>

      {/* 로그인 모달 */}
      {isLoginOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.35)', backdropFilter: 'blur(6px)' }}
          onClick={closeLogin}
        >
          <div
            className="relative w-full max-w-sm rounded-3xl overflow-hidden shadow-2xl"
            style={{ background: 'rgba(255,255,255,0.25)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.4)' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-green-400 inline-block shadow-sm" />
                  <h2 className="text-[22px] font-extrabold text-white tracking-tight drop-shadow">ET DashBoard 로그인</h2>
                </div>
                <button
                  onClick={closeLogin}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-white/20 hover:bg-white/40 text-white font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-white/80 text-[14px] font-medium">
                로그인하고 맞춤 뉴스 피드와 신뢰도 분석을 경험해보세요.
              </p>

              <form onSubmit={handleLogin} className="space-y-3 pt-1">
                <input
                  type="email"
                  placeholder="이메일"
                  value={loginEmail}
                  onChange={e => setLoginEmail(e.target.value)}
                  className="w-full bg-white/30 border border-white/40 rounded-xl px-4 py-3 text-[15px] text-white placeholder-white/60 font-medium focus:outline-none focus:bg-white/40 focus:border-white/70 transition-all"
                  style={{ backdropFilter: 'blur(4px)' }}
                />
                <input
                  type="password"
                  placeholder="비밀번호"
                  value={loginPassword}
                  onChange={e => setLoginPassword(e.target.value)}
                  className="w-full bg-white/30 border border-white/40 rounded-xl px-4 py-3 text-[15px] text-white placeholder-white/60 font-medium focus:outline-none focus:bg-white/40 focus:border-white/70 transition-all"
                  style={{ backdropFilter: 'blur(4px)' }}
                />
                {loginError && (
                  <p className="text-red-200 text-[13px] font-semibold px-1">{loginError}</p>
                )}
                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-blue-600/90 hover:bg-blue-600 text-white font-extrabold text-[16px] tracking-tight transition-all hover:scale-[1.02] shadow-lg shadow-blue-900/30"
                >
                  로그인
                </button>
                <button
                  type="button"
                  onClick={() => { closeLogin(); setIsSignupOpen(true) }}
                  className="w-full py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white font-semibold text-[13px] tracking-tight transition-all border border-white/20 hover:border-white/40"
                >
                  회원가입
                </button>
              </form>

              <p className="text-center text-white/60 text-[12px] font-medium pt-1">
                © ET DashBoard — Everyday Trusted News
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 회원가입 모달 */}
      {isSignupOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.35)', backdropFilter: 'blur(6px)' }}
          onClick={closeSignup}
        >
          <div
            className="relative w-full max-w-sm rounded-3xl overflow-hidden shadow-2xl"
            style={{ background: 'rgba(255,255,255,0.25)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.4)' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-green-400 inline-block shadow-sm" />
                  <h2 className="text-[22px] font-extrabold text-white tracking-tight drop-shadow">ET DashBoard 회원가입</h2>
                </div>
                <button
                  onClick={closeSignup}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-white/20 hover:bg-white/40 text-white font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-white/80 text-[14px] font-medium">
                로그인하고 맞춤 뉴스 피드와 신뢰도 분석을 경험해보세요.
              </p>

              <form onSubmit={handleSignup} className="space-y-3 pt-1">
                <input
                  type="email"
                  placeholder="이메일"
                  value={signupEmail}
                  onChange={e => setSignupEmail(e.target.value)}
                  className="w-full bg-white/30 border border-white/40 rounded-xl px-4 py-3 text-[15px] text-white placeholder-white/60 font-medium focus:outline-none focus:bg-white/40 focus:border-white/70 transition-all"
                  style={{ backdropFilter: 'blur(4px)' }}
                />
                <input
                  type="password"
                  placeholder="비밀번호"
                  value={signupPassword}
                  onChange={e => setSignupPassword(e.target.value)}
                  className="w-full bg-white/30 border border-white/40 rounded-xl px-4 py-3 text-[15px] text-white placeholder-white/60 font-medium focus:outline-none focus:bg-white/40 focus:border-white/70 transition-all"
                  style={{ backdropFilter: 'blur(4px)' }}
                />
                <input
                  type="password"
                  placeholder="비밀번호 확인"
                  value={signupConfirm}
                  onChange={e => setSignupConfirm(e.target.value)}
                  className="w-full bg-white/30 border border-white/40 rounded-xl px-4 py-3 text-[15px] text-white placeholder-white/60 font-medium focus:outline-none focus:bg-white/40 focus:border-white/70 transition-all"
                  style={{ backdropFilter: 'blur(4px)' }}
                />
                {signupError && (
                  <p className="text-red-200 text-[13px] font-semibold px-1">{signupError}</p>
                )}
                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-blue-600/90 hover:bg-blue-600 text-white font-extrabold text-[16px] tracking-tight transition-all hover:scale-[1.02] shadow-lg shadow-blue-900/30"
                >
                  회원가입
                </button>
                <button
                  type="button"
                  onClick={() => { closeSignup(); setIsLoginOpen(true) }}
                  className="w-full py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white font-semibold text-[13px] tracking-tight transition-all border border-white/20 hover:border-white/40"
                >
                  이미 계정이 있으신가요? 로그인
                </button>
              </form>

              <p className="text-center text-white/60 text-[12px] font-medium pt-1">
                © ET DashBoard — Everyday Trusted News
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 관리자 접근 불가 모달 */}
      {isAdminAuthOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.45)', backdropFilter: 'blur(6px)' }}
          onClick={() => setIsAdminAuthOpen(false)}
        >
          <div
            className="relative w-full max-w-sm rounded-3xl overflow-hidden shadow-2xl"
            style={{ background: 'rgba(255,255,255,0.18)', backdropFilter: 'blur(22px)', border: '1px solid rgba(255,255,255,0.35)' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-red-400 inline-block shadow-sm" />
                  <h2 className="text-[20px] font-extrabold text-white tracking-tight drop-shadow">관리자 전용 페이지</h2>
                </div>
                <button
                  onClick={() => setIsAdminAuthOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-white/20 hover:bg-white/40 text-white font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-white/85 text-[15px] font-semibold leading-relaxed">
                이 페이지는 관리자만 접근할 수 있습니다.<br />관리자 계정으로 로그인해주세요.
              </p>

              <button
                onClick={() => { setIsAdminAuthOpen(false); setIsLoginOpen(true) }}
                className="w-full py-3 rounded-xl bg-blue-600/80 hover:bg-blue-600 text-white font-extrabold text-[15px] tracking-tight transition-all hover:scale-[1.02] shadow-lg shadow-blue-900/30"
              >
                로그인하기
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 피드백 로그인 안내 모달 */}
      {isFeedbackAuthOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.45)', backdropFilter: 'blur(6px)' }}
          onClick={() => setIsFeedbackAuthOpen(false)}
        >
          <div
            className="relative w-full max-w-sm rounded-3xl overflow-hidden shadow-2xl"
            style={{ background: 'rgba(255,255,255,0.18)', backdropFilter: 'blur(22px)', border: '1px solid rgba(255,255,255,0.35)' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-blue-400 inline-block shadow-sm" />
                  <h2 className="text-[20px] font-extrabold text-white tracking-tight drop-shadow">로그인이 필요합니다</h2>
                </div>
                <button
                  onClick={() => setIsFeedbackAuthOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-white/20 hover:bg-white/40 text-white font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-white/85 text-[15px] font-semibold leading-relaxed">
                피드백 페이지는 로그인 후 이용할 수 있습니다.<br />로그인하고 이용해주세요.
              </p>

              <button
                onClick={() => { setIsFeedbackAuthOpen(false); setIsLoginOpen(true) }}
                className="w-full py-3 rounded-xl bg-blue-600/80 hover:bg-blue-600 text-white font-extrabold text-[15px] tracking-tight transition-all hover:scale-[1.02] shadow-lg shadow-blue-900/30"
              >
                로그인하기
              </button>
            </div>
          </div>
        </div>
      )}

      <main className={`max-w-6xl mx-auto px-4 pb-12 pt-10 space-y-10 transition-all duration-700 ease-out transform ${isPageLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <section className="space-y-6">
          <div className="flex flex-col items-center gap-5 max-w-5xl mx-auto">
            <h1
              className="text-4xl font-black tracking-tight cursor-pointer text-center text-blue-600 select-none"
              onClick={() => handleCategoryClick('전체')}
            >
              ET DashBoard
            </h1>
            <form onSubmit={handleSearch} className="relative w-full max-w-4xl">
              <input
                type="text"
                placeholder="관심있는 뉴스 키워드를 검색해보세요..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                className="w-full bg-white border border-gray-200 rounded-2xl px-6 py-4 text-[16px] font-medium text-gray-900 placeholder-gray-400 shadow-sm focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
              />
              <button type="submit" className="absolute right-2.5 top-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-xl font-semibold text-[15px] transition-all shadow-md shadow-blue-200">
                검색
              </button>
            </form>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 max-w-4xl mx-auto">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => handleCategoryClick(cat)}
                className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all duration-200 ${
                  selectedCategory === cat && !submittedQuery
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-200 scale-105'
                    : 'bg-white text-gray-600 border border-gray-200 hover:border-gray-300 hover:text-gray-900 hover:shadow-sm'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </section>

        <div className="space-y-10">
            {!isSearchMode && selectedCategory === '전체' && (
              <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm min-w-0">
                  <h2 className="text-[17px] font-bold text-gray-800 tracking-tight flex items-center gap-2.5 mb-5">
                    <span className="text-lg">✨</span>
                    <span>AI가 분석한 오늘의 추천 뉴스</span>
                  </h2>
                  {topPicks.length === 0 || isLoadingFirstPageArticles ? (
                    <div className="h-64 rounded-xl border border-blue-100 bg-blue-50/40 animate-pulse" />
                  ) : (
                    <div className="flex flex-col gap-5">
                      {topPicks.map((article) => (
                        <div
                          key={article.article_id}
                          onClick={() => {
                            trackEvent('click_article', article.article_id, { source: 'top_pick' })
                            navigate(`/article/${article.article_id}`)
                          }}
                          className="group cursor-pointer bg-gradient-to-r from-blue-50 to-blue-50/30 border border-blue-100 rounded-xl shadow-sm hover:shadow-lg hover:-translate-y-1 transition-all duration-300 ease-out overflow-hidden"
                        >
                          <div className="h-44 relative overflow-hidden border-b border-blue-100/50 bg-gray-100">
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-400 text-[12px] font-semibold">썸네일 불러오는 중...</div>
                            <img
                              src={getApiAssetUrl(`/api/articles/${article.article_id}/thumbnail`)}
                              alt="기사 썸네일"
                              className="relative z-10 object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out"
                              onError={(e) => { e.currentTarget.src = `https://picsum.photos/seed/${article.article_id}/800/600` }}
                            />
                          </div>
                          <div className="p-5 space-y-3">
                            <div className="flex items-center gap-2">
                              <span className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-[11px] font-bold px-3 py-1 rounded-md shadow-sm tracking-wide">TOP PICK</span>
                              <span
                                className="text-xs font-bold px-2.5 py-1 rounded-md border shadow-sm whitespace-nowrap"
                                style={{
                                  color: `hsl(${getHue(article.trust_score || 0)}, 80%, 35%)`,
                                  backgroundColor: `hsl(${getHue(article.trust_score || 0)}, 100%, 92%)`,
                                  borderColor: `hsl(${getHue(article.trust_score || 0)}, 85%, 85%)`
                                }}
                              >
                                신뢰도 {article.trust_score}점
                              </span>
                            </div>
                            <h3 className="text-[21px] font-extrabold text-gray-900 group-hover:text-blue-600 transition-colors break-keep leading-[1.35] tracking-tight line-clamp-2">{article.title}</h3>
                            <p className="text-[14px] text-gray-600 font-medium line-clamp-3 leading-relaxed break-keep">{article.summary_text || article.chunk_text}</p>
                            <div className="text-[12px] text-gray-400 font-semibold flex gap-2">
                              <span>{article.source}</span><span>·</span><span>{article.published_at?.substring(0, 10)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm min-w-0">
                  <h2 className="text-[17px] font-bold text-gray-800 tracking-tight flex items-center gap-2.5 mb-5">
                    <span className="text-lg">🎯</span>
                    <span>AI가 분석한 당신을 위한 뉴스</span>
                  </h2>
                  {isLoadingPersonalized ? (
                    <div className="space-y-3">
                      {Array.from({ length: 5 }).map((_, index) => (
                        <div key={index} className="h-20 rounded-xl border border-gray-100 bg-gray-50 animate-pulse" />
                      ))}
                    </div>
                  ) : personalizedArticles.length > 0 ? (
                    <div className="divide-y divide-gray-100">
                      {personalizedArticles.slice(0, 5).map((article, index) => (
                        <button
                          key={article.article_id}
                          type="button"
                          onClick={() => {
                            trackEvent('click_article', article.article_id, { source: 'personalized_recommendation', rank: index + 1 })
                            navigate(`/article/${article.article_id}`)
                          }}
                          className="group w-full text-left py-3.5 first:pt-0 last:pb-0 flex gap-3 hover:bg-blue-50/50 rounded-lg transition-colors"
                        >
                          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-blue-600 text-white text-[12px] font-extrabold">
                            {index + 1}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-[14px] font-extrabold text-gray-900 leading-snug break-keep line-clamp-2 group-hover:text-blue-600 transition-colors">
                              {article.title}
                            </span>
                            <span className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] font-semibold text-gray-400">
                              <span>{article.source}</span>
                              <span>·</span>
                              <span>{article.published_at?.substring(0, 10)}</span>
                              {article.trust_score > 0 && (
                                <>
                                  <span>·</span>
                                  <span className="text-blue-600">신뢰도 {article.trust_score}점</span>
                                </>
                              )}
                            </span>
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="h-64 rounded-xl border border-dashed border-gray-200 bg-gray-50/70 flex items-center justify-center text-[14px] font-semibold text-gray-400">
                      추천할 기사를 준비하는 중입니다.
                    </div>
                  )}
                </div>
              </section>
            )}

            <section className="space-y-5">
              <h2 className="text-[16px] font-bold text-gray-700 tracking-tight border-b border-gray-200 pb-3 flex items-center gap-2">
                {submittedQuery ? `"${submittedQuery}" 검색 결과` : selectedCategory !== '전체' ? `${selectedCategory} 관련 기사` : '최신 기사'}
                {isRefreshing && !isLoading && (
                  <span className="text-[12px] font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">업데이트 중</span>
                )}
                {(isSearchMode || selectedCategory !== '전체') && (
                  <span className="text-[13px] font-semibold text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded-full ml-1">{totalCountForDisplay}건</span>
                )}
              </h2>

              {isLoading ? (
                <ArticleListSkeleton />
              ) : hasPrimaryError ? (
                <div className="bg-white rounded-2xl border border-red-100 p-10 text-center">
                  <p className="text-gray-900 font-bold tracking-tight mb-2">기사를 불러오지 못했습니다.</p>
                  <p className="text-gray-500 text-[14px] font-medium mb-5">네트워크나 서버 상태를 확인한 뒤 다시 시도해주세요.</p>
                  <button
                    onClick={() => window.location.reload()}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-xl font-semibold text-[14px] transition-all shadow-sm"
                  >
                    다시 시도
                  </button>
                </div>
              ) : articlesToRender.length > 0 ? (
                <div className="flex flex-col gap-3 w-full">
                  {articlesToRender.map((article) => (
                    <div
                      key={article.article_id}
                      className="h-full transition-transform duration-300 ease-out hover:-translate-y-1"
                      onClickCapture={() => trackEvent('click_article', article.article_id, { source: 'main_list' })}
                    >
                      <ArticleCard article={article} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-white/80 rounded-2xl border border-dashed border-gray-200 p-10 text-center">
                  <p className="text-gray-900 font-bold tracking-tight mb-2">
                    {isSearchMode ? '검색 결과가 없습니다.' : '아직 표시할 기사가 없습니다.'}
                  </p>
                  <p className="text-gray-500 text-[14px] font-medium mb-5">
                    {isSearchMode ? '검색어를 바꾸거나 필터를 초기화해보세요.' : '관리자 화면에서 최신 기사를 수집할 수 있습니다.'}
                  </p>
                  <button
                    onClick={() => isSearchMode || selectedCategory !== '전체' ? handleCategoryClick('전체') : navigate('/admin')}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-xl font-semibold text-[14px] transition-all shadow-sm"
                  >
                    {isSearchMode || selectedCategory !== '전체' ? '필터 초기화' : '관리자 화면으로 이동'}
                  </button>
                </div>
              )}

              {(!isSearchMode || (searchResults?.length ?? 0) > 0) && (
                <div className="flex justify-center gap-3 pt-6 border-t border-gray-200">
                  <button onClick={() => updatePage(Math.max(1, page - 1))} disabled={page === 1} className="text-[14px] font-semibold px-5 py-2 border border-gray-200 rounded-xl disabled:opacity-40 disabled:pointer-events-none bg-white hover:bg-gray-50 text-gray-700 active:bg-gray-900 active:text-white active:scale-95 transition-all duration-200 shadow-sm">이전</button>
                  <span className="text-[14px] text-gray-500 flex items-center px-3 font-semibold">{page} 페이지</span>
                  <button onClick={() => updatePage(page + 1)} disabled={!hasNextPage} className="text-[14px] font-semibold px-5 py-2 border border-gray-200 rounded-xl disabled:opacity-40 disabled:pointer-events-none bg-white hover:bg-gray-50 text-gray-700 active:bg-gray-900 active:text-white active:scale-95 transition-all duration-200 shadow-sm">다음</button>
                </div>
              )}
            </section>
          </div>
      </main>
    </div>
  )
}
