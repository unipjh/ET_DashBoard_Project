import { useState, useEffect, useRef } from 'react'
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { fetchArticles, searchArticles, signup, login, getApiAssetUrl, fetchRecommendations, fetchUserInfo, deleteAccount } from '../api/client'
import type { Article, SearchResult } from '../api/client'
import ArticleCard from '../components/ArticleCard'
import { getSessionId, trackEvent, trackImpressions } from '../api/logger'

const FEEDBACK_EMAIL = 'your-email@example.com' // 피드백 수신 이메일 주소로 변경하세요

const CATEGORIES = ['전체', '정치', '경제', '사회', '생활/문화', '세계', 'IT/과학']

const getHue = (s: number) => {
  if (s <= 30) return (s / 30) * 15
  if (s <= 60) return 15 + ((s - 30) / 30) * 35
  return 50 + ((s - 60) / 40) * 90
}

const ArticleListSkeleton = () => (
  <div className="flex flex-col gap-3 w-full" aria-label="기사 목록 로딩 중">
    {Array.from({ length: 6 }).map((_, index) => (
      <div key={index} className="bg-white border border-gray-200 rounded-xl p-5">
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
  const [isRefreshFeedActive, setIsRefreshFeedActive] = useState(false)
  const [randomRefreshArticles, setRandomRefreshArticles] = useState<Article[]>([])
  const articleListRef = useRef<HTMLDivElement>(null)

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
  const isAdmin = currentUserId === 'etdashboard@naver.com'

  const SEARCH_HISTORY_LIMIT = 10
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const [isSearchFocused, setIsSearchFocused] = useState(false)
  const searchBoxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!currentUserId) { setSearchHistory([]); return }
    try {
      const stored: unknown = JSON.parse(localStorage.getItem(`search_history_${currentUserId}`) || '[]')
      setSearchHistory(Array.isArray(stored) ? stored.filter((v): v is string => typeof v === 'string') : [])
    } catch {
      setSearchHistory([])
    }
  }, [currentUserId])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchBoxRef.current && !searchBoxRef.current.contains(e.target as Node)) {
        setIsSearchFocused(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const saveSearchHistory = (term: string) => {
    if (!currentUserId || !term.trim()) return
    const trimmed = term.trim()
    setSearchHistory(prev => {
      const next = [trimmed, ...prev.filter(h => h !== trimmed)].slice(0, SEARCH_HISTORY_LIMIT)
      localStorage.setItem(`search_history_${currentUserId}`, JSON.stringify(next))
      return next
    })
  }

  const runSearch = (term: string) => {
    const trimmed = term.trim()
    const newParams = new URLSearchParams(searchParams)
    if (trimmed) {
      trackEvent('execute_search', null, { query: trimmed })
      newParams.set('q', trimmed)
      newParams.delete('cat')
      saveSearchHistory(trimmed)
    } else {
      newParams.delete('q')
    }
    newParams.set('page', '1')
    setSearchParams(newParams)
    setIsSearchFocused(false)
  }

  const handleHistoryClick = (term: string) => {
    setQuery(term)
    runSearch(term)
  }

  const removeSearchHistoryItem = (term: string) => {
    if (!currentUserId) return
    setSearchHistory(prev => {
      const next = prev.filter(h => h !== term)
      localStorage.setItem(`search_history_${currentUserId}`, JSON.stringify(next))
      return next
    })
  }
  const { data: personalizedArticles = [], isLoading: isLoadingPersonalized } = useQuery({
    queryKey: ['recommendations', sessionId, currentUserId, selectedCategory, location.key],
    queryFn: () => fetchRecommendations(sessionId, currentUserId, 10),
    enabled: !isSearchMode && selectedCategory === '전체' && !isRefreshFeedActive,
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
    runSearch(query)
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
    articleListRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
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

  useEffect(() => { window.scrollTo(0, 0) }, [selectedCategory, submittedQuery])

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

  const queryClient = useQueryClient()

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


  // 유저 드롭다운 메뉴
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isAboutOpen, setIsAboutOpen] = useState(false)
  const [isSendFeedbackOpen, setIsSendFeedbackOpen] = useState(false)
  const [userJoinDate, setUserJoinDate] = useState<string | null>(null)
  const [deletePassword, setDeletePassword] = useState('')
  const [deleteError, setDeleteError] = useState('')
  const [deleteConfirmMode, setDeleteConfirmMode] = useState(false)
  const [feedbackText, setFeedbackText] = useState('')
  const userMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setIsUserMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  useEffect(() => {
    if (isSettingsOpen && currentUserId) {
      fetchUserInfo(currentUserId).then(info => setUserJoinDate(info.created_at.substring(0, 10))).catch(() => {})
    }
  }, [isSettingsOpen, currentUserId])

  const handleRefreshFeed = () => {
    setIsRefreshFeedActive(true)
    queryClient.removeQueries({ queryKey: ['recommendations'] })
    const shuffled = [...firstPageArticlesForTopPicks]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    setRandomRefreshArticles(shuffled.slice(0, 5))
    setIsUserMenuOpen(false)
  }

  const handleDeleteAccount = async () => {
    if (!currentUserId || !deletePassword) { setDeleteError('비밀번호를 입력해주세요.'); return }
    try {
      await deleteAccount(currentUserId, deletePassword)
      localStorage.removeItem('et_user')
      setIsLoggedIn(false)
      setIsSettingsOpen(false)
      setDeleteConfirmMode(false)
      setDeletePassword('')
      setDeleteError('')
    } catch {
      setDeleteError('비밀번호가 올바르지 않습니다.')
    }
  }

  const handleSendFeedback = () => {
    if (!feedbackText.trim()) return
    const subject = encodeURIComponent('ET DashBoard 피드백')
    const body = encodeURIComponent(feedbackText)
    window.open(`mailto:${FEEDBACK_EMAIL}?subject=${subject}&body=${body}`)
    setFeedbackText('')
    setIsSendFeedbackOpen(false)
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
    <div className="min-h-screen bg-paper font-sans antialiased text-gray-900">
      <header className="w-full px-6 py-4 flex items-center gap-2 bg-paper/90 backdrop-blur-md sticky top-0 z-10">
        {isSearchMode && (
          <button
            onClick={() => navigate(-1)}
            className="flex items-center justify-center w-11 h-11 bg-white hover:bg-gray-50 rounded-full transition-colors border border-gray-200 hover:border-navy-300"
            title="뒤로가기"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="#2C4460" className="w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
            </svg>
          </button>
        )}

        {/* 로그인 / 유저 메뉴 */}
        <div className="ml-auto flex items-center gap-2">
          {isLoggedIn ? (
            <div className="relative" ref={userMenuRef}>
              <button
                onClick={() => setIsUserMenuOpen(v => !v)}
                className="flex items-center gap-2 h-11 bg-white hover:bg-gray-50 rounded-full transition-colors border border-gray-200 hover:border-navy-300 px-3.5"
              >
                <div className="w-7 h-7 rounded-full bg-navy-600 flex items-center justify-center text-white text-[13px] font-bold shrink-0">
                  {currentUserId?.charAt(0).toUpperCase()}
                </div>
                <span className="text-[13px] font-bold text-navy-600 hidden sm:inline">{currentUserId?.split('@')[0]}</span>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="#2C4460" className={`w-4 h-4 hidden sm:block transition-transform duration-200 ${isUserMenuOpen ? 'rotate-180' : ''}`}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
                </svg>
              </button>

              {isUserMenuOpen && (
                <div className="absolute right-0 top-13 w-72 rounded-xl shadow-lg z-50 overflow-hidden border border-gray-200 bg-white animate-in fade-in slide-in-from-top-2 duration-150">
                  {/* 유저 헤더 */}
                  <div className="px-4 py-3.5 border-b border-gray-100">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center text-white font-bold text-base shrink-0">
                        {currentUserId?.charAt(0).toUpperCase()}
                      </div>
                      <div className="min-w-0">
                        <p className="font-bold text-gray-900 text-sm truncate">{currentUserId?.split('@')[0]}</p>
                        <p className="text-gray-400 text-xs mt-0.5 truncate">{currentUserId}</p>
                      </div>
                    </div>
                  </div>

                  {/* 메뉴 아이템 1 */}
                  <div className="py-1">
                    <button onClick={() => { setIsSettingsOpen(true); setIsUserMenuOpen(false) }}
                      className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.99l1.004.828c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                      </svg>
                      Settings
                    </button>
                    <button onClick={handleRefreshFeed}
                      className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                      </svg>
                      Refresh feed
                    </button>
                  </div>

                  {/* 피드백 페이지 */}
                  <div className="border-t border-gray-100 py-1">
                    <button onClick={() => { navigate('/feedback'); setIsUserMenuOpen(false) }}
                      className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V2.75a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 0 1-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 0 0-1.302 4.665c0 1.194.232 2.333.654 3.375Z" />
                      </svg>
                      Feedback page
                    </button>
                  </div>

                  {/* 관리자 전용 메뉴 */}
                  {isAdmin && (
                    <div className="border-t border-gray-100 py-1">
                      <button onClick={() => { navigate('/admin'); setIsUserMenuOpen(false) }}
                        className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.99l1.004.828c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                        </svg>
                        Admin page
                      </button>
                      <button onClick={() => { navigate('/log'); setIsUserMenuOpen(false) }}
                        className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.5 7.5 9l3.5 3.5L18 5m0 0h-4.5M18 5v4.5M4.5 19.5h15a.75.75 0 0 0 .75-.75V6.75" />
                        </svg>
                        Log page
                      </button>
                    </div>
                  )}

                  {/* 메뉴 아이템 2 */}
                  <div className="border-t border-gray-100 py-1">
                    <button onClick={() => { setIsAboutOpen(true); setIsUserMenuOpen(false) }}
                      className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
                      </svg>
                      About
                    </button>
                    <button onClick={() => { setIsSendFeedbackOpen(true); setIsUserMenuOpen(false) }}
                      className="w-full text-left px-4 py-2.5 border border-transparent hover:border-navy-300 hover:bg-gray-50 flex items-center gap-3 text-[14px] font-medium text-gray-700 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-gray-400 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 0 1-2.25 2.25h-15a2.25 2.25 0 0 1-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0 0 19.5 4.5h-15a2.25 2.25 0 0 0-2.25 2.25m19.5 0v.243a2.25 2.25 0 0 1-1.07 1.916l-7.5 4.615a2.25 2.25 0 0 1-2.36 0L3.32 8.91a2.25 2.25 0 0 1-1.07-1.916V6.75" />
                      </svg>
                      Send feedback
                    </button>
                  </div>

                  {/* 로그아웃 */}
                  <div className="border-t border-gray-100 py-1">
                    <button onClick={handleLogout}
                      className="w-full text-left px-4 py-2.5 hover:bg-red-50 flex items-center gap-3 text-[14px] font-medium text-red-500 transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 shrink-0">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15M12 9l-3 3m0 0 3 3m-3-3h12.75" />
                      </svg>
                      Sign out
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() => setIsLoginOpen(true)}
              className="flex items-center justify-center w-11 h-11 bg-white hover:bg-gray-50 rounded-full transition-colors border border-gray-200 hover:border-navy-300"
              title="로그인"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="#2C4460" className="w-7 h-7">
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
            className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-xl bg-white border border-gray-200"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-green-400 inline-block shadow-sm" />
                  <h2 className="text-[22px] font-extrabold text-gray-900 tracking-tight">ET DashBoard 로그인</h2>
                </div>
                <button
                  onClick={closeLogin}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-gray-500 text-[14px] font-medium">
                로그인하고 맞춤 뉴스 피드와 신뢰도 분석을 경험해보세요.
              </p>

              <form onSubmit={handleLogin} className="space-y-3 pt-1">
                <input
                  type="email"
                  placeholder="이메일"
                  value={loginEmail}
                  onChange={e => setLoginEmail(e.target.value)}
                  className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all"
                />
                <input
                  type="password"
                  placeholder="비밀번호"
                  value={loginPassword}
                  onChange={e => setLoginPassword(e.target.value)}
                  className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all"
                />
                {loginError && (
                  <p className="text-red-500 text-[13px] font-semibold px-1">{loginError}</p>
                )}
                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-navy-600 hover:bg-navy-700 text-white font-extrabold text-[16px] tracking-tight transition-all"
                >
                  로그인
                </button>
                <button
                  type="button"
                  onClick={() => { closeLogin(); setIsSignupOpen(true) }}
                  className="w-full py-2 rounded-xl bg-gray-50 hover:bg-gray-100 text-gray-500 hover:text-gray-700 font-semibold text-[13px] tracking-tight transition-all border border-gray-200"
                >
                  회원가입
                </button>
              </form>

              <p className="text-center text-gray-400 text-[12px] font-medium pt-1">
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
            className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-xl bg-white border border-gray-200"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-green-400 inline-block shadow-sm" />
                  <h2 className="text-[22px] font-extrabold text-gray-900 tracking-tight">ET DashBoard 회원가입</h2>
                </div>
                <button
                  onClick={closeSignup}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 font-bold transition-all text-lg"
                >
                  ✕
                </button>
              </div>

              <p className="text-gray-500 text-[14px] font-medium">
                로그인하고 맞춤 뉴스 피드와 신뢰도 분석을 경험해보세요.
              </p>

              <form onSubmit={handleSignup} className="space-y-3 pt-1">
                <input
                  type="email"
                  placeholder="이메일"
                  value={signupEmail}
                  onChange={e => setSignupEmail(e.target.value)}
                  className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all"
                />
                <input
                  type="password"
                  placeholder="비밀번호"
                  value={signupPassword}
                  onChange={e => setSignupPassword(e.target.value)}
                  className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all"
                />
                <input
                  type="password"
                  placeholder="비밀번호 확인"
                  value={signupConfirm}
                  onChange={e => setSignupConfirm(e.target.value)}
                  className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all"
                />
                {signupError && (
                  <p className="text-red-500 text-[13px] font-semibold px-1">{signupError}</p>
                )}
                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-navy-600 hover:bg-navy-700 text-white font-extrabold text-[16px] tracking-tight transition-all"
                >
                  회원가입
                </button>
                <button
                  type="button"
                  onClick={() => { closeSignup(); setIsLoginOpen(true) }}
                  className="w-full py-2 rounded-xl bg-gray-50 hover:bg-gray-100 text-gray-500 hover:text-gray-700 font-semibold text-[13px] tracking-tight transition-all border border-gray-200"
                >
                  이미 계정이 있으신가요? 로그인
                </button>
              </form>

              <p className="text-center text-gray-400 text-[12px] font-medium pt-1">
                © ET DashBoard — Everyday Trusted News
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Settings 모달 ── */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.45)', backdropFilter: 'blur(6px)' }}
          onClick={() => { setIsSettingsOpen(false); setDeleteConfirmMode(false); setDeletePassword(''); setDeleteError('') }}>
          <div className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-xl bg-white border border-gray-200"
            onClick={e => e.stopPropagation()}>
            <div className="px-8 pt-8 pb-10 space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-[20px] font-extrabold text-gray-900 tracking-tight">설정</h2>
                <button onClick={() => { setIsSettingsOpen(false); setDeleteConfirmMode(false); setDeletePassword(''); setDeleteError('') }}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 font-bold transition-all text-lg">✕</button>
              </div>

              {/* 가입일 */}
              <div className="space-y-1">
                <p className="text-gray-400 text-[12px] font-bold uppercase tracking-widest">가입일</p>
                <p className="text-gray-900 font-semibold text-[15px]">{userJoinDate || '—'}</p>
              </div>

              {/* 회원 탈퇴 */}
              {currentUserId !== 'etdashboard@naver.com' && (
                <div className="space-y-3 pt-2 border-t border-gray-100">
                  {!deleteConfirmMode ? (
                    <button onClick={() => setDeleteConfirmMode(true)}
                      className="w-full py-2.5 rounded-xl bg-red-50 hover:bg-red-100 text-red-500 font-bold text-[14px] border border-red-200 transition-all">
                      회원 탈퇴
                    </button>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-red-500 text-[13px] font-semibold">탈퇴 시 모든 데이터가 삭제되며 복구할 수 없습니다.</p>
                      <input type="password" placeholder="비밀번호 입력" value={deletePassword}
                        onChange={e => setDeletePassword(e.target.value)}
                        className="w-full bg-white border border-gray-200 rounded-xl px-4 py-2.5 text-[14px] text-gray-900 placeholder-gray-400 focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all" />
                      {deleteError && <p className="text-red-500 text-[12px] font-semibold">{deleteError}</p>}
                      <div className="flex gap-2">
                        <button onClick={() => { setDeleteConfirmMode(false); setDeletePassword(''); setDeleteError('') }}
                          className="flex-1 py-2.5 rounded-xl bg-gray-50 hover:bg-gray-100 text-gray-500 font-bold text-[13px] border border-gray-200 transition-all">
                          취소
                        </button>
                        <button onClick={handleDeleteAccount}
                          className="flex-1 py-2.5 rounded-xl bg-red-500 hover:bg-red-600 text-white font-bold text-[13px] transition-all">
                          탈퇴 확인
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <p className="text-center text-gray-400 text-[12px] font-medium pt-1">
                © ET DashBoard — Everyday Trusted News
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── About 모달 ── */}
      {isAboutOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.45)', backdropFilter: 'blur(6px)' }}
          onClick={() => setIsAboutOpen(false)}>
          <div className="relative w-full max-w-md rounded-2xl overflow-hidden shadow-xl bg-white border border-gray-200"
            onClick={e => e.stopPropagation()}>
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-3 h-3 rounded-full bg-navy-400 inline-block" />
                  <h2 className="text-[20px] font-extrabold text-gray-900 tracking-tight">서비스 소개</h2>
                </div>
                <button onClick={() => setIsAboutOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 font-bold transition-all text-lg">✕</button>
              </div>

              <div className="space-y-4">
                <div className="bg-gray-50 rounded-2xl p-4 space-y-1.5 border border-gray-100">
                  <p className="text-gray-900 font-extrabold text-[15px]">📰 ET DashBoard란?</p>
                  <p className="text-gray-500 text-[13px] leading-relaxed font-medium">
                    매일 쏟아지는 뉴스 속에서 <span className="text-gray-900 font-bold">신뢰할 수 있는 정보</span>를 빠르게 찾을 수 있도록 만든 AI 뉴스 대시보드입니다. 클릭베이트와 편향된 기사에서 벗어나 팩트 기반의 뉴스를 제공합니다.
                  </p>
                </div>

                <div className="bg-gray-50 rounded-2xl p-4 space-y-1.5 border border-gray-100">
                  <p className="text-gray-900 font-extrabold text-[15px]">🔐 신뢰도 점수란?</p>
                  <p className="text-gray-500 text-[13px] leading-relaxed font-medium">
                    AI가 기사를 <span className="text-gray-900 font-bold">사실 확인·출처 신뢰성·균형성·선정성</span> 등 여러 기준으로 분석해 0~100점으로 수치화한 점수입니다. 점수가 높을수록 믿을 수 있는 기사입니다.
                  </p>
                </div>

                <div className="bg-gray-50 rounded-2xl p-4 space-y-1.5 border border-gray-100">
                  <p className="text-gray-900 font-extrabold text-[15px]">🎯 추천 뉴스는 어떻게 작동하나요?</p>
                  <p className="text-gray-500 text-[13px] leading-relaxed font-medium">
                    읽은 기사·검색 패턴·반응을 학습해 <span className="text-gray-900 font-bold">개인 맞춤형 뉴스</span>를 추천합니다. 로그인 없이도 세션 기반으로 작동합니다.
                  </p>
                </div>
              </div>

              <p className="text-center text-gray-400 text-[12px] font-medium pt-1">
                © ET DashBoard — Everyday Trusted News
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Send Feedback 모달 ── */}
      {isSendFeedbackOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(30,41,59,0.45)', backdropFilter: 'blur(6px)' }}
          onClick={() => setIsSendFeedbackOpen(false)}>
          <div className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-xl bg-white border border-gray-200"
            onClick={e => e.stopPropagation()}>
            <div className="px-8 pt-8 pb-10 space-y-5">
              <div className="flex items-center justify-between">
                <h2 className="text-[20px] font-extrabold text-gray-900 tracking-tight">피드백 보내기</h2>
                <button onClick={() => setIsSendFeedbackOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 font-bold transition-all text-lg">✕</button>
              </div>

              <textarea
                value={feedbackText}
                onChange={e => setFeedbackText(e.target.value)}
                placeholder="ET DashBoard에 대한 의견이나 개선점을 알려주세요..."
                rows={5}
                className="w-full bg-white border border-gray-200 rounded-xl px-4 py-3 text-[14px] text-gray-900 placeholder-gray-400 font-medium focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-all resize-none"
              />

              <button onClick={handleSendFeedback} disabled={!feedbackText.trim()}
                className="w-full py-3 rounded-xl bg-navy-600 hover:bg-navy-700 disabled:opacity-40 disabled:pointer-events-none text-white font-extrabold text-[15px] tracking-tight transition-all">
                보내기
              </button>

              <p className="text-center text-gray-400 text-[12px]">이메일 클라이언트가 열립니다</p>
            </div>
          </div>
        </div>
      )}

      <main className={`max-w-6xl mx-auto px-4 pb-12 pt-10 space-y-10 transition-all duration-700 ease-out transform ${isPageLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <section className="space-y-6">
          <div className="flex flex-col items-center gap-5 max-w-5xl mx-auto">
            <h1
              className="logo-oval text-4xl md:text-5xl font-extrabold tracking-tight cursor-pointer text-center text-navy-600 select-none -mt-3"
              onClick={() => handleCategoryClick('전체')}
            >
              ET DashBoard
            </h1>
            <div className="relative w-full max-w-4xl" ref={searchBoxRef}>
              <form onSubmit={handleSearch} className="relative w-full">
                <input
                  type="text"
                  placeholder="관심있는 뉴스 키워드를 검색해보세요..."
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onFocus={() => setIsSearchFocused(true)}
                  autoComplete="off"
                  className="w-full bg-white border border-gray-200 rounded-full px-6 py-4 text-[16px] font-medium text-gray-900 placeholder-gray-400 hover:border-navy-300 focus:outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100 transition-colors"
                />
                <button type="submit" className="absolute right-2 top-2 bg-navy-600 hover:bg-navy-700 text-white px-5 py-2 rounded-full font-semibold text-[15px] transition-colors">
                  검색
                </button>
              </form>

              {isSearchFocused && currentUserId && searchHistory.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-gray-200 rounded-2xl shadow-lg z-20 overflow-hidden py-2">
                  <p className="px-5 pb-1.5 text-[11px] font-bold text-gray-400 uppercase tracking-widest">최근 검색어</p>
                  {searchHistory.map((term, idx) => (
                    <div
                      key={`${term}-${idx}`}
                      className="group w-full flex items-center gap-2.5 pl-5 pr-2 py-2 hover:bg-gray-50 transition-colors"
                    >
                      <button
                        type="button"
                        onClick={() => handleHistoryClick(term)}
                        className="flex-1 min-w-0 flex items-center gap-2.5 text-left text-[14px] font-medium text-gray-700"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4 text-gray-300 shrink-0">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                        </svg>
                        <span className="truncate">{term}</span>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); removeSearchHistoryItem(term) }}
                        title="검색어 삭제"
                        className="shrink-0 w-6 h-6 flex items-center justify-center rounded-full text-gray-300 hover:text-gray-600 hover:bg-gray-200 transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-3.5 h-3.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 max-w-4xl mx-auto">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => handleCategoryClick(cat)}
                className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-colors duration-200 ${
                  selectedCategory === cat && !submittedQuery
                    ? 'bg-navy-600 text-white'
                    : 'bg-white text-gray-600 border border-gray-200 hover:border-navy-300 hover:text-gray-900'
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
                <div className="bg-white rounded-xl border border-gray-200 p-6 min-w-0">
                  <h2 className="text-[14px] font-semibold text-gray-500 tracking-tight border-b border-gray-200 pb-3 mb-5">
                    오늘의 추천 뉴스
                  </h2>
                  {topPicks.length === 0 || isLoadingFirstPageArticles ? (
                    <div className="h-64 rounded-lg border border-gray-100 bg-gray-50 animate-pulse" />
                  ) : (
                    <div className="flex flex-col gap-5">
                      {topPicks.map((article) => (
                        <div
                          key={article.article_id}
                          onClick={() => {
                            trackEvent('click_article', article.article_id, { source: 'top_pick' })
                            navigate(`/article/${article.article_id}`)
                          }}
                          className="group cursor-pointer bg-white border border-gray-200 rounded-lg hover:border-navy-300 transition-colors duration-200 overflow-hidden"
                        >
                          <div className="h-44 relative overflow-hidden border-b border-gray-200 bg-gray-100">
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-400 text-[12px] font-semibold">썸네일 불러오는 중...</div>
                            <img
                              src={getApiAssetUrl(`/api/articles/${article.article_id}/thumbnail`)}
                              alt="기사 썸네일"
                              className="relative z-10 object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out"
                              onError={(e) => { e.currentTarget.src = `https://picsum.photos/seed/${article.article_id}/800/600` }}
                            />
                          </div>
                          <div className="p-5 space-y-3">
                            <div className="flex items-center gap-2.5">
                              <span className="text-navy-600 text-[11px] font-bold uppercase tracking-widest border-b-2 border-navy-600 pb-0.5">Top Pick</span>
                              <span
                                className="ml-auto text-xs font-bold whitespace-nowrap"
                                style={{ color: `hsl(${getHue(article.trust_score || 0)}, 70%, 38%)` }}
                              >
                                신뢰도 {article.trust_score}점
                              </span>
                            </div>
                            <h3 className="text-[21px] font-extrabold text-gray-900 group-hover:text-navy-600 transition-colors break-keep leading-[1.35] tracking-tight line-clamp-2">{article.title}</h3>
                            <p className="text-[14px] text-gray-600 font-medium line-clamp-3 leading-relaxed break-keep">{article.summary_text || article.chunk_text}</p>
                            <div className="text-[12px] text-gray-400 font-semibold flex items-center gap-2">
                              <span>{article.source}</span><span className="text-gray-300">|</span><span>{article.published_at?.substring(0, 10)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="bg-white rounded-xl border border-gray-200 p-6 min-w-0 flex flex-col">
                  <h2 className="text-[14px] font-semibold text-gray-500 tracking-tight border-b border-gray-200 pb-3 mb-5 flex items-center gap-2.5 shrink-0">
                    <span>{currentUserId ? `${currentUserId.split('@')[0]}를 위한 뉴스` : '당신을 위한 뉴스'}</span>
                    {isRefreshFeedActive && (
                      <button
                        onClick={() => setIsRefreshFeedActive(false)}
                        className="ml-auto text-[11px] font-semibold normal-case tracking-normal text-navy-600 hover:text-navy-800 underline underline-offset-2 shrink-0"
                      >
                        이전 추천으로 되돌리기
                      </button>
                    )}
                  </h2>
                  {isLoadingPersonalized && !isRefreshFeedActive ? (
                    <div className="flex flex-col flex-1 justify-between gap-3">
                      {Array.from({ length: 5 }).map((_, index) => (
                        <div key={index} className="flex-1 rounded-lg border border-gray-100 bg-gray-50 animate-pulse" />
                      ))}
                    </div>
                  ) : (isRefreshFeedActive ? randomRefreshArticles : personalizedArticles).slice(0, 5).length > 0 ? (
                    <div className="flex flex-col flex-1 divide-y divide-gray-100">
                      {(isRefreshFeedActive ? randomRefreshArticles : personalizedArticles).slice(0, 5).map((article, index) => (
                        <button
                          key={article.article_id}
                          type="button"
                          onClick={() => {
                            trackEvent('click_article', article.article_id, { source: isRefreshFeedActive ? 'latest_feed' : 'personalized_recommendation', rank: index + 1 })
                            navigate(`/article/${article.article_id}`)
                          }}
                          className="group w-full text-left flex-1 flex gap-3 items-center ring-1 ring-transparent hover:ring-navy-300 hover:bg-gray-50 rounded-lg transition-colors duration-150 px-1"
                        >
                          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-navy-600 text-white text-[12px] font-extrabold">
                            {index + 1}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-[14px] font-extrabold text-gray-900 leading-snug break-keep line-clamp-2 group-hover:text-navy-600 transition-colors">
                              {article.title}
                            </span>
                            <span className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] font-semibold text-gray-400">
                              <span>{article.source}</span>
                              <span className="text-gray-300">|</span>
                              <span>{article.published_at?.substring(0, 10)}</span>
                            </span>
                          </span>
                          {article.trust_score > 0 && (
                            <span
                              className="ml-auto shrink-0 text-xs font-bold whitespace-nowrap"
                              style={{ color: `hsl(${getHue(article.trust_score)}, 70%, 38%)` }}
                            >
                              신뢰도 {article.trust_score}점
                            </span>
                          )}
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

            <section className="space-y-5" ref={articleListRef}>
              <h2 className="text-[14px] font-semibold text-gray-500 tracking-tight border-b border-gray-200 pb-3 flex items-center gap-2">
                {submittedQuery ? `"${submittedQuery}" 검색 결과` : selectedCategory !== '전체' ? `${selectedCategory} 관련 기사` : '최신 기사'}
                {isRefreshing && !isLoading && (
                  <span className="text-[12px] font-semibold text-navy-600 bg-navy-50 px-2 py-0.5 rounded-full">업데이트 중</span>
                )}
                {(isSearchMode || selectedCategory !== '전체') && (
                  <span className="text-[13px] font-semibold text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded-full ml-1">{totalCountForDisplay}건</span>
                )}
              </h2>

              {isLoading ? (
                <ArticleListSkeleton />
              ) : hasPrimaryError ? (
                <div className="bg-white rounded-xl border border-red-100 p-10 text-center">
                  <p className="text-gray-900 font-bold tracking-tight mb-2">기사를 불러오지 못했습니다.</p>
                  <p className="text-gray-500 text-[14px] font-medium mb-5">네트워크나 서버 상태를 확인한 뒤 다시 시도해주세요.</p>
                  <button
                    onClick={() => window.location.reload()}
                    className="bg-navy-600 hover:bg-navy-700 text-white px-5 py-2 rounded-full font-semibold text-[14px] transition-colors"
                  >
                    다시 시도
                  </button>
                </div>
              ) : articlesToRender.length > 0 ? (
                <div className="flex flex-col gap-3 w-full">
                  {articlesToRender.map((article) => (
                    <div
                      key={article.article_id}
                      className="h-full"
                      onClickCapture={() => trackEvent('click_article', article.article_id, { source: 'main_list' })}
                    >
                      <ArticleCard article={article} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-white/80 rounded-xl border border-dashed border-gray-200 p-10 text-center">
                  <p className="text-gray-900 font-bold tracking-tight mb-2">
                    {isSearchMode ? '검색 결과가 없습니다.' : '아직 표시할 기사가 없습니다.'}
                  </p>
                  <p className="text-gray-500 text-[14px] font-medium mb-5">
                    {isSearchMode ? '검색어를 바꾸거나 필터를 초기화해보세요.' : '관리자 화면에서 최신 기사를 수집할 수 있습니다.'}
                  </p>
                  <button
                    onClick={() => isSearchMode || selectedCategory !== '전체' ? handleCategoryClick('전체') : navigate('/admin')}
                    className="bg-navy-600 hover:bg-navy-700 text-white px-5 py-2 rounded-full font-semibold text-[14px] transition-colors"
                  >
                    {isSearchMode || selectedCategory !== '전체' ? '필터 초기화' : '관리자 화면으로 이동'}
                  </button>
                </div>
              )}

              {(!isSearchMode || (searchResults?.length ?? 0) > 0) && (
                <div className="flex justify-center gap-3 pt-6 border-t border-gray-200">
                  <button onClick={() => updatePage(Math.max(1, page - 1))} disabled={page === 1} className="text-[14px] font-semibold px-5 py-2 border border-gray-200 rounded-full disabled:opacity-40 disabled:pointer-events-none bg-white hover:bg-gray-50 text-gray-700 active:scale-95 transition-all duration-200">이전</button>
                  <span className="text-[14px] text-gray-500 flex items-center px-3 font-semibold">{page} 페이지</span>
                  <button onClick={() => updatePage(page + 1)} disabled={!hasNextPage} className="text-[14px] font-semibold px-5 py-2 border border-gray-200 rounded-full disabled:opacity-40 disabled:pointer-events-none bg-white hover:bg-gray-50 text-gray-700 active:scale-95 transition-all duration-200">다음</button>
                </div>
              )}
            </section>
          </div>
      </main>
    </div>
  )
}
