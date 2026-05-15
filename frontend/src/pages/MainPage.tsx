import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { fetchArticles, searchArticles } from '../api/client'
import ArticleCard from '../components/ArticleCard'

const CATEGORIES = ['전체', '정치', '경제', '사회', '생활/문화', '세계', 'IT/과학']
// 점수에 따라 HSL 색상의 Hue(색조) 값을 동적으로 계산하는 함수
const getHue = (s: number) => {
  if (s <= 30) return (s / 30) * 15; // 0~30점: 빨간색 영역 (Hue 0~15)
  if (s <= 60) return 15 + ((s - 30) / 30) * 35; // 30~60점: 주황~노란색 영역 (Hue 15~50)
  return 50 + ((s - 60) / 40) * 90; // 60~100점: 노란색~초록색 영역 (Hue 50~140)
}

export default function MainPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const submittedQuery = searchParams.get('q') || ''
  const selectedCategory = searchParams.get('cat') || '전체'
  const page = parseInt(searchParams.get('page') || '1', 10)

  const [query, setQuery] = useState(submittedQuery)
  const location = useLocation()
  const navigate = useNavigate()
  const [isPageLoaded, setIsPageLoaded] = useState(false)

  const isSearchMode = !!submittedQuery;

  // 뒤로가기 등으로 URL 파라미터가 변했을 때, 검색창 텍스트 동기화
  useEffect(() => {
    setQuery(submittedQuery)
  }, [submittedQuery])

  // 현재 페이지와 선택된 카테고리에 해당하는 기사들을 불러옵니다. (검색 모드가 아닐 때)
  const { data: categoryFilteredData, isLoading: isLoadingCategoryArticles } = useQuery({
    queryKey: ['articles', page, selectedCategory],
    queryFn: () => fetchArticles(page, 10, selectedCategory === '전체' ? undefined : selectedCategory),
    enabled: !isSearchMode, // 검색 모드가 아닐 때만 활성화
  })
  const categoryFilteredArticles = categoryFilteredData?.articles || [];
  const categoryTotalCount = categoryFilteredData?.total_count || 0;

  // '전체' 카테고리일 때 '오늘의 추천 뉴스'를 위해 1페이지 데이터를 별도로 가져옵니다 (캐싱됨)
  const { data: firstPageDataForTopPicks, isLoading: isLoadingFirstPageArticles } = useQuery({
    queryKey: ['articles', 1, selectedCategory],
    queryFn: () => fetchArticles(1, 10, selectedCategory === '전체' ? undefined : selectedCategory),
    enabled: !isSearchMode && selectedCategory === '전체', // 검색 모드가 아니고 '전체' 카테고리일 때만 활성화
  })
  const firstPageArticlesForTopPicks = firstPageDataForTopPicks?.articles || [];

  // 검색 결과 (검색 모드일 때)
  const { data: searchResults, isLoading: isSearching } = useQuery({
    queryKey: ['search', submittedQuery],
    queryFn: () => searchArticles(submittedQuery),
    enabled: isSearchMode, // 검색 모드일 때만 활성화
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const newParams = new URLSearchParams(searchParams);
    if (query.trim()) { // 검색어가 있으면
      newParams.set('q', query.trim());
      newParams.delete('cat'); // 검색 시 카테고리 파라미터 제거
    } else { // 검색어가 비어있으면
      newParams.delete('q'); // 검색 파라미터 제거
      // 검색어가 비어있고, 현재 URL에 cat 파라미터가 없으면 '전체'로 간주 (기본 동작)
      // newParams.set('cat', '전체'); // 이 부분은 불필요, selectedCategory가 '전체'로 기본 설정됨
    }
    newParams.set('page', '1');
    setSearchParams(newParams);
  }

  const handleCategoryClick = (cat: string) => {
    setQuery('') // 카테고리 클릭 시 검색어 초기화
    const newParams = new URLSearchParams(searchParams);
    newParams.delete('q'); // 카테고리 클릭 시 검색 파라미터 제거
    if (cat === '전체') {
      newParams.delete('cat'); // '전체' 카테고리 선택 시 cat 파라미터 제거
    } else {
      newParams.set('cat', cat);
    }
    newParams.set('page', '1');
    setSearchParams(newParams);
  }

  const updatePage = (newPage: number) => {
    const newParams = new URLSearchParams(searchParams)
    newParams.set('page', newPage.toString())
    setSearchParams(newParams)
  }

  // 전체 로딩 상태
  const isLoading = isSearchMode ? isSearching : (isLoadingCategoryArticles || isLoadingFirstPageArticles);

  // 최종적으로 화면에 렌더링할 기사 목록 결정
  let articlesToRender = [];
  let totalCountForDisplay = 0; // 검색 결과나 카테고리 필터링 시 총 개수를 표시하기 위함

  if (isSearchMode) {
    // 검색 모드일 때는 검색 결과를 프론트엔드에서 페이지네이션합니다.
    articlesToRender = (searchResults || []).slice((page - 1) * 9, page * 9);
    totalCountForDisplay = (searchResults || []).length;
  } else {
    // 일반 최신 기사 또는 카테고리 필터링 모드일 때
    const topPicks = selectedCategory === '전체'
      ? [...firstPageArticlesForTopPicks].sort((a, b) => (b.trust_score || 0) - (a.trust_score || 0)).slice(0, 1)
      : [];

    articlesToRender = selectedCategory === '전체'
      ? categoryFilteredArticles.filter(a => !topPicks.find(t => t.article_id === a.article_id))
      : categoryFilteredArticles; // 카테고리 필터링 시에는 백엔드에서 이미 페이지네이션된 결과를 받습니다.
    totalCountForDisplay = categoryTotalCount; // 백엔드에서 받은 총 개수 사용
  }

  // 화면 전환 시 부드럽게 나타나는 애니메이션 트리거
  useEffect(() => {
    if (!isLoading) {
      setIsPageLoaded(false)
      const timer = setTimeout(() => setIsPageLoaded(true), 50)
      return () => clearTimeout(timer)
    }
  }, [isLoading, isSearchMode, page, selectedCategory])

  // ✨ 오늘의 추천 뉴스: 전체보기 상태일 때, 1페이지(최신) 기사 중 신뢰도 점수가 가장 높은 1개로 고정
  const topPicks = !isSearchMode && selectedCategory === '전체'
    ? [...firstPageArticlesForTopPicks].sort((a, b) => (b.trust_score || 0) - (a.trust_score || 0)).slice(0, 1)
    : []

  // remainingArticles 변수는 이제 articlesToRender에 통합되었으므로 제거합니다.
  // const remainingArticles = !isSearchMode && selectedCategory === '전체'
  //   ? articlesToRender.filter(a => !topPicks.find(t => t.article_id === a.article_id)).slice(0, 9)
  //   : displayArticles.slice((page - 1) * 9, page * 9)

  // Pagination disabled logic
  const hasNextPage = isSearchMode ? page * 9 < totalCountForDisplay : page * 10 < totalCountForDisplay;

  // URL 변경 시 스크롤을 맨 위로 올림
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.search]);


  return (
    <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-900">
      <header className="w-full px-8 py-6 flex items-center justify-start gap-3">
        {/* 피드백 버튼 (스마일 아이콘) */}
        <button
          onClick={() => navigate('/feedback')}
          className="flex items-center justify-center w-12 h-12 text-gray-700 bg-white hover:bg-gray-50 rounded-xl transition-all shadow-sm border border-gray-200 hover:scale-105"
          title="피드백 (마이페이지)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.182 15.182a4.5 4.5 0 0 1-6.364 0M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM9.75 9.75c0 .414-.168.75-.375.75S9 10.164 9 9.75 9.168 9 9.375 9s.375.336.375.75Zm-.375 0h.008v.015h-.008V9.75Zm5.625 0c0 .414-.168.75-.375.75s-.375-.336-.375-.75.168-.75.375-.75.375.336.375.75Zm-.375 0h.008v.015h-.008V9.75Z" />
          </svg>
        </button>

        <button
          onClick={() => navigate('/admin')}
          className="flex items-center justify-center w-12 h-12 text-gray-700 bg-white hover:bg-gray-50 rounded-xl transition-all shadow-sm border border-gray-200 hover:scale-105"
          title="Admin 설정"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-7 h-7">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.99l1.004.828c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
          </svg>
        </button>
      </header>

      <main className={`max-w-6xl mx-auto px-4 pb-10 pt-6 space-y-10 transition-all duration-700 ease-out transform ${isPageLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        
        {/* 검색 및 카테고리 필터 섹션 */}
        <section className="space-y-6">
          <div className="flex flex-col items-center justify-center gap-4 max-w-5xl mx-auto">
            <h1 
              className="text-[32px] font-extrabold text-gray-900 tracking-tighter whitespace-nowrap cursor-pointer text-center relative -top-14"
              onClick={() => handleCategoryClick('전체')}
            >
              ET DashBoard
            </h1>
            <form onSubmit={handleSearch} className="relative w-full max-w-4xl -top-6">
              <input
                type="text"
                placeholder="관심있는 뉴스 키워드를 검색해보세요..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                className="w-full bg-white border border-gray-300 rounded-full px-6 py-4 text-[17px] font-medium text-gray-900 placeholder-gray-400 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
              />
              <button type="submit" className="absolute right-3 top-2.5 bg-blue-600 text-white px-6 py-2 rounded-full font-medium hover:bg-blue-700 transition-colors">
                검색
              </button>
            </form>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 max-w-4xl mx-auto">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => handleCategoryClick(cat)}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                  selectedCategory === cat && !submittedQuery
                    ? 'bg-gray-900 text-white shadow-md transform scale-105 font-bold tracking-tight'
                    : 'bg-gray-200/60 text-gray-600 border border-transparent hover:bg-gray-200 hover:text-gray-900'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </section>

        {isLoading ? (
          <div className="text-center py-20">
            <div className="flex justify-center items-center gap-2 mb-6">
              <div className="w-3.5 h-3.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '-0.3s' }} />
              <div className="w-3.5 h-3.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '-0.15s' }}></div>
              <div className="w-3.5 h-3.5 bg-blue-600 rounded-full animate-bounce"></div>
            </div>
            <p className="text-gray-500 font-bold tracking-tight">기사를 불러오는 중입니다...</p>
          </div>
        ) : (
          <div className="space-y-12">
            {/* ✨ 추천 뉴스 (검색 모드가 아니고 '전체' 카테고리일 때만 노출) */}
            {!isSearchMode && selectedCategory === '전체' && topPicks.length > 0 && (
              <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                <h2 className="text-[20px] font-extrabold text-gray-900 tracking-tight flex items-center gap-2.5 mb-6">
                  <span className="text-2xl">✨</span> AI가 분석한 오늘의 추천 뉴스
                </h2>
                <div className="flex flex-col gap-6">
                  {topPicks.map((article: any) => (
                    <div 
                      key={article.article_id} 
                      onClick={() => navigate(`/article/${article.article_id}`)}
                      className="group cursor-pointer bg-gradient-to-r from-blue-50/40 to-white border border-blue-100 rounded-xl shadow-sm hover:shadow-lg hover:-translate-y-1.5 transition-all duration-300 ease-out flex flex-col md:flex-row overflow-hidden"
                    >
                      <div className="flex flex-col justify-center p-6 space-y-4 flex-1 order-2 md:order-1">
                        <div className="flex items-center gap-2">
                          <span className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-[11px] font-bold px-3 py-1 rounded-md shadow-sm">TOP PICK</span>
                          <span 
                            className="text-xs font-bold px-2 py-1 rounded-md border shadow-sm whitespace-nowrap"
                            style={{
                              color: `hsl(${getHue(article.trust_score || 0)}, 80%, 35%)`,
                              backgroundColor: `hsl(${getHue(article.trust_score || 0)}, 100%, 92%)`,
                              borderColor: `hsl(${getHue(article.trust_score || 0)}, 85%, 85%)`
                            }}
                          >
                            신뢰도 {article.trust_score}점
                          </span>
                        </div>
                        <h3 className="text-2xl sm:text-[26px] font-extrabold text-gray-900 group-hover:text-blue-600 transition-colors break-keep leading-[1.3]">{article.title}</h3>
                        <p className="text-[15.5px] text-gray-600 font-medium line-clamp-3 leading-relaxed break-keep">{article.summary_text || article.chunk_text}</p>
                        <div className="text-[13px] text-gray-400 font-bold flex gap-2">
                          <span>{article.source}</span>
                          <span>·</span>
                          <span>{article.published_at?.substring(0, 10)}</span>
                        </div>
                      </div>
                      <div className="w-full md:w-1/3 lg:w-[35%] h-52 md:h-auto relative overflow-hidden order-1 md:order-2 border-b md:border-b-0 md:border-l border-blue-100/50 bg-gray-100">
                        {/* 로딩 중이거나 에러가 났을 때 보여줄 배경 글씨 */}
                        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-400 text-[13px] font-bold tracking-tight">
                          썸네일 불러오는 중...
                        </div>
                        <img 
                          src={`/api/articles/${article.article_id}/thumbnail`} 
                          alt="기사 썸네일" 
                          className="relative z-10 object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out"
                          onError={(e) => {
                            e.currentTarget.src = `https://picsum.photos/seed/${article.article_id}/800/600`;
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* 📰 메인 기사 리스트 (검색 결과, 카테고리 필터링, 최신 기사 공용) */}
            <section className="space-y-6">
              <h2 className="text-[18px] font-extrabold text-gray-900 tracking-tight border-b border-gray-200 pb-3 flex items-center gap-2">
                {submittedQuery 
                  ? `"${submittedQuery}" 검색 결과` 
                  : selectedCategory !== '전체' 
                    ? `${selectedCategory} 관련 기사` 
                    : '최신 기사'} {/* '전체' 카테고리일 때 */}
                {(isSearchMode || selectedCategory !== '전체') && (
                  <span className="text-[14px] font-bold text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded-full ml-2">
                    {totalCountForDisplay}건
                  </span>
                )}
              </h2>
              
              {articlesToRender.length > 0 ? ( 
                <div className="flex flex-col gap-4 w-full">
                  {articlesToRender.map((article: any) => (
                    <div key={article.article_id} className="h-full transition-transform duration-300 ease-out hover:-translate-y-1.5">
                      <ArticleCard article={article} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-gray-50 rounded-2xl border border-dashed border-gray-200 p-16 text-center">
                  <p className="text-gray-500 font-bold tracking-tight">관련 기사를 찾을 수 없습니다.</p>
                </div>
              )}

              {/* 페이징 (전체 기사 및 검색/카테고리 공용) */}
              {(!isSearchMode || searchResults.length > 0) && ( // 검색 모드가 아니거나, 검색 결과가 있을 때만 페이징 표시
                <div className="flex justify-center gap-3 pt-6 border-t border-gray-100">
                  <button
                    onClick={() => updatePage(Math.max(1, page - 1))}
                    disabled={page === 1}
                    className="text-[15px] font-bold px-5 py-2 border border-gray-200 rounded-lg disabled:opacity-40 disabled:pointer-events-none hover:bg-gray-50 text-gray-700 active:bg-gray-900 active:text-white active:border-gray-900 active:scale-95 transition-all duration-200"
                  >
                    이전
                  </button>
                  <span className="text-[15px] text-gray-500 flex items-center px-2 font-bold">{page} 페이지</span>
                  <button
                    onClick={() => updatePage(page + 1)} // 다음 페이지로 이동
                    disabled={!hasNextPage} // Use hasNextPage
                    className="text-[15px] font-bold px-5 py-2 border border-gray-200 rounded-lg disabled:opacity-40 disabled:pointer-events-none hover:bg-gray-50 text-gray-700 active:bg-gray-900 active:text-white active:border-gray-900 active:scale-95 transition-all duration-200"
                  >
                    다음
                  </button>
                </div>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  )
}
