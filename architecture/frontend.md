# 프론트엔드

React 19 + TypeScript + Vite 기반 SPA. `frontend/src/`에 소스가 있고, 빌드 결과는 `frontend/dist/`에 생성된다.

---

## 페이지 라우팅 (`App.tsx`)

```mermaid
flowchart LR
    A[/] --> B[MainPage\n기사 목록 + 검색]
    A --> C[/article/:id\nDetailPage\n기사 상세 + 신뢰도]
    A --> D[/admin\nAdminPage\n크롤링 제어]
    A --> E[/feedback\nFeedbackPage\n피드백 목록]
```

React Router를 사용한다.

---

## 페이지별 기능

### MainPage (`pages/MainPage.tsx`)

메인 화면. 기사 목록, 검색, 카테고리 필터를 담당한다.

- **검색창**: 텍스트 입력 → `POST /api/search` 호출 (TanStack Query)
- **카테고리 버튼**: 9개 (전체 + 8개 카테고리) → 필터링
- **추천 뉴스 (Top Pick)**: 현재 페이지 기사 중 신뢰도 최고 1개를 상단에 표시
- **신뢰도 색상**: HSL 기반 동적 컬러링
  - 0~30점: 빨간색 (Hue 0~15)
  - 30~60점: 주황~노란색 (Hue 15~50)
  - 60~100점: 초록색 (Hue 50~140)
- **URL 상태**: 검색 쿼리, 카테고리, 페이지 번호를 URL 파라미터로 관리 (뒤로가기 지원)
- **페이지네이션**: 이전/다음 버튼

### DetailPage (`pages/DetailPage.tsx`)

기사 상세 화면. 신뢰도 분석 결과를 시각화한다.

- 기사 제목, 출처, 발행일, 본문
- **신뢰도 원형 차트**: SVG 기반, 로딩 시 1초 애니메이션
- **오각형 레이더 차트**: 5개 기준(source, evidence, style, logical, clickbait) 점수 시각화
- **기준별 상세 모달**: 클릭 시 점수 + reason 표시
- **피드백 버튼**: 좋아요/싫어요 (localStorage로 중복 방지)
- **관련 기사 5개**: `GET /api/articles/{id}/related` 결과

### AdminPage (`pages/AdminPage.tsx`)

크롤링 및 분석 제어 화면. 관리자용.

- **DB 통계**: 5초 간격 자동 갱신 (총 기사수, 미분석 수, 카테고리별 현황)
- **진행률 바**: 카테고리별 분석 완료 비율
- **크롤링 제어**:
  - 카테고리 멀티셀렉트
  - 최대 기사 수 입력
  - 시작 버튼 → `POST /api/admin/crawl`
- **실시간 로그**: 200ms 간격 폴링 (`GET /api/admin/process-status`)
- **일괄 작업 버튼**: 분석, 중복제거, 키워드 추출

### FeedbackPage (`pages/FeedbackPage.tsx`)

사용자 피드백 목록 화면. `GET /api/feedback` 결과를 테이블로 표시한다.

---

## 주요 컴포넌트

### ArticleCard (`components/ArticleCard.tsx`)

기사 목록에서 사용하는 카드 컴포넌트.

| 표시 정보 | 설명 |
|----------|------|
| 제목 | 클릭 시 DetailPage로 이동 |
| 출처 | 언론사명 |
| 신뢰도 점수 | HSL 색상 배지 |
| 요약 | summary_text (2줄 말줄임) |
| 카테고리 | 태그 |

---

## API 통신 (`api/client.ts`)

Axios 인스턴스를 공통으로 사용한다.

```typescript
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
})
```

주요 함수:

| 함수 | 메서드/경로 |
|------|------------|
| `fetchArticles(page, size, category?)` | GET /api/articles |
| `fetchArticle(id)` | GET /api/articles/{id} |
| `fetchRelatedArticles(id, limit?)` | GET /api/articles/{id}/related |
| `searchArticles(query, limit?)` | POST /api/search |
| `fetchStats()` | GET /api/admin/stats |
| `startCrawl(options)` | POST /api/admin/crawl |
| `startAnalyze()` | POST /api/admin/analyze |
| `fetchProcessStatus()` | GET /api/admin/process-status |
| `postFeedback(articleId, type)` | POST /api/feedback |

---

## 상태 관리

별도 전역 상태 라이브러리 없이 두 가지 방식을 조합한다.

### TanStack Query (서버 상태)

서버에서 오는 데이터는 모두 TanStack Query로 캐싱한다.

```typescript
// 기사 목록
const { data, isLoading } = useQuery({
  queryKey: ['articles', page, category],
  queryFn: () => fetchArticles(page, 10, category),
})

// 검색
const { data } = useQuery({
  queryKey: ['search', query],
  queryFn: () => searchArticles(query),
  enabled: !!query,
})

// 통계 (5초 자동 갱신)
const { data } = useQuery({
  queryKey: ['stats'],
  queryFn: fetchStats,
  refetchInterval: 5000,
})
```

### URL 파라미터 (UI 상태)

페이지, 카테고리, 검색 쿼리를 URL에 저장한다. 뒤로가기/새로고침 후에도 상태가 유지된다.

```
/?page=2&category=경제&q=인플레이션
```

### localStorage (피드백)

로그인 없이 피드백 중복 방지를 위해 로컬에 저장한다.

```typescript
// 키 형식: "feedbacks_{userId}"
const feedbacks = JSON.parse(localStorage.getItem(`feedbacks_${userId}`) || '{}')
feedbacks[articleId] = 'like'
localStorage.setItem(`feedbacks_${userId}`, JSON.stringify(feedbacks))
```

---

## 빌드 및 배포

```bash
# 개발 서버
npm run dev    # Vite dev server (localhost:5173)

# 프로덕션 빌드
npm run build  # dist/ 생성
```

FastAPI 서버가 `frontend/dist/`를 정적 파일로 서빙하거나, 별도 CDN에 배포한다.
