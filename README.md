# ET DashBoard

**한국어 뉴스 신뢰도 분석 플랫폼**

네이버 뉴스를 수집하고 Gemini AI로 요약·임베딩·신뢰도 점수를 생성한 뒤,
하이브리드 검색(시맨틱 + BM25 RRF)으로 사용자에게 제공합니다.

---

## 배포 URL

| 서비스 | URL |
|--------|-----|
| **프론트엔드 (Vercel)** | `https://<your-project>.vercel.app` |
| **백엔드 API (Railway)** | `https://<your-service>.railway.app` |
| **GitHub** | https://github.com/unipjh/ET_DashBoard_Project |

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Frontend | React 19, Vite 7, TypeScript, TailwindCSS 4, React Query 5, React Router 7 |
| Backend | Python 3.11, FastAPI, uvicorn |
| Database | Supabase (PostgreSQL + pgvector) |
| AI | Google Gemini API (`gemini-embedding-001` 임베딩, Flash 요약·신뢰도) |
| Deployment | Vercel (프론트), Railway (백엔드) |

---

## 주요 기능

- **뉴스 크롤링** — 네이버 뉴스 8개 카테고리(IT/과학·경제·사회·생활문화·세계·정치·연예·스포츠) 자동 수집
- **AI 파이프라인** — Gemini API로 요약·키워드 추출·768차원 임베딩 생성 (Contextual Chunking)
- **신뢰도 분석** — TELLER 기반 5개 기준(출처신뢰성·근거지지도·문체중립성·논리일관성·어뷰징위험도) 0~100 점수화
- **하이브리드 검색** — pgvector HNSW 시맨틱 검색 + BM25 키워드 검색, RRF 방식 결합
- **관련 기사 추천** — 벡터 유사도 기반 관련 기사 5개 제공
- **핵심 키워드 검색** — 상세 페이지 키워드 클릭 시 동일 키워드 기사 검색
- **실시간 주가 지수** — 상세 페이지 상단 코스피·코스닥·DOW 표시, 클릭 시 네이버 금융 이동
- **관리자 대시보드** — 크롤링·분석·중복제거 실행, DB 현황·카테고리별 통계 모니터링 (관리자 계정 전용)
- **회원 시스템** — 회원가입·로그인, 피드백 페이지 로그인 필수
- **피드백** — 기사별 좋아요/싫어요 수집 (회원 전용, Supabase 저장)
- **행동 로그** — 비회원·회원 이벤트 로그 수집 (`event_logs`, 회원은 `user_id` 포함)
- **AI 요약 하이라이트** — 본문에서 요약과 일치하는 중요 문장 형광펜 표시
- **URL 공유** — AI 요약 옆 클립 아이콘으로 기사 링크 클립보드 복사

---

## 전체 플로우

```
[관리자]
  → 로그인 (관리자 계정) → Admin 페이지 접근
  → 크롤링 수 설정 후 시작
  → crawl.py: 네이버 뉴스 멀티스레드 크롤링
  → admin_pipeline.py: Gemini 요약 + Contextual Chunking + 임베딩
  → repo.py: Supabase(PostgreSQL + pgvector) 적재

[사용자]
  → 메인 페이지: 최신 기사 목록 조회 (카테고리 필터, 페이지네이션)
  → 검색창: 검색어 임베딩 → 하이브리드 검색 (시맨틱 + BM25 RRF)
  → 상세 페이지: 요약 + 핵심 키워드 + 신뢰도 패널 + 원문 + 관련 기사 추천
  → 피드백: 로그인 후 좋아요/싫어요 → Supabase feedback_logs 저장
```

---

## 로컬 실행

### 사전 조건

- Python 3.11+
- Node.js 20+
- Supabase 프로젝트 (pgvector 확장 활성화 필요)

### 환경 변수 설정

루트 `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@host:5432/postgres
ALLOWED_ORIGINS=http://localhost:5173
KRX_API_KEY=   # KRX 정보데이터시스템 API 키 (선택)
```

### 백엔드

```bash
pip install -r requirements.txt

uvicorn backend.main:app --reload
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 배포 설정

### Vercel (프론트엔드)

`vercel.json` 설정 포함. 필요한 환경변수:

```
VITE_API_URL = https://<your-railway-service>.railway.app
```

### Railway (백엔드)

`railway.json` 설정 포함. 필요한 환경변수:

```
DATABASE_URL    = postgresql://...   (Supabase 연결 문자열)
GEMINI_API_KEY  = ...
ALLOWED_ORIGINS = https://<your-project>.vercel.app
```

---

## API 엔드포인트

### 공개

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 서버 상태 확인 |
| `GET` | `/api/articles` | 기사 목록 (`?page=1&size=10&category=경제`) |
| `GET` | `/api/articles/{id}` | 기사 상세 |
| `GET` | `/api/articles/{id}/related` | 관련 기사 (`?limit=5`) |
| `GET` | `/api/articles/{id}/thumbnail` | 기사 썸네일 |
| `POST` | `/api/search` | 하이브리드 검색 (`{"query":"...", "limit":10}`) |
| `POST` | `/api/feedback` | 피드백 제출 |
| `GET` | `/api/feedback` | 전체 피드백 목록 |
| `POST` | `/api/logs` | 이벤트 로그 수집 |
| `GET` | `/api/admin/process-status` | 현재 작업 상태 |
| `GET` | `/api/stocks` | 주가 지수 조회 (코스피·코스닥·DOW) |
| `POST` | `/api/auth/signup` | 회원가입 |
| `POST` | `/api/auth/login` | 로그인 |

### 관리자 전용

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/admin/stats` | DB 현황 통계 |
| `POST` | `/api/admin/crawl` | 크롤링 시작 |
| `POST` | `/api/admin/analyze` | 신뢰도 분석 (미분석 기사 일괄) |
| `POST` | `/api/admin/dedupe` | 중복 기사 제거 |
| `POST` | `/api/admin/keywords` | 키워드 누락 기사 일괄 추출 |

---

## 프로젝트 구조

```
ET_DashBoard_Project/
├── backend/
│   ├── main.py                  # FastAPI 진입점, CORS
│   ├── schemas.py               # Pydantic 모델
│   ├── requirements.txt
│   ├── routers/
│   │   ├── articles.py          # /api/articles, /related, /thumbnail
│   │   ├── search.py            # /api/search (하이브리드)
│   │   ├── admin.py             # /api/admin/*
│   │   ├── trust.py             # /api/trust/{id}
│   │   ├── feedback.py          # /api/feedback
│   │   ├── logs.py              # /api/logs (이벤트 로그)
│   │   ├── stocks.py            # /api/stocks (주가 지수)
│   │   └── auth.py              # /api/auth/signup, /api/auth/login
│   └── services/
│       ├── repo.py              # DB CRUD (BM25 캐시 5분)
│       ├── crawl.py             # 네이버 뉴스 크롤러
│       ├── admin_pipeline.py    # Gemini 요약·임베딩 오케스트레이션
│       ├── eval_trust.py        # TELLER 기반 신뢰도 분석
│       ├── process_status.py    # 백그라운드 작업 상태 공유
│       └── config.py            # 환경변수 로드
├── frontend/
│   └── src/
│       ├── App.tsx              # 라우터, 스플래시 화면
│       ├── api/
│       │   ├── client.ts        # axios 인스턴스, API 함수
│       │   └── logger.ts        # 이벤트 로그 전송 (회원/비회원 구분)
│       ├── components/
│       │   └── ArticleCard.tsx
│       └── pages/
│           ├── MainPage.tsx     # 메인 (검색·카테고리·로그인·피드백 접근 제어)
│           ├── DetailPage.tsx   # 상세 (신뢰도·키워드·주가·관련기사)
│           ├── AdminPage.tsx    # 관리자 전용 (관리자 계정 필요)
│           └── FeedbackPage.tsx # 피드백 이력 (로그인 필요)
├── vercel.json                  # Vercel 배포 설정
├── railway.json                 # Railway 배포 설정
└── .env                         # 로컬 환경변수 (gitignore)
```

---

## DB 스키마

```sql
-- 기사
articles (
  article_id PK, title, source, url, published_at,
  full_text, summary_text, keywords,
  embed_full, embed_summary,
  trust_score INTEGER, trust_verdict, trust_reason, trust_per_criteria,
  status, category
)

-- 벡터 검색용 청크
article_chunks (
  chunk_id PK, article_id FK,
  chunk_text, embedding vector(768)
)

-- 피드백 (회원 전용)
feedback_logs (
  feedback_id PK, article_id FK,
  feedback_type, created_at, user_id FK → users
)

-- 이벤트 로그 (비회원·회원 모두)
event_logs (
  log_id PK, session_id,
  event_type, article_id, event_data, created_at,
  user_id FK → users  -- 비회원 NULL, 회원 user_id
)

-- 회원
users (
  user_id PK, user_pw, created_at
)
```

---

## 검색 품질 설계

| 항목 | 내용 |
|------|------|
| 임베딩 모델 | `gemini-embedding-001` (768차원) |
| task_type 분리 | 적재: `retrieval_document` / 검색: `retrieval_query` |
| Contextual Chunking | 청크에 `[제목][출처][카테고리]` prefix 부착 |
| 청킹 파라미터 | chunk_size=400, chunk_overlap=150, 한국어 sentence separator |
| 제목 전용 청크 | `[제목] {title}` 별도 임베딩 → 키워드 검색 대응 |
| 벡터 인덱스 | HNSW (코사인 유사도) |
| 하이브리드 융합 | RRF (Reciprocal Rank Fusion, k=60) |
| 아티클 중복 제거 | article_id당 최고 점수 청크만 사용 |
