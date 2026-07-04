# ET DashBoard

**한국어 뉴스 신뢰도 분석 + 개인화 추천 플랫폼**

네이버 뉴스를 수집하고 Gemini AI로 요약·임베딩·신뢰도 점수를 생성합니다.
사용자 읽기 이력을 실시간으로 반영한 개인화 추천과 하이브리드 검색(시맨틱 + BM25 RRF)을 제공합니다.

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
| AI | Google Gemini API (`google-genai` SDK — `gemini-embedding-001` 임베딩, Flash 요약·신뢰도) |
| 추천 모델 | NRMS 기반 경량 Attention Encoder (PyTorch, 768d) |
| 테스트 | pytest (백엔드), Playwright (프론트 E2E) |
| Deployment | Vercel (프론트), Railway (백엔드) |

---

## 주요 기능

- **뉴스 크롤링** — 네이버 뉴스 8개 카테고리 자동 수집 (IT/과학·경제·사회·생활문화·세계·정치·연예·스포츠)
- **AI 파이프라인** — Gemini API로 요약·키워드 추출·768차원 임베딩 생성 (Contextual Chunking)
- **신뢰도 분석** — TELLER 기반 5개 기준 0~100 점수화 (출처신뢰성·근거지지도·문체중립성·논리일관성·어뷰징위험도)
- **개인화 추천** — 세션 읽기 이력 기반 4단계 폴백 추천 (Attention Encoder → 임베딩 유사도 → 카테고리 → 최신)
- **하이브리드 검색** — pgvector HNSW 시맨틱 검색 + BM25 키워드 검색, RRF 방식 결합
- **관련 기사 추천** — 벡터 유사도 기반 관련 기사 5개 제공
- **실시간 주가 지수** — 상세 페이지 코스피·코스닥·DOW 표시
- **관리자 대시보드** — 크롤링·분석·중복제거 실행, 배경 작업 이력, 성능 지표 모니터링
- **회원 시스템** — 회원가입·로그인, 피드백 페이지 로그인 필수
- **피드백** — 기사별 좋아요/싫어요 수집 (회원 전용)
- **행동 로그** — impression·click·페이지뷰 이벤트 자동 수집 (추천 모델 학습 원천)

---

## 전체 플로우

```
[관리자]
  → Admin 페이지 → 크롤링 수 설정 후 시작
  → crawl.py: 네이버 뉴스 멀티스레드 크롤링
  → admin_pipeline.py: Gemini 요약 + Contextual Chunking + 임베딩
  → repo.py: Supabase (PostgreSQL + pgvector) 적재

[사용자]
  → 메인 페이지: 개인화 추천 기사 + 최신 기사 목록 (카테고리 필터)
  → 검색창: 검색어 임베딩 → 하이브리드 검색 (시맨틱 + BM25 RRF)
  → 상세 페이지: 요약 + 키워드 + 신뢰도 패널 + 관련 기사 추천
  → 피드백: 로그인 후 좋아요/싫어요 → Supabase feedback_logs 저장

[추천 모델 학습]
  → event_logs에 impression/click 이벤트 축적
  → python -m backend.training.train_user_encoder
  → python -m backend.training.export_weights
```

---

## 개인화 추천 시스템

추천은 4단계 폴백 체인으로 동작합니다. 앞 단계가 결과를 반환하면 뒤 단계는 실행되지 않습니다.

```
읽기 이력 없음 ──────────────────────────▶ 최신 기사 반환

읽기 이력 있음
  │
  ├─ [1단계] Attention Encoder
  │   읽은 기사 시퀀스 → UserEncoder(Multi-Head Self-Attention) → 사용자 벡터
  │   → learned_embedding 검색
  │
  ├─ [2단계] 임베딩 유사도
  │   읽은 기사 embed_summary 가중 평균 → pgvector 코사인 유사도 검색
  │
  ├─ [3단계] 카테고리 기반
  │   읽은 기사 카테고리 추출 → 동일 카테고리 최신 기사
  │
  └─ [4단계] 최신 기사 (최후 폴백)
```

학습 파이프라인 (`backend/training/`):
- **합성 데이터 학습**: 8개 페르소나(IT마니아·경제독자·정치독자 등) 기반 시나리오로 초기 모델 부트스트랩
- **실데이터 재학습**: impression/click 이벤트 100회+ 축적 후 `--synthetic` 없이 재실행

자세한 내용: [docs/personalized_recommendation_2026-06-25.md](docs/personalized_recommendation_2026-06-25.md)

---

## 로컬 실행

### 사전 조건

- Python 3.11+, Node.js 20+
- Supabase 프로젝트 (pgvector 확장 활성화 필요)

### 환경 변수 설정 (루트 `.env`)

```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@host:5432/postgres
ALLOWED_ORIGINS=http://localhost:5173
ADMIN_PASSWORD=your_admin_password
```

### 백엔드

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# → http://localhost:8000  /  Swagger: http://localhost:8000/docs
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 테스트 / 프리플라이트

```bash
# 백엔드 유닛 테스트
python -m unittest backend.tests.test_smoke

# 프론트 E2E (Playwright)
cd frontend && npm run test:e2e

# 전체 프리플라이트 (lint → build → e2e → import → smoke)
python scripts/preflight.py
```

---

## API 엔드포인트

### 공개

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/articles` | 기사 목록 (`?page=1&size=10&category=경제`) |
| `GET` | `/api/articles/{id}` | 기사 상세 |
| `GET` | `/api/articles/{id}/related` | 관련 기사 |
| `POST` | `/api/search` | 하이브리드 검색 |
| `GET` | `/api/recommendations` | 개인화 추천 (`?session_id=xxx&limit=10`) |
| `POST` | `/api/logs` | 이벤트 로그 수집 |
| `POST` | `/api/auth/signup` | 회원가입 |
| `POST` | `/api/auth/login` | 로그인 |
| `GET` | `/api/stocks` | 주가 지수 (코스피·코스닥·DOW) |

### 관리자 전용 (`X-Admin-Password` 헤더 필요)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/admin/stats` | DB 현황 통계 |
| `POST` | `/api/admin/crawl` | 크롤링 시작 |
| `POST` | `/api/admin/analyze` | 신뢰도 분석 (미분석 일괄) |
| `POST` | `/api/admin/dedupe` | 중복 기사 제거 |
| `POST` | `/api/admin/keywords` | 키워드 누락 기사 일괄 추출 |
| `GET` | `/api/admin/process-status` | 현재 작업 상태 |
| `GET` | `/api/admin/jobs` | 배경 작업 이력 (최근 20건) |
| `GET` | `/api/admin/performance-metrics` | 클라이언트 성능 지표 (`?hours=24`) |

---

## 프로젝트 구조

```
ET_by_claude/
├── backend/
│   ├── main.py                  # FastAPI 진입점, CORS
│   ├── schemas.py               # Pydantic 모델
│   ├── requirements.txt
│   ├── migrations/              # SQL 마이그레이션 파일
│   ├── routers/
│   │   ├── articles.py          # /api/articles
│   │   ├── search.py            # /api/search (하이브리드)
│   │   ├── recommendations.py   # /api/recommendations (개인화 추천)
│   │   ├── admin.py             # /api/admin/*
│   │   ├── logs.py              # /api/logs (이벤트 로그)
│   │   ├── auth.py              # /api/auth/*
│   │   └── stocks.py            # /api/stocks
│   ├── services/
│   │   ├── repo.py              # DB CRUD
│   │   ├── recommend.py         # 추천 로직 + 폴백 체인
│   │   ├── encoder_inference.py # Attention Encoder 추론
│   │   ├── admin_pipeline.py    # Gemini 오케스트레이션
│   │   ├── crawl.py             # 네이버 뉴스 크롤러
│   │   ├── trust.py             # TELLER 신뢰도 분석
│   │   ├── process_status.py    # 배경 작업 상태 (DB 연동)
│   │   ├── config.py            # 환경변수 로드
│   │   └── model_weights/       # 배포용 학습 가중치 (.pt)
│   ├── training/
│   │   ├── news_encoder.py      # NewsEncoder 모델
│   │   ├── user_encoder.py      # UserEncoder 모델
│   │   ├── dataset.py           # 학습 데이터 로딩
│   │   ├── generate_synthetic_samples.py  # 페르소나 기반 합성 샘플
│   │   ├── train_user_encoder.py          # 학습 실행
│   │   ├── export_weights.py              # 가중치 서빙 경로 복사
│   │   └── backfill_learned_embeddings.py # 기존 기사 임베딩 채우기
│   └── tests/
│       └── test_smoke.py        # 백엔드 스모크 테스트
├── frontend/
│   └── src/
│       ├── api/
│       │   ├── client.ts        # axios 인스턴스, API 함수
│       │   ├── logger.ts        # 이벤트 로그 자동 전송
│       │   └── performance.ts   # 클라이언트 성능 지표 수집
│       └── pages/
│           ├── MainPage.tsx     # 메인 (개인화 추천 포함)
│           ├── DetailPage.tsx   # 상세 (신뢰도·관련기사)
│           ├── AdminPage.tsx    # 관리자 (작업 이력·성능 지표)
│           └── FeedbackPage.tsx
├── frontend/e2e/                # Playwright E2E 테스트
├── scripts/
│   └── preflight.py             # 배포 전 전체 검증
└── docs/                        # 구현 문서
```

---

## DB 스키마

```sql
-- 기사
articles (
  article_id PK, title, source, url, published_at, category,
  full_text, summary_text, keywords,
  embed_full, embed_summary,          -- Gemini 768d 임베딩
  learned_embedding,                  -- Attention Encoder 학습 임베딩
  trust_score, trust_verdict, trust_reason, trust_per_criteria,
  status
)

-- 벡터 검색용 청크
article_chunks (chunk_id PK, article_id FK, chunk_text, embedding vector(768))

-- 이벤트 로그 (추천 학습 원천)
event_logs (
  log_id PK, session_id, user_id,
  event_type,    -- impression | click_article | view_detail | visit_main | ...
  article_id, event_data JSONB, created_at
)

-- 배경 작업 이력
background_jobs (
  job_id PK, job_type, status,
  current_step, last_message, articles_processed,
  started_at, finished_at, error_text
)

-- 피드백 / 회원
feedback_logs (feedback_id PK, article_id FK, feedback_type, user_id FK, created_at)
users (user_id PK, user_pw, created_at)
```

---

## 추천 모델 재학습

```bash
# 합성 데이터로 즉시 학습 (현재 방식)
python -m backend.training.train_user_encoder --synthetic --epochs 5

# 실제 사용자 데이터로 재학습 (impression 1000+, click 100+ 축적 후)
python -m backend.training.train_user_encoder --epochs 10

# 가중치 서빙 경로에 복사
python -m backend.training.export_weights

# 기존 기사에 learned_embedding 채우기
python -m backend.training.backfill_learned_embeddings
```
