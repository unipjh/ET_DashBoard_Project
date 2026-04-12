# 📰 Explainable Trust — 뉴스 신뢰도 분석 플랫폼

네이버 뉴스 크롤링 → Gemini 요약/임베딩 → TELLER 기반 신뢰도 분석 → React 대시보드

---

## 1. 프로젝트 개요

### 🎯 목표

- **React(Vite+TS) + FastAPI 기반 설명 가능한 뉴스 신뢰도 분석 시스템**
- 핵심 기능:
  - Gemini API를 통한 자동 요약
  - Contextual Chunking + 벡터 임베딩 기반 시맨틱 검색
  - TELLER 논문 기반 5기준 신뢰도 분석 (출처 신뢰성·근거 지지도·문체 중립성·논리 일관성·어뷰징 위험도)
  - 관련 기사 추천 (제목+요약 임베딩 기반 유사도)

### 전체 플로우

```
[관리자]
  → 크롤링 버튼 클릭 (POST /api/admin/crawl)
  → crawl.py: 네이버 뉴스 멀티스레드 크롤링 (8개 카테고리)
  → admin_pipeline.py: Gemini 요약 + Contextual Chunking + 임베딩 + 신뢰도 분석
  → repo.py: DuckDB (articles + article_chunks) 적재

[사용자]
  → 메인 페이지: 최신 기사 리스트 조회 (GET /api/articles, 페이지네이션)
  → AI 검색: 검색어 임베딩(retrieval_query) → 유사 청크 검색 (POST /api/search)
  → 상세 페이지: 요약문 + 신뢰도 패널 + 원문 + 관련 기사 추천 (GET /api/articles/{id}/related)
```

---

## 2. 기술 스택

| 영역 | 스택 |
|---|---|
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS + TanStack Query |
| Backend | FastAPI + Uvicorn |
| DB | DuckDB (파일 기반, `app_db.duckdb`) |
| AI | Gemini 2.5 Flash (요약·신뢰도), Gemini Embedding 001 (768차원 벡터) |
| 크롤링 | BeautifulSoup + ThreadPoolExecutor |

---

## 3. 실행 방법

### 3-1. 환경 설정

`.env` 파일에 Gemini API 키 설정:

```
GEMINI_API_KEY=your_api_key_here
```

### 3-2. 백엔드 실행 (프로젝트 루트)

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### 3-3. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

- 프론트엔드: http://localhost:5173
- 백엔드 API: http://localhost:8000
- API 문서 (Swagger): http://localhost:8000/docs

### 3-4. 사용 순서

1. Admin 페이지 이동
2. **카테고리당 기사 수** 설정 (5~50개)
3. **크롤링 시작** 버튼 → 크롤링·요약·임베딩·신뢰도 분석 자동 진행 (백그라운드)
4. 미분석 기사가 있으면 **신뢰도 분석** 버튼으로 일괄 처리 가능
5. 메인 페이지에서 기사 조회 및 AI 검색

---

## 4. 프로젝트 디렉토리 구조

```text
.
├─ .env                          # Gemini API 키 (.gitignore)
├─ app_db.duckdb                 # DuckDB 파일 (자동 생성, .gitignore)
│
├─ backend/
│   ├─ main.py                   # FastAPI 앱 진입점, CORS, 라우터 등록
│   ├─ schemas.py                # Pydantic 스키마 (ArticleOut, ArticleDetail, ...)
│   ├─ requirements.txt
│   │
│   ├─ routers/
│   │   ├─ articles.py           # GET /api/articles, /api/articles/{id}, /api/articles/{id}/related
│   │   ├─ search.py             # POST /api/search (시맨틱 검색)
│   │   ├─ admin.py              # POST /api/admin/crawl, /api/admin/analyze, GET /api/admin/stats
│   │   └─ trust.py             # GET /api/trust/{id}
│   │
│   └─ services/
│       ├─ config.py             # GEMINI_API_KEY 로드 (.env)
│       ├─ crawl.py              # 네이버 뉴스 크롤링 전담
│       ├─ admin_pipeline.py     # Gemini 요약 + Contextual Chunking + 임베딩 오케스트레이션
│       ├─ trust.py              # TELLER 기반 신뢰도 분석 (Gemini)
│       └─ repo.py               # DuckDB CRUD (articles + article_chunks)
│
├─ frontend/
│   ├─ src/
│   │   ├─ api/client.ts         # Axios API 클라이언트
│   │   ├─ pages/
│   │   │   ├─ MainPage.tsx      # 메인 (검색 + 기사 리스트)
│   │   │   ├─ DetailPage.tsx    # 상세 (요약·신뢰도·본문·관련기사)
│   │   │   └─ AdminPage.tsx     # 관리자 (크롤링·일괄분석·DB현황)
│   │   └─ components/
│   │       ├─ ArticleCard.tsx   # 기사 카드 (신뢰도 뱃지 포함)
│   │       └─ TrustGauge.tsx    # 신뢰도 패널 (5기준 바 차트)
│   └─ ...
│
└─ docs/
    ├─ CLAUDE.md                 # 프로젝트 가이드 (Claude 전용)
    ├─ TD1.md                    # TD1 신뢰도 모듈 기술 계획
    └─ TD_explain.md             # 전체 기술 계획 상세 문서
```

---

## 5. API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/articles` | 기사 목록 (페이지네이션) |
| GET | `/api/articles/{id}` | 기사 상세 |
| GET | `/api/articles/{id}/related` | 관련 기사 (유사도 검색) |
| POST | `/api/search` | 시맨틱 검색 |
| GET | `/api/admin/stats` | DB 현황 |
| POST | `/api/admin/crawl` | 크롤링 시작 (BackgroundTask) |
| POST | `/api/admin/analyze` | 미분석 기사 일괄 신뢰도 분석 (BackgroundTask) |
| GET | `/api/trust/{id}` | 기사 신뢰도 조회 |

---

## 6. DB 스키마

### articles 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| article_id | VARCHAR (PK) | 고유 기사 ID |
| title | VARCHAR | 제목 |
| source | VARCHAR | 출처 |
| url | VARCHAR | 원문 URL |
| published_at | VARCHAR | 발행일 |
| full_text | VARCHAR | 기사 전문 |
| summary_text | VARCHAR | Gemini 요약 |
| keywords | VARCHAR | 키워드 (JSON) |
| embed_full | VARCHAR | 전문 임베딩 (placeholder) |
| embed_summary | VARCHAR | 요약 임베딩 (placeholder) |
| trust_score | INTEGER | 신뢰도 점수 (0~100) |
| trust_verdict | VARCHAR | `likely_true` / `uncertain` / `likely_false` |
| trust_reason | VARCHAR | 종합 판단 근거 |
| trust_per_criteria | VARCHAR | 5기준별 점수+근거 (JSON) |
| status | VARCHAR | 처리 상태 |

### article_chunks 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| chunk_id | VARCHAR (PK) | `{article_id}_{번호}` 또는 `{article_id}_title` |
| article_id | VARCHAR | 기사 ID (FK) |
| chunk_text | VARCHAR | 원문 청크 (표시용) |
| embedding | FLOAT[768] | Contextual Chunking 임베딩 (검색용) |

---

## 7. 검색 품질 튜닝 포인트

| 포인트 | 내용 | 적용 위치 |
|---|---|---|
| 임베딩 모델 통일 | `gemini-embedding-001` 단일 모델 | `admin_pipeline.py`, `routers/search.py` |
| task_type 분리 | 적재: `retrieval_document` / 검색: `retrieval_query` | `admin_pipeline.py`, `routers/` |
| Contextual Chunking | 청크에 `[제목][출처][카테고리]` prefix 부착 | `admin_pipeline.py` |
| 청킹 전략 | `chunk_size=400`, `chunk_overlap=150`, 한국어 sentence separator | `admin_pipeline.py` |
| 제목 전용 청크 | `[제목] {title}` 별도 임베딩 → 키워드 검색 대응 | `admin_pipeline.py` |
| 관련 기사 검색 | full_text 대신 제목+요약 재임베딩으로 관련 기사 탐색 | `routers/articles.py` |
| SQL단 현재 기사 제외 | `search_similar_chunks_excluding()` | `services/repo.py` |
| dedupe_per_article | article_id당 최고 점수 청크만 남김 | `services/repo.py` |
| min_score 완화 | 기본값 0.5 (검색 결과 다양성 확보) | `services/repo.py` |

---

## 8. 팀 협업 가이드

| 역할 | 수정 파일 |
|---|---|
| 크롤링 | `backend/services/crawl.py` |
| 요약 / 임베딩 / 청킹 | `backend/services/admin_pipeline.py` |
| 신뢰도 모델 | `backend/services/trust.py` |
| API 엔드포인트 | `backend/routers/` |
| DB 스키마 / 검색 쿼리 | `backend/services/repo.py` |
| UI 페이지 | `frontend/src/pages/` |
| 공통 컴포넌트 | `frontend/src/components/` |
| API 클라이언트 | `frontend/src/api/client.ts` |
