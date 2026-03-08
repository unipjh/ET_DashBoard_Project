# 📰 Explainable Fake News Detection & Recommendation Platform (MVP)

본 프로젝트는 **Streamlit + DuckDB + Gemini API 기반의 설명 가능한 뉴스 검색·요약 플랫폼**입니다.
현재 단계에서는 **네이버 뉴스 크롤링 → Gemini 요약/임베딩 → DuckDB 적재 → 시맨틱 검색 UI 흐름을 안정적으로 구축하는 것**을 진행 중입니다.

---

## 1. 프로젝트 개요

### 🎯 목표

- **Streamlit 기반의 "설명 가능한 뉴스 검색·추천 시스템" 대시보드**
- 핵심 기능:
  - Gemini API를 통한 자동 요약
  - Contextual Chunking + 벡터 임베딩 기반 시맨틱 검색
  - 관련 기사 추천 (제목+요약 임베딩 기반 유사도)
  - **TELLER 기반 신뢰도 분석** (5개 기준, Gemini Cognitive + Weighted Sum Rule)
- 성과물:
  - ✔ 로컬 및 URL로 작동하는 서비스 (Streamlit Cloud)
  - ✔ 네이버 뉴스 크롤링 → Gemini 자동 요약 파이프라인
  - ✔ 청크 단위 임베딩 + DuckDB 벡터 검색
  - ✔ 기사별 신뢰도 자동 분석 및 UI 표시
  - ✔ 이후 확장 가능한 토대 확보

### 전체 플로우

```
[관리자]
  → 크롤링 버튼 클릭
  → crawl.py: 네이버 뉴스 크롤링 (8개 카테고리, 병렬, 본문 정제 + 중복 제거)
  → admin_pipeline.py: Gemini 요약 + Contextual Chunking + 임베딩 + 신뢰도 분석
  → repo.py: DuckDB (articles + article_chunks) 적재

[관리자 - 신뢰도 일괄 재분석]
  → trust_score=0인 기사 목록 조회
  → score_trust() 호출 → repo.update_article_trust() 로 개별 업데이트

[사용자]
  → 메인 페이지: 최신 기사 리스트 조회 (페이지네이션, 10개씩)
  → AI 검색: 검색어 임베딩(retrieval_query) → 유사 청크 검색 → 관련 기사 반환
  → 상세 페이지: AI 요약 + 신뢰도 분석 패널 + 원문 + 관련 기사 추천
```

### 🧠 현재 단계 (MVP)

- 크롤링: 네이버 뉴스 실제 크롤러 (`crawl.py`) — 8개 카테고리, MAX_WORKERS=10 병렬 처리
- 요약: Gemini 2.5 Flash Lite (`admin_pipeline.py`)
- 임베딩: Gemini Embedding 001 모델 (768차원), `task_type` 분리 적용
- 청킹: Contextual Chunking (제목/출처/카테고리 prefix) + 제목 전용 청크
- DB: DuckDB 파일 기반 (`articles` + `article_chunks` 테이블)
- **신뢰도: TELLER 기반 실제 분석 구현 완료** (`trust.py`) — Gemini 2.5 Flash, 5개 기준 가중합산
- UI: Streamlit

---

## 2. 실행 방법

### 2-1. 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일에 Gemini API 키를 설정합니다:

```
GEMINI_API_KEY=your_api_key_here
```

### 2-2. 실행

```bash
streamlit run app.py
```

브라우저가 열리면:

1. **⚙️ Admin** 버튼 클릭
2. **카테고리당 기사 수** 설정 (5~50개)
3. **🚀 크롤링 시작** 버튼 클릭
4. 크롤링 → Gemini 요약/임베딩/신뢰도 분석 → DB 적재 자동 진행
5. 사용자 페이지에서 기사 조회 및 AI 검색 가능

> 기존 DB에 trust_score=0인 기사가 있다면 Admin의 **🔍 신뢰도 일괄 분석** 버튼으로 재분석 가능

---

## 3. 프로젝트 디렉토리 구조

```text
.
├─ app.py                        # Streamlit 엔트리 포인트
├─ app_db.duckdb                 # DuckDB 파일 (자동 생성, .gitignore)
├─ .env                          # Gemini API 키 설정 (.gitignore)
├─ requirements.txt
├─ README.md
│
├─ docs/
│   ├─ CLAUDE.md                 # 프로젝트 가이드 (Claude 전용)
│   ├─ TD1.md                    # TD1 신뢰도 모듈 기술 계획
│   └─ TD_explain.md             # 전체 기술 계획 상세 문서
│
└─ Streamlit_Rendering/
    ├─ __init__.py
    │
    ├─ functions.py              # UI 렌더링 (사용자 / 관리자 페이지)
    ├─ admin_pipeline.py         # 관리자 파이프라인 (Gemini 요약/임베딩/신뢰도, 청킹)
    ├─ crawl.py                  # 네이버 뉴스 크롤링 전담
    │
    ├─ repo.py                   # DuckDB 입출력 전담
    ├─ data.py                   # MOCK 데이터
    │
    ├─ trust.py                  # TELLER 기반 신뢰도 분석 모듈 (실제 구현)
    └─ search.py                 # Gemini 임베딩 기반 RAG 검색 실험용 스크립트
```

---

## 4. 파일별 역할 및 주요 함수 설명

---

### 4.1 `app.py`

**역할**

- Streamlit 앱의 진입점
- 세션 상태 기반 페이지 라우팅

**세션 상태**

- `selected_article_id`: 현재 선택된 기사 ID
- `search_executed`: 검색 실행 여부
- `search_query`: 검색어
- `admin_mode`: 관리자 모드 여부

**라우팅 흐름**

- `admin_mode=True` → `render_admin_page()`
- `selected_article_id` 있음 → `render_detail_page()`
- `search_executed=True` → `render_search_results_page()`
- 기본 → `render_main_page()`

---

### 4.2 `functions.py` (UI 레이어)

**역할**

- Streamlit 화면 렌더링 전담
- DB 조회 및 Gemini 임베딩 호출
- 뒤로가기 로직으로 상세 → 검색결과 → 메인 단계별 이동 지원

#### 주요 함수

```python
go_back()
```

- 현재 상태에 따라 직전 페이지로 이동
- 상세 → 검색결과(또는 메인) / 관리자 → 메인

```python
render_main_page()
```

- 검색창 + 최신 기사 리스트 (페이지네이션, 10개씩)
- Admin 버튼 제공

```python
render_search_results_page(query)
```

- 검색어를 `task_type="retrieval_query"`로 임베딩
- `repo.search_similar_chunks(min_score=0.65)` 호출
- 유사도 점수(%) 및 매칭된 청크 미리보기 표시 (🟢/🟡/🔴)

```python
render_detail_page(aid)
```

- 좌: AI 요약 + 키워드 제공 + 관련 기사 추천
  - 추출된 키워드를 대분류 > 중분류 > 소분류 기준으로 그룹화
  - 관련 기사: **제목+요약문** 임베딩으로 검색 (full_text 대비 노이즈 감소)
  - `repo.search_similar_chunks_excluding()` 으로 현재 기사 SQL단에서 제외
- 우: 출처/날짜/원문 링크 + **신뢰도 분석 패널** + 기사 전문
  - 신뢰도 패널: 종합 점수, 판정(likely_true/uncertain/likely_false), 기준별 세부 점수, 종합 판단 근거

```python
render_admin_page()
```

- 카테고리당 기사 수 설정 (5~50개)
- 크롤링 → 요약/임베딩/신뢰도 → DB 적재 버튼
- **신뢰도 일괄 분석**: trust_score=0인 미분석 기사 재분석 버튼
- DB 현황 (총 기사 수, 소스 종류, 최신 기사 목록)

---

### 4.3 `admin_pipeline.py` (관리자 파이프라인)

**역할**

- 크롤링된 기사를 받아 Gemini 요약 + Contextual Chunking + 임베딩 생성 + 신뢰도 분석 후 DB 적재
- `.env`의 `GEMINI_API_KEY`를 자동으로 로드
- 임베딩 모델: `models/gemini-embedding-001` (768차원), `task_type` 파라미터로 용도 분리

#### 주요 함수

```python
run_gemini_summary(text: str) -> str
```

- Gemini 2.5 Flash Lite로 글 길이에 맞춘 요약 진행
- 5개의 핵심 키워드(대분류 > 중분류 > 소분류 형태)를 딕셔너리 형태로 반환.
- 429(할당량 초과) 에러 시 30~60초 후 자동 재시도

```python
_make_chunk_context(title, source, category, chunk_text) -> str
```

- **Contextual Chunking**: 청크에 `[제목] [출처] [카테고리]` prefix 부착
- 청크가 어떤 기사의 일부인지 의미 정보가 임베딩에 반영됨 → 검색 정확도 향상

```python
build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int
```

- 청킹 전략: `chunk_size=400`, `chunk_overlap=150`, 한국어 문장 단위 separator 적용
- 기사별 처리 순서: Gemini 요약 → **신뢰도 분석** → DB 저장 (`articles`)
- 청크별: Contextual Chunking → `retrieval_document` 임베딩 → DB 저장 (`article_chunks`)
- **제목 전용 청크 추가**: `[제목] {title}` 형태로 별도 임베딩 저장 → 짧은 키워드 검색 대응
- API 속도 제한 회피: 요약 후 2초, 신뢰도 분석 후 2초, 청크당 0.5초, 기사 완료 후 3초 대기

**crawl.py와의 인터페이스 계약**

| crawl.py 컬럼 | pipeline 내부 변수 |
|---|---|
| `content` | `full_text` |
| `link` | `url` |
| `date` | `published_at` |
| `category` | `category` (Contextual Chunking에 활용) |

> ⚠️ crawl.py의 반환 컬럼명은 **절대 변경하지 말 것**

---

### 4.4 `crawl.py` (크롤링 전담)

**역할**

- 네이버 뉴스 실제 크롤링 로직
- 본문 정제, 중복 제거, 유사 기사 필터링, 기자명 필터링 포함

#### 설정값

| 항목 | 값 |
|---|---|
| 크롤링 대상 | 정치/경제/사회/생활·문화/세계/IT·과학/연예/스포츠 (8개 카테고리) |
| 병렬 처리 | `MAX_WORKERS=10` (ThreadPoolExecutor) |
| 수집 기간 | 최근 7일 (`START_DATE = now - 7days`) |
| 본문 최소 길이 | 200자 미만 필터링 |
| 중복 제거 임계값 | Jaccard 유사도 0.7 이상 시 제거 |

#### 주요 함수

```python
fetch_articles_from_naver(max_articles_per_category=30) -> pd.DataFrame
```

- 8개 카테고리 URL 수집 → ThreadPoolExecutor 병렬 크롤링
- 반환 컬럼: `date`, `category`, `source`, `title`, `reporter`, `comment_cnt`, `like_cnt`, `link`, `content`
- 처리 파이프라인: 본문 정제(`clean_news_content`) → 200자 필터링 → 제목 중복 제거 → 기자명 필터링 → Jaccard 유사 기사 제거

```python
clean_news_content(text) -> str
```

- 언론사명, 이메일, SNS ID, 전화번호, URL, 기자명 서명, 특수문자 제거
- 불완전한 마지막 문장 자동 잘라내기 (마지막 `다.` 또는 `.` 이후 제거)

```python
fetch_article_from_url(url, source, timeout_sec) -> pd.DataFrame
```

- 단일 URL 크롤링 (하위 호환성 유지용)

> ⚠️ `fetch_articles_from_naver()`의 반환 포맷은 **절대 변경하지 말 것** (admin_pipeline과의 계약 인터페이스)

---

### 4.5 `repo.py` (DB 레이어)

**역할**

- DuckDB 입출력 전담
- 2개 테이블 관리: `articles`, `article_chunks`

#### 주요 함수

```python
init_db()
```

- `articles` 테이블 (15컬럼) 및 `article_chunks` 테이블 생성

```python
upsert_articles(df: pd.DataFrame)
upsert_chunks(df_chunks: pd.DataFrame)
```

- DataFrame → `INSERT OR REPLACE`

```python
search_similar_chunks(query_vector, limit=10, min_score=0.5, dedupe_per_article=True) -> pd.DataFrame
```

- DuckDB `list_cosine_similarity()`로 벡터 유사도 검색
- `dedupe_per_article=True`: article_id당 최고 점수 청크만 남김 → 한 기사가 결과를 독점하는 문제 해결
- 내부 limit을 `limit * 5`로 넉넉히 잡고 dedupe 후 상위 N개 반환

```python
search_similar_chunks_excluding(query_vector, exclude_article_id, limit=5, min_score=0.5) -> pd.DataFrame
```

- 상세 페이지 전용: 현재 기사를 **SQL 단에서 제외** 후 관련 기사 검색

```python
load_articles() -> pd.DataFrame
```

- 전체 기사 조회 (published_at 내림차순)

```python
load_articles_without_trust() -> pd.DataFrame
```

- `trust_score=0`인 기사만 반환 (신뢰도 일괄 재분석용)

```python
update_article_trust(article_id, score, verdict, reason, per_criteria)
```

- 단일 기사의 trust 컬럼만 UPDATE (일괄 재분석 시 사용)

---

### 4.6 `trust.py` (신뢰도 분석 모듈)

**역할**

- **TELLER 기반 신뢰도 분석 실제 구현** (Gemini Cognitive + Weighted Sum Rule)
- 관리자 파이프라인(크롤링 시 자동) 및 일괄 재분석에서 모두 사용

#### 분석 모델

- 사용 모델: `gemini-2.5-flash` (`response_mime_type: application/json`)
- 5개 기준으로 0~10점 채점 후 가중합산 → 0~100점 종합 점수 산출

#### 가중치 설정

| 기준 | 가중치 | 방향 | 설명 |
|---|---|---|---|
| `source_credibility` | 0.25 | 정방향 | 출처 신뢰성 |
| `evidence_support` | 0.25 | 정방향 | 근거 지지도 |
| `style_neutrality` | 0.20 | 정방향 | 문체 중립성 |
| `logical_consistency` | 0.20 | 정방향 | 논리 일관성 |
| `clickbait_risk` | -0.10 | **역방향** | 어뷰징 위험도 (높을수록 감점) |

#### 판정 기준

| 점수 범위 | 판정 | 배지 |
|---|---|---|
| 70점 이상 | `likely_true` | 🟢 |
| 40~69점 | `uncertain` | 🟡 |
| 39점 이하 | `likely_false` | 🔴 |

```python
score_trust(text: str, source: str | None = None) -> dict
```

- 반환 포맷: `score`, `verdict`, `reason`, `per_criteria` (5개 기준별 점수)
- 429(할당량 초과) 에러 시 30~60초 후 자동 재시도
- Gemini 호출 실패 시 `_fallback()` 으로 score=0, verdict="uncertain" 반환

---

### 4.7 `search.py` (RAG 실험 스크립트)

**역할**

- Gemini 임베딩 + DuckDB 벡터 검색 실험용 독립 스크립트
- `__main__` 블록으로 단독 실행 가능

#### 주요 함수

```python
run_gemini_embedding(text: str, task_type: str = "retrieval_document") -> list
```

- 빈 텍스트나 실패 시 영벡터(768차원) 반환
- 길이 초과/부족 시 잘라내기/패딩 처리

> ⚠️ `search.py`는 실험용이므로 `admin_pipeline.py`와 혼용 시 `task_type` 설정에 주의

---

### 4.8 `data.py`

**역할**

- MOCK 뉴스 데이터 보관 (`MOCK_DB_NORMALIZED`)
- `search.py` 실험 및 초기 테스트용

---

## 5. DB 스키마

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
| keywords | VARCHAR | 핵심 키워드 리스트 (JSON 문자열) |
| embed_full | VARCHAR | 원문 전체 임베딩 |
| embed_summary | VARCHAR | 요약문 임베딩 |
| embed_keywords | VARCHAR | 키워드 임베딩 |
| trust_score | INTEGER | 신뢰도 종합 점수 (0~100) |
| trust_verdict | VARCHAR | 신뢰도 판정 (likely_true / uncertain / likely_false) |
| trust_reason | VARCHAR | 종합 판단 근거 |
| trust_per_criteria | VARCHAR | 기준별 신뢰도 점수 (JSON) |
| status | VARCHAR | 처리 상태 |

### article_chunks 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| chunk_id | VARCHAR (PK) | `{article_id}_{청크번호}` 또는 `{article_id}_title` |
| article_id | VARCHAR | 기사 ID (FK) |
| chunk_text | VARCHAR | 원문 청크 텍스트 (표시용) |
| embedding | FLOAT[768] | Contextual Chunking 임베딩 (검색용) |

> 청크 임베딩은 `[제목] [출처] [카테고리]` prefix가 붙은 텍스트로 생성되지만, `chunk_text`에는 원문만 저장 (UI 표시용)

---

## 6. 검색 품질 튜닝 포인트

현재 코드에 적용된 튜닝 사항 요약입니다.

| 포인트 | 내용 | 적용 위치 |
|---|---|---|
| 임베딩 모델 통일 | `gemini-embedding-001` 단일 모델 사용 | `admin_pipeline.py`, `search.py` |
| task_type 분리 | 적재: `retrieval_document` / 검색: `retrieval_query` | `admin_pipeline.py`, `functions.py` |
| Contextual Chunking | 청크에 제목/출처/카테고리 prefix 부착 | `admin_pipeline.py` |
| 청킹 전략 | `chunk_size=400`, `chunk_overlap=150`, 한국어 sentence separator | `admin_pipeline.py` |
| 제목 전용 청크 | `[제목] {title}` 별도 임베딩 저장 → 키워드 검색 대응 | `admin_pipeline.py` |
| JSON 응답 강제 | response_mime_type을 사용하여 안정적인 요약 및 키워드 파싱 보장 | `admin_pipeline.py` |
| 계층형 키워드 UI | 대분류 > 중분류 > 소분류 형식의 키워드를 파싱해 UI상에서 보기 좋게 그룹화 | `functions.py` |
| 관련 기사 검색 개선 | full_text 대신 제목+요약 임베딩으로 관련 기사 탐색 | `functions.py` |
| SQL단 현재 기사 제외 | `search_similar_chunks_excluding()` 으로 현재 기사 SQL에서 제외 | `repo.py` |
| dedupe_per_article | article_id당 최고 점수 청크만 남김 | `repo.py` |
| min_score 완화 | 기본값 0.5, 검색 결과 페이지 0.65 적용 | `repo.py`, `functions.py` |
| 본문 정제 강화 | 언론사명/이메일/SNS/전화번호/불완전문장 제거 | `crawl.py` |
| 병렬 크롤링 | ThreadPoolExecutor (MAX_WORKERS=10) | `crawl.py` |
| 신뢰도 분석 | TELLER 기반 5개 기준 가중합산, Gemini 2.5 Flash | `trust.py` |

---

## 7. 설계 제약 (중요)

### ✅ UI / 크롤링 / 모델링 / DB 분리

- `admin_pipeline.py`가 백엔드 모든 흐름을 조율
- UI(`functions.py`)는 DB 조회 + Gemini 임베딩 호출만 수행
- 신뢰도 분석(`trust.py`)은 pipeline과 독립적으로 호출 가능 (일괄 재분석 지원)

### ✅ 서버 없는 구조

- Streamlit 실행 프로세스 자체가 서버 역할
- DuckDB는 파일 기반 DB (`app_db.duckdb`)
- API 키는 `.env` + `load_dotenv()` 또는 Streamlit Cloud secrets로 관리

### ✅ API 속도 제한 대응

- 기사당 요약 후 2초 대기
- 기사당 신뢰도 분석 후 2초 대기
- 청크당 임베딩 후 0.5초 대기
- 기사 처리 완료 후 3초 대기
- 429 에러 시 30~60초 랜덤 대기 후 자동 재시도 (요약, 신뢰도 분석 모두 적용)

### ✅ 확장 가능성

- 추후 FastAPI / 서버 환경으로 전환해도 `crawl.py`, `admin_pipeline.py`, `repo.py` 그대로 재사용 가능
- `trust.py`의 반환 포맷 (`score`, `verdict`, `reason`, `per_criteria`) 유지 시 모델 교체 가능

---

## 8. 팀 협업 가이드

| 역할 | 수정 파일 |
|---|---|
| 크롤링 | `crawl.py` |
| 요약 / 임베딩 / 청킹 / 파이프라인 조율 | `admin_pipeline.py` |
| 신뢰도 모델 | `trust.py` |
| UI / 페이지 라우팅 | `functions.py`, `app.py` |
| DB 스키마 / 검색 쿼리 | `repo.py` |
| RAG 실험 | `search.py` |
