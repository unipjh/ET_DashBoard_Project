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
- 성과물:
  - ✔ 로컬 및 URL로 작동하는 서비스 (Streamlit Cloud)
  - ✔ 네이버 뉴스 크롤링 → Gemini 자동 요약 파이프라인
  - ✔ 청크 단위 임베딩 + DuckDB 벡터 검색
  - ✔ 이후 확장 가능한 토대 확보

### 전체 플로우

```
[관리자]
  → 크롤링 버튼 클릭
  → crawl.py: 네이버 뉴스 크롤링
  → admin_pipeline.py: Gemini 요약 + Contextual Chunking + 임베딩 생성
  → repo.py: DuckDB (articles + article_chunks) 적재

[사용자]
  → 메인 페이지: 최신 기사 리스트 조회 (페이지네이션)
  → AI 검색: 검색어 임베딩(retrieval_query) → 유사 청크 검색 → 관련 기사 반환
  → 상세 페이지: 요약문 + 원문 + 관련 기사 추천 (제목+요약 임베딩 기반)
```

### 🎯 진행 중인 기능

- 관리자 버튼을 통해 **네이버 뉴스 크롤링 → Gemini 요약/임베딩 → DB 적재**
- 사용자 페이지에서는 **DB에 저장된 기사만 조회**
- 시맨틱 검색: **검색어 임베딩(retrieval_query) → article_chunks 코사인 유사도 검색**
- 상세 페이지: **제목+요약 임베딩 기반 관련 기사 추천**
- UI, DB, 크롤링, 모델링을 **파일 단위로 명확히 분리**

### 🧠 현재 단계 (MVP)

- 크롤링: 네이버 뉴스 실제 크롤러 (`crawl.py`)
- 요약: Gemini 2.5 Flash Lite (`admin_pipeline.py`)
- 임베딩: Gemini Embedding 001 모델, `task_type` 분리 적용 (`admin_pipeline.py`)
- 청킹: Contextual Chunking (제목/출처/카테고리 prefix) + 제목 전용 청크
- DB: DuckDB 파일 기반 (`articles` + `article_chunks` 테이블)
- 신뢰도: 더미 구현 (`trust.py`)
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
4. 크롤링 → Gemini 요약/임베딩 → DB 적재 자동 진행
5. 사용자 페이지에서 기사 조회 및 AI 검색 가능

---

## 3. 프로젝트 디렉토리 구조

```text
.
├─ app.py                        # Streamlit 엔트리 포인트
├─ app_db.duckdb                 # DuckDB 파일 (자동 생성)
├─ .env                          # Gemini API 키 설정
├─ requirements.txt
│
└─ Streamlit_Rendering/
    ├─ __init__.py
    │
    ├─ functions.py              # UI 렌더링 (사용자 / 관리자 페이지)
    ├─ admin_pipeline.py         # 관리자 파이프라인 (Gemini 요약/임베딩, 청킹)
    ├─ crawl.py                  # 네이버 뉴스 크롤링 전담
    │
    ├─ repo.py                   # DuckDB 입출력 전담
    ├─ data.py                   # MOCK 데이터
    │
    ├─ trust.py                  # 신뢰도 모델 (현재 더미)
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
- DB 조회 및 Gemini 임베딩 호출 (`ap.run_gemini_embedding()`)
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

- 좌: AI 요약 + 관련 기사 추천
  - 관련 기사: **제목+요약문** 임베딩으로 검색 (기존 full_text 대비 노이즈 감소)
  - `repo.search_similar_chunks_excluding()` 으로 현재 기사 SQL단에서 제외
- 우: 출처/날짜/원문 링크 + 기사 전문

```python
render_admin_page()
```

- 카테고리당 기사 수 설정
- 크롤링 → 요약/임베딩 → DB 적재 버튼
- DB 현황 (총 기사 수, 소스 종류, 최신 기사 목록)

---

### 4.3 `admin_pipeline.py` (관리자 파이프라인)

**역할**

- 크롤링된 기사를 받아 Gemini 요약 + Contextual Chunking + 임베딩 생성 후 DB 적재
- `.env`의 `GEMINI_API_KEY`를 자동으로 로드
- 임베딩 모델: `models/gemini-embedding-001` (768차원), `task_type` 파라미터로 용도 분리

#### 주요 함수

```python
run_gemini_summary(text: str) -> str
```

- Gemini 2.5 Flash Lite로 뉴스 3문장 요약
- 429(할당량 초과) 에러 시 30~60초 후 자동 재시도

```python
run_gemini_embedding(text: str, task_type: str = "retrieval_document") -> list
```

- `task_type` 파라미터로 용도 분리
  - 문서 적재 시: `"retrieval_document"` (기본값)
  - 검색 쿼리 시: `"retrieval_query"` (함수 호출 측에서 지정)
- 빈 텍스트나 실패 시 영벡터(768차원) 반환

```python
_make_chunk_context(title, source, category, chunk_text) -> str
```

- **Contextual Chunking**: 청크에 `[제목] [출처] [카테고리]` prefix 부착
- 청크가 어떤 기사의 일부인지 의미 정보가 임베딩에 반영됨 → 검색 정확도 향상

```python
build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int
```

- 청킹 전략: `chunk_size=400`, `chunk_overlap=150`, 한국어 문장 단위 separator 적용
- 기사별: Gemini 요약 → DB 저장 (`articles`)
- 청크별: Contextual Chunking → `retrieval_document` 임베딩 → DB 저장 (`article_chunks`)
- **제목 전용 청크 추가**: `[제목] {title}` 형태로 별도 임베딩 저장 → 짧은 키워드 검색 대응
- API 속도 제한 회피: 요약 후 2초, 청크당 0.5초, 기사 완료 후 3초 대기

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
- 본문 정제, 중복 제거, 유사 기사 필터링 포함

#### 주요 함수

```python
fetch_articles_from_naver(max_articles_per_category=30) -> pd.DataFrame
```

- 8개 카테고리(정치/경제/사회/생활·문화/세계/IT·과학/연예/스포츠) 병렬 크롤링
- 반환 컬럼: `date`, `category`, `source`, `title`, `reporter`, `comment_cnt`, `like_cnt`, `link`, `content`
- 200자 미만 본문 필터링, Jaccard 유사도 기반 중복 제거 적용

> ⚠️ 반환 포맷은 **절대 변경하지 말 것** (admin_pipeline과의 계약 인터페이스)

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
- 기존 Python 단 필터링 대비 limit 개수를 정확하게 채울 수 있음

```python
load_articles() -> pd.DataFrame
```

- 전체 기사 조회 (published_at 내림차순)

---

### 4.6 `trust.py` (신뢰도 모델)

**역할**

- 신뢰도 모델 구현 위치 (현재 더미)
- 관리자 파이프라인에서만 사용 예정

```python
score_trust_dummy(text, source, low=30, high=100) -> dict
```

- 30~100 사이 무작위 점수 반환
- 반환 포맷: `score`, `verdict`, `reason`, `per_criteria` (5개 기준별 점수)
- 추후 실제 모델 교체 시 동일 반환 포맷 유지

---

### 4.7 `search.py` (RAG 실험 스크립트)

**역할**

- Gemini 임베딩 + DuckDB 벡터 검색 실험용 독립 스크립트
- `__main__` 블록으로 단독 실행 가능

#### 주요 함수

```python
get_gemini_embedding(text) -> list
```

- `task_type="retrieval_query"` 고정 (검색용)

```python
run_experiment()
```

- MOCK_DB에서 기사 5개 처리 → 청크 분할 → 임베딩 → DB 저장

```python
search_and_analyze(query)
```

- 검색어 임베딩 → 코사인 유사도 상위 3개 청크 검색
- Gemini로 최종 분석 브리핑 생성 (Explainable AI)

> ⚠️ API 키가 하드코딩되어 있음 → `.env` + `load_dotenv()` 방식으로 교체 필요  
> ⚠️ 문서 적재 시 `task_type="retrieval_document"` 미적용 → `admin_pipeline.py`와 혼용 시 유사도 왜곡 가능

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
| keywords | VARCHAR | 키워드 (JSON, 미사용) |
| embed_full | VARCHAR | 전문 임베딩 (미사용) |
| embed_summary | VARCHAR | 요약 임베딩 (미사용) |
| trust_score | INTEGER | 신뢰도 점수 (더미) |
| trust_verdict | VARCHAR | 신뢰도 판정 |
| trust_reason | VARCHAR | 판정 근거 |
| trust_per_criteria | VARCHAR | 기준별 신뢰도 (JSON) |
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
| 임베딩 모델 통일 | `gemini-embedding-001` 단일 모델 사용 | `admin_pipeline.py` |
| task_type 분리 | 적재: `retrieval_document` / 검색: `retrieval_query` | `admin_pipeline.py`, `functions.py` |
| Contextual Chunking | 청크에 제목/출처/카테고리 prefix 부착 | `admin_pipeline.py` |
| 청킹 전략 | `chunk_size=400`, `chunk_overlap=150`, 한국어 sentence separator | `admin_pipeline.py` |
| 제목 전용 청크 | `[제목] {title}` 별도 임베딩 저장 → 키워드 검색 대응 | `admin_pipeline.py` |
| 관련 기사 검색 개선 | full_text 대신 제목+요약 임베딩으로 관련 기사 탐색 | `functions.py` |
| SQL단 현재 기사 제외 | `search_similar_chunks_excluding()` 으로 현재 기사 SQL에서 제외 | `repo.py` |
| dedupe_per_article | article_id당 최고 점수 청크만 남김 | `repo.py` |
| min_score 완화 | 기본값 0.7 → 0.5, 검색 결과 페이지 0.65 적용 | `repo.py`, `functions.py` |

---

## 7. 설계 제약 (중요)

### ✅ UI / 크롤링 / 모델링 / DB 분리

- `admin_pipeline.py`가 백엔드 모든 흐름을 조율
- UI(`functions.py`)는 DB 조회 + Gemini 임베딩 호출만 수행

### ✅ 서버 없는 구조

- Streamlit 실행 프로세스 자체가 서버 역할
- DuckDB는 파일 기반 DB (`app_db.duckdb`)
- API 키는 `.env` + `load_dotenv()`로 관리

### ✅ API 속도 제한 대응

- 기사당 요약 후 2초 대기
- 청크당 임베딩 후 0.5초 대기
- 기사 처리 완료 후 3초 대기
- 429 에러 시 30~60초 랜덤 대기 후 자동 재시도

### ✅ 확장 가능성

- 추후 FastAPI / 서버 환경으로 전환해도 `crawl.py`, `admin_pipeline.py`, `repo.py` 그대로 재사용 가능
- `trust.py`의 더미 구현을 실제 모델로 교체 시 반환 포맷 유지

---

## 8. 팀 협업 가이드

| 역할 | 수정 파일 |
|---|---|
| 크롤링 | `crawl.py`, `admin_pipeline.py` |
| 요약 / 임베딩 / 청킹 | `admin_pipeline.py` |
| 신뢰도 모델 | `trust.py`, `admin_pipeline.py` |
| UI / 페이지 라우팅 | `functions.py`, `app.py` |
| DB 스키마 / 검색 쿼리 | `repo.py` |
| RAG 실험 | `search.py` |
