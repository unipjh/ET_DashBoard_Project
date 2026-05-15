# 시스템 개요

ET(Explainable Trust)는 네이버 뉴스 기사를 수집하고, AI로 신뢰도를 분석하며, RAG 기반 시맨틱 검색을 제공하는 뉴스 신뢰도 평가 웹 서비스다.

---

## 전체 흐름

```mermaid
flowchart TD
    A[네이버 뉴스] -->|HTTP 크롤링| B[crawl.py\n멀티스레드 크롤러]
    B -->|기사 원문| C[admin_pipeline.py\n전처리 파이프라인]

    C -->|요약·키워드| D[Gemini 2.5 Flash\nLLM]
    C -->|신뢰도 분석| D
    C -->|텍스트 임베딩| E[Gemini Embedding\nmodels/gemini-embedding-001]

    D -->|분석 결과| F[(DuckDB\narticles 테이블)]
    E -->|768차원 벡터| G[(DuckDB\narticle_chunks 테이블)]
    F --- G

    G -->|시맨틱 검색| H[search.py\n하이브리드 검색]
    F -->|BM25 검색| H

    F -->|REST API| I[FastAPI 백엔드]
    H -->|검색 결과| I

    I -->|JSON| J[React 프론트엔드]
    J -->|신뢰도 시각화| K[사용자]
```

---

## 주요 컴포넌트

| 컴포넌트 | 위치 | 역할 |
|---------|------|------|
| 크롤러 | `backend/services/crawl.py` | 네이버 뉴스 8개 카테고리 수집 (ThreadPoolExecutor) |
| 신뢰도 분석 | `backend/services/trust.py` | TELLER 기반 5개 기준 AI 평가 |
| 전처리 파이프라인 | `backend/services/admin_pipeline.py` | 요약·키워드·임베딩·청킹 오케스트레이션 |
| 저장소 | `backend/services/repo.py` | DuckDB CRUD + 벡터 검색 + BM25 |
| API 서버 | `backend/main.py` + `routers/` | FastAPI REST API |
| 프론트엔드 | `frontend/src/` | React 19 + TanStack Query + Tailwind |
| 평가셋 빌더 | `crawl_exp/` | SNU 팩트체크 기반 검증 데이터 구축 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **백엔드** | Python 3, FastAPI, Uvicorn |
| **AI/LLM** | Google Gemini 2.5 Flash (분석·요약), Gemini Embedding 001 (임베딩) |
| **DB** | DuckDB (로컬), MotherDuck (클라우드) |
| **검색** | DuckDB cosine_similarity (시맨틱), rank-bm25 (BM25), RRF (하이브리드) |
| **프론트엔드** | React 19, TypeScript, Vite, TanStack Query, Tailwind CSS, Axios |
| **크롤링** | requests, BeautifulSoup4, ThreadPoolExecutor |
| **배포** | Heroku (Procfile: uvicorn) |
| **외부 API** | Naver 뉴스 검색 API, Naver Like/Comment API |
