# CLAUDE.md — ET_by_claude 프로젝트 가이드

> ET (Explainable Trust) — 한국어 뉴스 신뢰도 분석 플랫폼
> Python 3.11 | Streamlit 1.49.1 | DuckDB | Gemini API

---

## 프로젝트 개요

Naver News를 크롤링하고, Gemini API로 요약·임베딩·신뢰도를 분석하여
사용자에게 뉴스 신뢰도를 설명 가능한 형태로 제공하는 플랫폼.

---

## 핵심 파일 구조

```
ET_by_claude/
├── app.py                              # Streamlit 진입점, 세션 라우팅
├── requirements.txt
├── TD_explain.md                       # 기술 계획 문서 (TD1, TD2)
├── Streamlit_Rendering/
│   ├── __init__.py
│   ├── admin_pipeline.py               # Gemini 요약/임베딩 오케스트레이션
│   ├── crawl.py                        # Naver News 멀티스레드 크롤러
│   ├── data.py                         # MOCK_DB (레거시 참조용)
│   ├── functions.py                    # 전체 UI 렌더링 (4개 페이지)
│   ├── repo.py                         # DuckDB CRUD 레이어
│   ├── search.py                       # RAG 시맨틱 검색 (Gemini embedding)
│   └── trust.py                        # 신뢰도 모듈 (현재 더미 → TD1 교체 예정)
└── ET_DashBoard_Project/               # 미사용 (빈 디렉토리)
```

---

## 실행 방법

```bash
# 환경변수 설정 (.env 파일에 GEMINI_API_KEY 작성)
echo "GEMINI_API_KEY=your_key_here" > .env

# 앱 실행
streamlit run app.py
```

---

## DB 스키마

**articles 테이블**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| article_id | VARCHAR PK | `naver_YYYYMMDD_HHMMSS_{idx}` 형식 |
| title | VARCHAR | 기사 제목 |
| source | VARCHAR | 언론사 |
| url | VARCHAR | 원문 URL |
| published_at | VARCHAR | 발행일 |
| full_text | VARCHAR | 기사 본문 전체 |
| summary_text | VARCHAR | Gemini 3문장 요약 |
| keywords | VARCHAR | JSON 배열 문자열 |
| embed_full | VARCHAR | (미사용, 예비 컬럼) |
| embed_summary | VARCHAR | (미사용, 예비 컬럼) |
| trust_score | INTEGER | 신뢰도 점수 0~100 |
| trust_verdict | VARCHAR | `likely_true` / `uncertain` / `likely_false` |
| trust_reason | VARCHAR | 신뢰도 판단 근거 |
| trust_per_criteria | VARCHAR | JSON — 5개 기준별 점수+이유 |
| status | VARCHAR | `ready` |

**article_chunks 테이블**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| chunk_id | VARCHAR PK | `{article_id}_{i}` 또는 `{article_id}_title` |
| article_id | VARCHAR | FK → articles |
| chunk_text | VARCHAR | 청크 원문 (표시용) |
| embedding | FLOAT[768] | Contextual Chunking 임베딩 (검색용) |

---

## Gemini API 사용 구조

| 모듈 | 모델 | 용도 |
|------|------|------|
| `admin_pipeline.py` | `gemini-2.5-flash-lite` | 기사 요약 |
| `admin_pipeline.py` | `models/gemini-embedding-001` | 청크 임베딩 저장 (`retrieval_document`) |
| `functions.py` | `models/gemini-embedding-001` | 검색 쿼리 임베딩 (`retrieval_query`) |
| `trust.py` (TD1 후) | `gemini-2.5-flash` | 신뢰도 5기준 점수 추출 |

**API 키 설정**: `.env` 파일에 `GEMINI_API_KEY` 기재 → `admin_pipeline.py`에서 `load_dotenv()`로 로드.

---

## RAG 파이프라인 핵심 설계

### 임베딩 전략 (admin_pipeline.py)

- **Contextual Chunking**: `[제목] [출처] [카테고리]` prefix를 청크에 추가 후 임베딩
- **청크 설정**: `chunk_size=400`, `chunk_overlap=150`, 한국어 문장 우선 분할
- **제목 전용 청크**: 각 기사에 `{article_id}_title` 청크 별도 추가 (키워드 검색 강화)

### 검색 (repo.py)

- `search_similar_chunks()`: 코사인 유사도 ≥ 0.5, article당 best chunk만 반환 (dedupe)
- `search_similar_chunks_excluding()`: 상세 페이지 전용, 현재 기사 SQL 레벨 제외

### UI 라우팅 (app.py + functions.py)

```
session_state["admin_mode"] == True      → render_admin_page()
session_state["selected_article_id"]     → render_detail_page(aid)
session_state["search_executed"] == True → render_search_results_page(query)
else                                     → render_main_page()
```

---

## 진행 중인 기술 계획 (TD_explain.md 참조)

### TD1 — TELLER 기반 신뢰도 모듈

**목표**: `trust.py`의 더미 함수 → 실질적 분석 모듈로 교체

| 단계 | 내용 |
|------|------|
| Cognitive System | Gemini API로 5개 기준별 점수(0~10) + 이유 추출 |
| Decision System | 가중 합산 규칙 (학습 데이터 불필요) |
| Explainability | `per_criteria` dict + 종합 `reason` 텍스트 |

**5개 신뢰도 기준**:
- `source_credibility` (0.25) — 출처 신뢰성
- `evidence_support` (0.25) — 근거 지지도
- `style_neutrality` (0.20) — 문체 중립성
- `logical_consistency` (0.20) — 논리 일관성
- `clickbait_risk` (-0.10) — 어뷰징 위험도 (역방향)

**반환 포맷** (하위 호환 유지):
```python
{
    "score": int,           # 0~100
    "verdict": str,         # "likely_true" | "uncertain" | "likely_false"
    "reason": str,
    "per_criteria": {
        "source_credibility":  {"score": int, "reason": str},
        "evidence_support":    {"score": int, "reason": str},
        "style_neutrality":    {"score": int, "reason": str},
        "logical_consistency": {"score": int, "reason": str},
        "clickbait_risk":      {"score": int, "reason": str},
    }
}
```

**변경 파일**: `trust.py` 전면 재작성, `admin_pipeline.py` 호출부 수정

---

### TD2 — React + FastAPI 마이그레이션

**목표**: Streamlit 단일 앱 → 프론트/백엔드 분리

```
frontend/   React (Vite + TypeScript)   → localhost:5173
backend/    FastAPI (uvicorn)           → localhost:8000
DB          DuckDB 유지 (스키마 변경 없음)
```

**작업 우선순위**: TD1 완료 → TD2 백엔드 → TD2 프론트엔드 → 통합

---

## 코딩 컨벤션

- **DB 연결**: 함수 내부에서 `duckdb.connect()` → `con.close()` 패턴 (커넥션 풀 미사용)
- **임베딩 실패 처리**: `[0.0] * 768` 제로벡터 반환 (예외 삼킴)
- **Gemini Rate Limit**: 429 오류 시 30~60초 랜덤 sleep 후 재시도
- **기사 ID**: `naver_YYYYMMDD_HHMMSS_{idx}` 형식
- **신뢰도 verdict 기준**: score ≥ 70 → `likely_true`, ≥ 40 → `uncertain`, < 40 → `likely_false`

---

## 주의사항

- `search.py` 내 `GEMINI_API_KEY`는 하드코딩 잔재가 있음 → `.env`로 통일 필요
- `embed_full`, `embed_summary` 컬럼은 현재 `"[]"` 빈값으로 저장됨 (미사용)
- `ET_DashBoard_Project/` 디렉토리는 현재 비어 있음 (TD2 작업 예정 위치 아님)
- DuckDB 파일(`app_db.duckdb`)은 `.gitignore`에 포함되어 버전 관리 제외
