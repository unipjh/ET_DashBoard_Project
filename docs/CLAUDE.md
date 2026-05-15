# CLAUDE.md — ET (Explainable Trust)

> 한국어 뉴스 신뢰도 분석 플랫폼 | Python 3.11 | FastAPI + React (Vite + TS) | DuckDB | Gemini API

## 구조

```
backend/
├── main.py                 # FastAPI 진입점, CORS
├── schemas.py              # Pydantic 모델
├── routers/
│   ├── articles.py         # GET /api/articles, /api/articles/{id}, /api/articles/{id}/related
│   ├── search.py           # POST /api/search
│   ├── admin.py            # GET /api/admin/stats, POST /api/admin/crawl·analyze
│   └── trust.py            # GET /api/trust/{id}
└── services/
    ├── config.py           # get_gemini_api_key()
    ├── repo.py             # DuckDB CRUD
    ├── crawl.py            # Naver News 크롤러
    ├── admin_pipeline.py   # 요약/임베딩 오케스트레이션
    └── trust.py            # 신뢰도 모듈 ← 현재 작업 대상

frontend/src/
├── api/client.ts           # axios (VITE_API_URL)
├── components/             # ArticleCard, TrustGauge
└── pages/                  # MainPage, DetailPage, AdminPage

docs/
├── CLAUDE.md               # 이 파일
├── trust_logic.md          # 신뢰도 모듈 원본 설계
└── trust_revise.md         # trust.py 리팩토링 상세 지침 ← 작업 전 반드시 읽기
```

## DB 핵심 컬럼 (articles)

`trust_score INTEGER` / `trust_verdict VARCHAR` / `trust_reason VARCHAR` / `trust_per_criteria VARCHAR(JSON)`

## trust.py 반환 스키마 — 변경 금지

```python
{
    "score": int,        # 0~100
    "verdict": str,      # "likely_true" | "uncertain" | "likely_false"
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

## Gemini 모델

| 용도 | 모델 |
|------|------|
| 요약 | `gemini-2.5-flash-lite` |
| 임베딩 | `models/gemini-embedding-001` |
| 신뢰도 | `gemini-2.5-flash` |

## 컨벤션

- DB: 함수 내 `duckdb.connect()` → `con.close()` (풀 미사용)
- Rate limit: 429 시 30~60초 랜덤 sleep 재시도
- verdict 기준: ≥70 `likely_true` / ≥40 `uncertain` / <40 `likely_false`
- 백그라운드 작업: FastAPI `BackgroundTasks`
- API 키: `.env`의 `GEMINI_API_KEY` → `config.py`
