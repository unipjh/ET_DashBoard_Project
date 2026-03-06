# TD1.md — TELLER 기반 신뢰도 모듈 구현 완료 보고서

> 작성일: 2026-03-02
> 기반 계획: TD_explain.md § TD1

---

## 개요

`Streamlit_Rendering/trust.py`의 더미 구현을 TELLER 논문 구조(Cognitive System → Decision System → Explainability)를 차용하여 Gemini API 기반 실질 신뢰도 분석 모듈로 교체했습니다.

---

## 변경 파일 목록

| 파일 | 변경 유형 | 주요 내용 |
|------|---------|---------|
| `Streamlit_Rendering/trust.py` | 전면 재작성 | Gemini JSON 모드 + Weighted Sum Rule |
| `Streamlit_Rendering/admin_pipeline.py` | 수정 | 크롤링 파이프라인에 `score_trust()` 연동 |
| `Streamlit_Rendering/repo.py` | 수정 | trust UPDATE 전용 함수 2개 추가 |
| `Streamlit_Rendering/functions.py` | 수정 | 상세 페이지 trust 패널 + Admin 일괄 분석 버튼 |

---

## trust.py — 핵심 구현

### 아키텍처 매핑

| TELLER 논문 구성 요소 | 본 구현 |
|----------------------|---------|
| Cognitive System | Gemini API (`gemini-2.5-flash`) |
| Decision System | Weighted Sum Rule (학습 데이터 불필요) |
| Explainability | `per_criteria` dict + `overall_reason` 텍스트 |

### 신뢰도 기준 및 가중치

```python
WEIGHTS = {
    "source_credibility":  0.25,   # 출처 신뢰성
    "evidence_support":    0.25,   # 근거 지지도
    "style_neutrality":    0.20,   # 문체 중립성
    "logical_consistency": 0.20,   # 논리 일관성
    "clickbait_risk":     -0.10,   # 어뷰징 위험도 (역방향)
}
```

가중치는 파일 상단 상수로 노출되어 있어 수동 조정 가능합니다.

### Gemini 호출 방식

- `response_mime_type="application/json"` 설정으로 JSON 직접 수신 (파싱 오류 최소화)
- 본문 상한: `text[:3000]` (토큰 절약)
- 단일 프롬프트로 5개 기준 점수·이유 + 종합 이유(`overall_reason`) 동시 추출 → **기사 1건당 1회 호출**

### 점수 정규화 (Weighted Sum Rule)

```python
raw = sum(criteria[k]["score"] * w for k, w in WEIGHTS.items())
# 최대값 9.0 = (0.25+0.25+0.20+0.20) × 10 (clickbait=0일 때)
score = int(max(0, min(100, raw / 9.0 * 100)))
```

### Verdict 기준

| score | verdict |
|-------|---------|
| ≥ 70 | `likely_true` |
| 40 ~ 69 | `uncertain` |
| < 40 | `likely_false` |

### Rate Limit 처리

429 또는 `"Quota exceeded"` 예외 시 30~60초 랜덤 sleep 후 재시도.
그 외 예외는 fallback 반환 (`score=0, verdict="uncertain"`).

### 반환 포맷 (하위 호환 유지)

```python
{
    "score":        int,   # 0~100
    "verdict":      str,   # "likely_true" | "uncertain" | "likely_false"
    "reason":       str,   # 종합 판단 근거 (Gemini 생성)
    "per_criteria": {
        "source_credibility":  {"score": int, "reason": str},
        "evidence_support":    {"score": int, "reason": str},
        "style_neutrality":    {"score": int, "reason": str},
        "logical_consistency": {"score": int, "reason": str},
        "clickbait_risk":      {"score": int, "reason": str},
    }
}
```

---

## admin_pipeline.py — 파이프라인 연동

`build_ready_rows_from_naver()` 내부에서 요약 생성 직후 `score_trust()` 호출:

```
기사 1건 처리 순서:
  1. run_gemini_summary()       → summary_text         [Gemini 1회]
  2. time.sleep(2)
  3. score_trust()              → trust_score/verdict/reason/per_criteria [Gemini 1회]
  4. time.sleep(2)
  5. run_gemini_embedding() × N → article_chunks        [Gemini N회]
```

`trust_per_criteria`는 `json.dumps(..., ensure_ascii=False)`로 직렬화하여 VARCHAR 컬럼 저장.

### 기사 1건당 총 Gemini 호출 횟수

| 단계 | 호출 수 |
|------|------:|
| 요약 | 1회 |
| 신뢰도 분석 (TD1 신규) | 1회 |
| 청크 임베딩 (평균) | 2~3회 |
| **합계** | **4~5회** |

---

## repo.py — 신규 함수

### `load_articles_without_trust()`

```python
SELECT * FROM articles WHERE trust_score = 0 ORDER BY published_at DESC
```

Admin 일괄 분석 시 미분석 기사만 선별하는 용도.

### `update_article_trust(article_id, score, verdict, reason, per_criteria)`

```python
UPDATE articles
SET trust_score = ?, trust_verdict = ?, trust_reason = ?, trust_per_criteria = ?
WHERE article_id = ?
```

개별 기사의 trust 컬럼만 UPDATE (기존 `upsert_articles` 전체 재삽입 대신 효율적 단건 업데이트).

---

## functions.py — UI 변경

### 상세 페이지 — 신뢰도 패널 (`render_detail_page`)

기사 본문 상단에 신뢰도 패널 추가:

```
┌─────────────────────────────────────────────────────┐
│  🔐 신뢰도 분석                                       │
│  종합 점수: 75점/100    판정: 🟢 likely_true           │
│  ████████████████████░░░░  (progress bar)            │
│                                                       │
│  ▼ 기준별 세부 점수 보기                               │
│    출처 신뢰성     ████████░░  8/10  [판단 근거]       │
│    근거 지지도     ██████░░░░  6/10  [판단 근거]       │
│    문체 중립성     █████████░  9/10  [판단 근거]       │
│    논리 일관성     ███████░░░  7/10  [판단 근거]       │
│    어뷰징 위험도   ██░░░░░░░░  2/10  [판단 근거]       │
│                                                       │
│  📝 종합 판단: ...                                    │
└─────────────────────────────────────────────────────┘
```

- `trust_score=0` && `verdict ∈ {"", "None", "uncertain"}` 이면 미분석 안내 표시
- `trust_per_criteria`는 `json.loads()`로 파싱하여 기준별 렌더링

### Admin 페이지 — 신뢰도 일괄 분석 (`render_admin_page`)

크롤링 섹션 하단에 "🔍 신뢰도 일괄 분석" 섹션 추가:

- 미분석 기사 건수 실시간 표시 (`trust_score=0` 카운트)
- "신뢰도 일괄 분석 시작" 버튼 → 기사별 순차 분석 + 진행 바
- 분석 완료 시 `st.rerun()`으로 카운트 자동 갱신
- 기사별 실패 시 `st.warning()`으로 개별 표시 (전체 중단 없음)

---

## 마일스톤 달성 현황

| 마일스톤 | 상태 | 비고 |
|---------|:----:|------|
| M1: Gemini 프롬프트 설계 및 per_criteria JSON 추출 | ✅ | JSON 모드 적용 |
| M2: Weighted Sum Rule 가중치 조정 및 score 정규화 | ✅ | clamp(0, 100) 적용 |
| M3: `score_trust()` 함수 완성 | ✅ | 더미 함수 대체 |
| M4: admin_pipeline.py 연동 및 DB 반영 | ✅ | 크롤링 시 자동 분석 |
| M4+: 기존 기사 일괄 재분석 기능 | ✅ | Admin 버튼 추가 (계획 외 추가) |
| M5: 실기사 10건 이상 수동 검증 | 🔲 | 미진행 |
| (TD1.5) RF + AI Hub 고도화 | 🔲 | 별도 단계 |

---

## 환경 설정

`.env` 파일에 API 키 필요:

```
GEMINI_API_KEY=your_key_here
```

`trust.py`와 `admin_pipeline.py` 모두 `load_dotenv()`로 로드.

---

## 알려진 제한사항

- `google.generativeai` 패키지가 deprecated됨 (FutureWarning). 향후 `google.genai`로 마이그레이션 필요 → TD2 백엔드 작업 시 함께 처리 예정
- Weighted Sum Rule은 Gemini 점수 품질에 전적으로 의존 (모델 환각 시 점수 신뢰도 저하 가능)
- 기사 본문 3000자 이후는 분석에서 제외됨 (긴 기사의 후반부 내용 미반영)
