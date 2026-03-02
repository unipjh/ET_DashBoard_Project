# TD_explain.md — 실험 계획 문서

> 작성일: 2026-02-25
> 프로젝트: ET_by_claude (Explainable Trust News Platform)
> 현재 스택: Streamlit + DuckDB + Gemini API (Python 3.11)

---

## TD1 — TELLER 프레임워크 기반 신뢰도 모듈 구현

### 배경

현재 `Streamlit_Rendering/trust.py`는 30~100 사이 난수를 반환하는 더미 구현입니다.
이를 2024년 TELLER 논문의 프레임워크 구조를 차용하여 실질적인 신뢰도 분석 모듈로 교체합니다.

### 참고 논문

**TELLER: A Trustworthy Framework For Explainable, Generalizable and Controllable Fake News Detection (2024)**

논문의 핵심 아키텍처 3단계:

```
[Cognitive System]  →  [Decision System]  →  [Explainability Layer]
  (LLM 기반 특징 추출)    (분류기)               (판단 근거 제공)
```

| 논문 구성 요소      | 본 프로젝트 구현 방식                       |
|--------------------|--------------------------------------------|
| Cognitive System   | Gemini API (gemini-2.5-flash)              |
| Decision System    | sklearn Random Forest Classifier           |
| Explainability     | per_criteria dict + Gemini 설명 텍스트     |

---

### 구현 계획

#### Phase 1 — Feature Extraction (Cognitive System)

Gemini API를 통해 기사 텍스트에서 5개 기준별 점수(0~10)를 추출합니다.

| 기준 (Criterion)       | 설명                                                    |
|------------------------|---------------------------------------------------------|
| `source_credibility`   | 출처 신뢰성 — 언론사 규모, 공신력                        |
| `evidence_support`     | 근거 지지도 — 수치, 인용, 전문가 등장 여부               |
| `style_neutrality`     | 문체 중립성 — 감정적·선동적 표현 여부                    |
| `logical_consistency`  | 논리 일관성 — 전후 문맥, 주장 일관성                     |
| `clickbait_risk`       | 어뷰징 위험도 — 자극적 제목 / 내용 불일치 여부           |

Gemini 프롬프트 출력 형식 (JSON):
```json
{
  "source_credibility": {"score": 7, "reason": "..."},
  "evidence_support":   {"score": 5, "reason": "..."},
  "style_neutrality":   {"score": 8, "reason": "..."},
  "logical_consistency":{"score": 6, "reason": "..."},
  "clickbait_risk":     {"score": 3, "reason": "..."}
}
```

#### Phase 2 — Classification (Decision System)

추출된 5개 점수를 입력으로 최종 신뢰도 점수와 verdict를 산출합니다.

```
입력: [source_credibility, evidence_support, style_neutrality,
       logical_consistency, clickbait_risk]  (각 0~10)
출력: score (0~100), verdict (likely_true / uncertain / likely_false)
```

##### Decision System 후보 비교

| 후보 | 학습 데이터 | 구현 복잡도 | 설명 가능성 | 비고 |
|------|:-----------:|:-----------:|:-----------:|------|
| **A. 가중 합산 규칙** | 불필요 | 최저 | 명확 | **TD1 베이스라인 채택** |
| B. Logistic Regression | 소량 합성 | 낮음 | 높음 | TD1 고도화 시 후보 |
| C. Random Forest + AI Hub 데이터 | 대용량 외부 | 높음 | 중간 | TD1 이후 별도 단계 |

> **RF + AI Hub 낚시성 기사 탐지 데이터**를 활용하는 방식은 외부 데이터셋 수집·전처리·학습
> 파이프라인이 별도로 필요하여 베이스라인 목적에 과함. TD1 이후 고도화 단계(TD1.5)로 분리.

##### 채택: A. 가중 합산 규칙 (Weighted Sum Rule)

Gemini가 이미 의미 있는 점수를 반환하므로, 학습 데이터 없이 가중치 합산만으로 즉시 작동.

```python
# clickbait_risk는 높을수록 신뢰도 감소 → 역전 처리
weights = {
    "source_credibility":  0.25,
    "evidence_support":    0.25,
    "style_neutrality":    0.20,
    "logical_consistency": 0.20,
    "clickbait_risk":     -0.10,   # 역방향 (높을수록 감점)
}
# raw_score: 0~10 스케일 → 0~100으로 정규화
score = int(sum(criteria[k]["score"] * w for k, w in weights.items()) / 9.0 * 100)
```

가중치는 수동 조정 가능하도록 `trust.py` 상단 상수로 노출.

#### Phase 3 — 최종 출력 형식

현재 더미와 동일한 반환 포맷 유지 (하위 호환성 보장):

```python
{
    "score": int,           # 0~100
    "verdict": str,         # "likely_true" | "uncertain" | "likely_false"
    "reason": str,          # Gemini가 생성한 종합 판단 이유
    "per_criteria": {
        "source_credibility":   {"score": int, "reason": str},
        "evidence_support":     {"score": int, "reason": str},
        "style_neutrality":     {"score": int, "reason": str},
        "logical_consistency":  {"score": int, "reason": str},
        "clickbait_risk":       {"score": int, "reason": str},
    }
}
```

---

### 파일 변경 범위

| 파일                                    | 변경 내용                                          |
|-----------------------------------------|----------------------------------------------------|
| `Streamlit_Rendering/trust.py`          | 전면 재작성 (Gemini Cognitive + Weighted Sum Rule) |
| `requirements.txt`                      | 추가 의존성 없음 (numpy, google-generativeai 기존 사용) |
| `Streamlit_Rendering/admin_pipeline.py` | trust 점수 산출 호출부 수정 (더미 → 실제 함수)     |

> **삭제**: `models/trust_rf.pkl`, `models/train_trust_rf.py` — Weighted Sum Rule은 모델 파일 불필요.
> RF 고도화는 TD1.5 이후로 분리.

---

### 마일스톤

- [ ] M1: Gemini 프롬프트 설계 및 per_criteria JSON 추출 테스트
- [ ] M2: Weighted Sum Rule 가중치 조정 및 score 정규화 검증
- [ ] M3: `score_trust()` 함수 완성 (더미 함수 대체)
- [ ] M4: admin_pipeline.py 연동 및 DB 반영 확인
- [ ] M5: 실기사 10건 이상 수동 검증
- [ ] (TD1.5) RF + AI Hub 낚시성 기사 탐지 데이터 고도화 — 별도 단계

---

---

## TD2 — Streamlit → React + FastAPI 마이그레이션

### 배경 및 동기

| 문제점                          | 설명                                                              |
|---------------------------------|-------------------------------------------------------------------|
| UI 제약                         | Streamlit 컴포넌트 커스터마이징 불가, CSS 직접 제어 어려움        |
| 렌더링 방식                     | 상태 변경 시 전체 페이지 리렌더링 → UX 저하                       |
| 확장성                          | 복잡한 인터랙션 (드래그앤드롭, 실시간 업데이트 등) 구현 불가      |
| 배포                            | Streamlit Cloud 종속성, 멀티 사용자 세션 관리 불안정              |
| 개발 생산성                     | 프론트/백엔드 분리 불가 → 테스트·유지보수 어려움                  |

### 목표 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Browser                        │
│              React (Vite + TypeScript)                      │
│           npm run dev  →  http://localhost:5173             │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│              uvicorn  →  http://localhost:8000              │
│                                                             │
│  /api/articles     GET  - 기사 목록                         │
│  /api/articles/{id} GET  - 기사 상세                        │
│  /api/search       POST - 시맨틱 검색                       │
│  /api/admin/crawl  POST - 크롤링 실행                       │
│  /api/admin/stats  GET  - DB 통계                           │
│  /api/trust/{id}   GET  - 신뢰도 분석 (TD1 연동)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   DuckDB    │
                    │ app_db.duckdb│
                    └─────────────┘
```

---

### 디렉토리 구조 (마이그레이션 후)

```
ET_by_claude/
│
├── backend/                        # FastAPI 서버
│   ├── main.py                     # FastAPI app 진입점, CORS 설정
│   ├── routers/
│   │   ├── articles.py             # 기사 CRUD 라우터
│   │   ├── search.py               # 검색 라우터
│   │   ├── admin.py                # 어드민 라우터
│   │   └── trust.py                # 신뢰도 라우터 (TD1 연동)
│   ├── services/                   # 비즈니스 로직 (현 Streamlit_Rendering/ 이전)
│   │   ├── crawl.py
│   │   ├── search.py
│   │   ├── trust.py
│   │   └── admin_pipeline.py
│   ├── db/
│   │   └── repo.py                 # DuckDB 레이어 (거의 그대로 이전)
│   └── requirements.txt            # 백엔드 전용 의존성
│
├── frontend/                       # React 앱
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── MainPage.tsx        # 기사 목록 + 검색
│   │   │   ├── DetailPage.tsx      # 기사 상세 + 신뢰도
│   │   │   └── AdminPage.tsx       # 어드민 대시보드
│   │   ├── components/
│   │   │   ├── ArticleCard.tsx
│   │   │   ├── TrustGauge.tsx      # 신뢰도 시각화
│   │   │   ├── SearchBar.tsx
│   │   │   └── Pagination.tsx
│   │   └── api/
│   │       └── client.ts           # axios 기반 API 클라이언트
│   └── public/
│
├── models/                         # ML 모델 저장 (TD1)
│   ├── trust_rf.pkl
│   └── train_trust_rf.py
│
├── data/
├── .gitignore
└── README.md
```

---

### 마이그레이션 단계별 계획

#### Step 1 — 백엔드 셋업 (현 로직 이전)

1. `backend/` 디렉토리 생성
2. `Streamlit_Rendering/` 모듈들을 `backend/services/`로 복사·정리
3. FastAPI `main.py` 작성 (CORS 허용: `localhost:5173`)
4. 라우터별 엔드포인트 작성
5. Pydantic 스키마 정의 (ArticleOut, SearchRequest, TrustResult 등)
6. `uvicorn backend.main:app --reload`로 단독 실행 확인

#### Step 2 — 프론트엔드 셋업

1. `frontend/` 디렉토리에 `npm create vite@latest . -- --template react-ts`
2. Tailwind CSS 또는 shadcn/ui 설치 (스타일링)
3. axios 설치, `api/client.ts` 작성
4. 페이지별 컴포넌트 작성 (MainPage → DetailPage → AdminPage 순)
5. React Router 적용 (`/`, `/article/:id`, `/admin`)
6. `npm run dev`로 개발 서버 확인

#### Step 3 — 기능 통합 및 검증

1. 검색 기능 E2E 테스트 (프론트 → FastAPI → DuckDB 벡터 검색)
2. 크롤링 어드민 연동 테스트
3. TD1 신뢰도 결과 TrustGauge 컴포넌트 시각화
4. 기존 Streamlit 앱과 기능 동등성 확인

#### Step 4 — 구 Streamlit 코드 정리

1. `app.py` 및 `Streamlit_Rendering/` 제거 또는 `legacy/` 이동
2. `.devcontainer/devcontainer.json` 업데이트
3. `README.md` 업데이트

---

### 주요 기술 선택

| 항목              | 선택                  | 이유                                          |
|-------------------|-----------------------|-----------------------------------------------|
| Frontend 프레임워크 | React + Vite + TypeScript | 생태계, 컴포넌트 재사용성, 타입 안전성     |
| 스타일링          | Tailwind CSS          | 빠른 UI 작성, 커스터마이징 유연성              |
| 상태관리          | React Query (TanStack) | 서버 상태 캐싱, 로딩/에러 상태 처리 자동화   |
| API 통신          | axios                 | 인터셉터, 에러 핸들링 편의성                  |
| Backend 프레임워크 | FastAPI               | 비동기 지원, 자동 OpenAPI 문서, Pydantic 통합 |
| DB 레이어         | DuckDB (유지)         | 현재 임베딩 스키마 유지, 마이그레이션 비용 최소화 |

---

### 마일스톤

- [ ] M1: FastAPI 서버 구동 및 `/api/articles` 응답 확인
- [ ] M2: React 프로젝트 셋업 및 MainPage 렌더링
- [ ] M3: 검색 기능 프론트↔백엔드 E2E 연동
- [ ] M4: 어드민 크롤링 기능 연동
- [ ] M5: TD1 신뢰도 결과 TrustGauge 시각화 적용
- [ ] M6: Streamlit 코드 legacy 이전 완료

---

## TD 요약

### TD1 — 한눈에 보기

| 항목           | 내용                                                                 |
|----------------|----------------------------------------------------------------------|
| 목표           | 더미 신뢰도 함수 → TELLER 구조 기반 실질적 분석 모듈로 교체          |
| 핵심 변경 파일 | `Streamlit_Rendering/trust.py` 전면 재작성                           |
| 외부 의존성 추가 | 없음 (기존 `numpy`, `google-generativeai` 재사용)                |
| 반환 포맷      | 현행 더미와 동일 (하위 호환) — `score`, `verdict`, `reason`, `per_criteria` |
| Decision System | Weighted Sum Rule (학습 데이터 불필요, 가중치 상수로 조정)        |

#### 기사 1건당 Gemini API 호출 횟수

```
기사 텍스트 입력
      │
      ▼
[Gemini 호출 1회] ← per_criteria 5개 점수·이유 + 종합 reason을 단일 JSON 프롬프트로 묶음
      │
      ▼
 JSON 파싱 → RF 분류기 입력 (로컬, 추가 호출 없음)
      │
      ▼
 최종 결과 반환
```

| 호출 단계                  | 횟수  | 비고                                              |
|----------------------------|-------|---------------------------------------------------|
| per_criteria 추출 + reason | 1회   | 5개 기준 점수·이유와 종합 reason을 하나의 프롬프트로 처리 |
| RF 분류                    | 0회   | 로컬 모델, API 미사용                              |
| **합계**                   | **1회/기사** |                                            |

> **비교**: 현재 `admin_pipeline.py`는 기사 1건당 요약(1회) + 임베딩(청크 수만큼, 평균 2~3회) 호출.
> TD1 추가 시 기사 1건당 **+1회** 증가.

---

### TD2 — 한눈에 보기

| 항목           | 내용                                                                 |
|----------------|----------------------------------------------------------------------|
| 목표           | Streamlit 단일 앱 → React(프론트) + FastAPI(백엔드) 분리             |
| 프론트 서버    | `npm run dev` → `localhost:5173`                                     |
| 백엔드 서버    | `uvicorn` → `localhost:8000`                                         |
| DB 레이어      | DuckDB 유지 (스키마 변경 없음)                                       |
| 비즈니스 로직  | `Streamlit_Rendering/` → `backend/services/`로 이전 (대부분 재사용) |
| 주요 신규 작업 | FastAPI 라우터·Pydantic 스키마 작성, React 컴포넌트 작성             |

---

## 작업 우선순위 및 의존 관계

```
TD1 ─────────────────────────────────────────────────────────────► 완성된 trust.py
                                                                          │
TD2-Step1 (FastAPI 백엔드) ──────────────────────────────────────────────┤
          │                                                               │
TD2-Step2 (React 프론트엔드) ─────────────────────────────────────► 통합 완성
```

> TD1과 TD2는 병렬 진행 가능.
> TD2-Step3 (신뢰도 시각화)는 TD1 완성 후 진행.

---

## 참고 자료

- TELLER 논문: *A Trustworthy Framework For Explainable, Generalizable and Controllable Fake News Detection* (2024)
- FastAPI 공식 문서: https://fastapi.tiangolo.com
- React Query: https://tanstack.com/query
- scikit-learn RandomForest: https://scikit-learn.org/stable/modules/ensemble.html#forest
