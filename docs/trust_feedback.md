# trust_feedback.md — 신뢰도 모듈 피드백 루프 설계

**작성일**: 2026-04-05  
**대상 코드**: `backend/services/trust.py` (W3 기준)  
**선행 문서**: `docs/trust_limitations.md`, `docs/trust_pipeline.md`

---

## 0. 왜 피드백 루프가 필요한가

현재 trust.py의 구조적 약점은 다음 두 가지로 요약된다.

| 약점 | 원인 |
|------|------|
| 가중치(W3)가 수동 튜닝 | 9쌍 invariant_pairs만으로 검증 — 통계적으로 불충분 |
| LLM 응답의 적절성 검증 불가 | Gemini가 "맞는 방향으로 답하는지" 외부 기준이 없음 |

피드백 루프는 이 두 약점을 **데이터 기반**으로 보완하는 유일한 방법이다.  
목표는 "모델을 자동으로 학습시키는 것"이 아니라, **어떤 기준이 틀렸는지를 식별해서 개발자가 올바르게 조정할 수 있게 하는 것**이다.

---

## 1. 수집할 피드백 신호

### 1-1. 명시적 피드백 (Explicit)

**Tier 1 — verdict 동의 여부** ← 가장 구현하기 쉽고, 가장 중요

```
사용자가 상세 페이지에서 TrustGauge 아래에:
  👍 "이 분석이 도움이 됐나요?"   [예 / 아니오]
```

- 마찰 최소화: 클릭 1회
- 익명 처리 (session_id or 로컬스토리지 UUID)
- 저장: `verdict_agree: bool`

**Tier 2 — 기준별 이의 제기** ← 선택적 확장

```
"아니오" 클릭 시 추가 질문:
  어떤 항목이 이상한가요? (복수 선택 가능)
  □ 출처 신뢰성   □ 근거 지지도   □ 문체 중립성
  □ 논리 일관성   □ 어뷰징 위험도
```

- 마찰 있음 → 선택적으로만 노출
- 저장: `criteria_flags: list[str]`

### 1-2. 묵시적 피드백 (Implicit)

현재 구현된 기능에서 수집 가능한 행동 신호:

| 신호 | 의미 | 구현 위치 |
|------|------|-----------|
| 상세 페이지 체류 시간 | 기사에 관심이 있음 | 프론트 이벤트 |
| 원문 링크 클릭 | 신뢰도에 의문이 있거나 추가 확인 필요 | `client.ts` 이벤트 |
| 관련 기사 클릭률 | 분석 결과가 유용한지 | 기존 라우터 로그 |

> 묵시적 신호는 해석이 모호하므로 **보조 지표**로만 활용.  
> 핵심은 Tier 1 명시적 피드백이다.

### 1-3. 관리자 직접 레이블링 (Gold Standard)

Admin 페이지에서 특정 기사에 `gold_label` 을 부여하는 기능:

```
기사별: [실제 신뢰 가능] / [실제 가짜뉴스] / [판단 유보]
```

- 양이 적어도 **가중치 실험의 기준선**으로 사용 가능
- 향후 SNU 팩트체크 데이터 연동 시 자동 부여 가능 (`docs/trust_limitations.md §L1` 참조)

---

## 2. DB 스키마 확장

### 2-1. article_feedback 테이블 (신규)

```sql
CREATE TABLE IF NOT EXISTS article_feedback (
    feedback_id   VARCHAR PRIMARY KEY,  -- UUID
    article_id    VARCHAR NOT NULL,
    session_id    VARCHAR,              -- 익명 식별자 (로컬스토리지 UUID)
    verdict_agree BOOLEAN,             -- True=동의, False=이의
    criteria_flags VARCHAR,            -- JSON array: ["evidence_support", ...]
    created_at    TIMESTAMP DEFAULT current_timestamp
);
```

### 2-2. articles 테이블 컬럼 추가

```sql
ALTER TABLE articles ADD COLUMN gold_label VARCHAR;
-- "true" | "false" | "uncertain" | NULL (미레이블)
```

### 2-3. 집계 뷰 (분석용)

```sql
CREATE VIEW feedback_summary AS
SELECT
    a.article_id,
    a.trust_verdict,
    a.trust_score,
    COUNT(f.feedback_id)                          AS total_feedback,
    SUM(CASE WHEN f.verdict_agree THEN 1 ELSE 0 END) AS agree_count,
    ROUND(AVG(CASE WHEN f.verdict_agree THEN 1.0 ELSE 0.0 END) * 100, 1) AS agree_rate
FROM articles a
LEFT JOIN article_feedback f USING (article_id)
GROUP BY a.article_id, a.trust_verdict, a.trust_score;
```

---

## 3. API 설계 (신규 엔드포인트)

```
POST /api/feedback
Body: { article_id, session_id, verdict_agree, criteria_flags? }
Response: { status: "ok" }

GET /api/admin/feedback-stats
Response: {
    total_feedback: int,
    agree_rate_by_verdict: {
        likely_true:  float,   # 예: 0.82 (82% 동의)
        uncertain:    float,
        likely_false: float,
    },
    criteria_flag_counts: {
        evidence_support:    int,
        style_neutrality:    int,
        logical_consistency: int,
        clickbait_risk:      int,
        source_credibility:  int,
    },
    low_agreement_articles: [  # 이의 비율 높은 기사 목록
        { article_id, title, trust_verdict, agree_rate },
        ...
    ]
}
```

---

## 4. Admin 페이지 모니터링 대시보드

Admin 페이지에 **"신뢰도 모델 현황"** 섹션을 추가한다.

### 4-1. 점수 분포 히스토그램 (K1 KPI)

```
0~10  ■■
10~20 ■■■
20~30 ■
30~40 ■■■■
40~50 ■■■■■■■■
50~60 ■■■■■■■
60~70 ■■■■■
70~80 ■■■
80~90 ■■
90~100 ■
```

- 정상 분포: 표준편차 ≥ 15, 세 verdict 구간 모두에 기사 존재
- 이상 패턴: 40~70 구간에 80% 이상 밀집 → uncertain 과밀 문제

### 4-2. verdict별 사용자 동의율

```
likely_true  (N건): ████████████░░  82% 동의
uncertain    (N건): ██████░░░░░░░░  45% 동의  ← 주의
likely_false (N건): █████████░░░░░  61% 동의
```

- `uncertain` 동의율이 낮으면: 분별력 없는 "모름" 판정 남발 신호
- `likely_false` 동의율이 50% 미만: 잘못된 가짜뉴스 판정 남발 신호 (심각)

### 4-3. 기준별 이의 빈도 차트

```
evidence_support    ████████ 38%  ← 가장 많은 이의
style_neutrality    █████    22%
logical_consistency ████     18%
clickbait_risk      ███      13%
source_credibility  ██        9%
```

→ 이의가 가장 많은 기준 = 프롬프트 또는 가중치 조정 우선 대상

### 4-4. 이의 많은 기사 목록 (플래그 목록)

```
| 기사 제목                          | verdict       | 동의율 | 플래그 |
|------------------------------------|---------------|--------|--------|
| "[충격] ..."                       | likely_false  |  22%   | 보기   |
| "정부, 경제 회복세..."             | likely_true   |  31%   | 보기   |
```

→ 관리자가 해당 기사를 직접 확인하고 `gold_label` 부여 가능

---

## 5. 가중치 업데이트 파이프라인

현재 가중치(W3)는 수동 실험의 결과다. 피드백이 쌓이면 다음 3단계 업데이트 방식을 적용할 수 있다.

### 5-1. 단기 — 피드백 기반 경험적 조정

**조건**: 피드백 50건 이상 수집 후

```
알고리즘:
1. verdict별 동의율 계산
2. 이의가 가장 많은 기준 식별
3. 해당 기준 가중치 ±0.05 조정 후 eval_trust.py 재실행
4. invariant_pairs 통과율이 유지되면 새 가중치 채택
```

예시:
```
evidence_support 이의 38% → W_evidence 0.25 → 0.20으로 하향 시험
style_neutrality 이의 22% → W_style 0.25 유지 또는 소폭 상향 시험
```

`eval_trust.py`의 `WEIGHT_VARIANTS`에 후보를 추가해 배치 실행하는 방식으로 자동화 가능.

### 5-2. 중기 — gold_label 기반 그리드 서치

**조건**: gold_label이 붙은 기사 30건 이상

```python
# 의사코드
from itertools import product

weight_grid = {
    "source_credibility":  [0.15, 0.20, 0.25],
    "evidence_support":    [0.20, 0.25, 0.30],
    "style_neutrality":    [0.20, 0.25, 0.30],
    "logical_consistency": [0.15, 0.20, 0.25],
    "clickbait_risk":      [-0.05, -0.10, -0.15],
}

best_accuracy = 0
best_weights = None

for w_combo in product(*weight_grid.values()):
    weights = dict(zip(weight_grid.keys(), w_combo))
    accuracy = eval_on_gold_labels(weights, gold_labeled_articles)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = weights
```

- 평가 지표: gold_label과 trust.py verdict 일치율
- 부가 제약: invariant_pairs 통과율 ≥ 88%를 유지해야 채택

### 5-3. 장기 — 선형 회귀 기반 가중치 학습

**조건**: gold_label 100건 이상 (현실적으로 6개월~1년 뒤)

```
입력: per_criteria 5개 점수 (특성)
출력: gold_label (이진: 신뢰/불신)
모델: 로지스틱 회귀

회귀 계수 → 새 가중치로 사용
제약: 계수 부호 유지 (clickbait_risk는 반드시 음수)
```

이 단계에서는 trust.py가 "수동 설계된 규칙 시스템"에서  
"데이터로 학습된 얕은 모델"로 진화한다.

---

## 6. 프롬프트 개선 루프

### 6-1. LLM 일관성 모니터링 (K4 KPI)

Admin에서 주기적으로 "일관성 검사" 실행:
- DB에서 10개 기사 무작위 샘플링
- 동일 기사를 trust.py로 2회 분석
- 점수 차이 > 10점이면 불안정 판정

```
일관성 검사 결과 예시:
기사 A: 1회=72점, 2회=69점 → ✅ 안정 (±3)
기사 B: 1회=55점, 2회=38점 → ❌ 불안정 (±17) — 점검 필요
```

→ 불안정한 기준이 확인되면 해당 predicate 프롬프트를 더 명시적으로 재작성

### 6-2. per_criteria 피드백 → 프롬프트 A/B 테스트

```
이의 많은 기준 식별 (예: logical_consistency 이의 30%↑)
  ↓
프롬프트 B 작성 (기존 대비 더 명시적인 판단 기준 추가)
  ↓
동일한 gold_label 기사 집합에 프롬프트 A/B 각각 적용
  ↓
gold_label 일치율 비교 → 높은 쪽 채택
```

현재 `logical_consistency` 개선 예시(B-1)가 이미 이 방식으로 진행됨.  
이 과정을 **체계화**해서 반복 가능하게 만드는 것이 목표.

---

## 7. 레퍼런스 데이터 확장

현재 검증 데이터의 한계: 9쌍 invariant_pairs (수작업)

### 7-1. 단기 — 기존 데이터 활용

| 소스 | 방법 | 예상 건수 |
|------|------|----------|
| DB 누적 기사 + 사용자 피드백 | 동의율 < 30% → likely_false pseudo-label | 크롤링 규모에 비례 |
| 관리자 직접 레이블 | Admin 페이지에서 gold_label 수동 부여 | 주 5~10건 |

### 7-2. 중기 — 외부 팩트체크 연동

| 소스 | 특징 | 연동 방식 |
|------|------|-----------|
| SNU 팩트체크 (snufactcheck.or.kr) | 전문가 검증, 신뢰도 높음 | 기사 URL 매칭 → gold_label 자동 부여 |
| 한국언론진흥재단 뉴스트러스트 | 언론사 신뢰도 평가 데이터 | SOURCE_TIER 사전 업데이트 |

### 7-3. 장기 — 능동 학습 (Active Learning)

```
1. 현재 모델로 전체 기사 분석
2. 모델이 "가장 불확실해하는" 기사 (uncertain, 점수 45~55) 선별
3. 해당 기사만 집중 레이블링 → 학습 효율 극대화
```

모델 불확실성이 높은 기사 = 레이블 1건당 학습 효과가 가장 큰 기사.

---

## 8. 구현 로드맵

### Phase 1 — 피드백 수집 (즉시 시작 가능)

- [ ] `article_feedback` 테이블 생성 (`repo.py` 확장)
- [ ] `POST /api/feedback` 엔드포인트 추가
- [ ] TrustGauge 하단에 thumbs up/down 버튼 추가
- [ ] 피드백 데이터 Admin 페이지에 총 건수 표시

**기대 효과**: 실사용자 반응 데이터 수집 시작, 이후 모든 개선의 근거가 됨

---

### Phase 2 — 분석 대시보드 (Phase 1에서 피드백 20건 이상 수집 후)

- [ ] `GET /api/admin/feedback-stats` 엔드포인트 구현
- [ ] Admin 페이지에 점수 분포 + 동의율 차트 추가
- [ ] 기준별 이의 빈도 시각화
- [ ] "이의 많은 기사" 플래그 목록 표시
- [ ] 관리자 gold_label 부여 UI

**기대 효과**: 어떤 기준이 문제인지 데이터로 가시화

---

### Phase 3 — 배치 가중치 실험 (gold_label 30건 이상 수집 후)

- [ ] gold_label 기반 정확도 평가 함수 `eval_trust.py`에 추가
- [ ] W_candidate 그리드 서치 배치 스크립트 작성
- [ ] invariant_pairs 통과율 + gold_label 정확도 복합 채택 기준 정의
- [ ] 새 가중치 채택 시 DB 전체 기사 재채점 배치 실행

**기대 효과**: 수동 튜닝 → 데이터 기반 튜닝으로 전환

---

### Phase 4 — 프롬프트 A/B 테스트 자동화 (Phase 3 완료 후)

- [ ] predicate 프롬프트 버전 관리 (`PREDICATES_V1`, `PREDICATES_V2`)
- [ ] 동일 기사 셋으로 두 프롬프트 결과 비교하는 스크립트
- [ ] 통과율 유의미 향상 시 자동 채택 → trust.py 업데이트

---

## 9. 설계 원칙 — 피드백 루프 운영 시 주의사항

| 원칙 | 이유 |
|------|------|
| **피드백은 신호일 뿐, 정답이 아니다** | 사용자도 가짜뉴스를 진짜로 믿을 수 있음. 피드백 편향에 주의 |
| **invariant_pairs 통과율 ≥ 88%를 어떤 업데이트에서도 유지** | 기본 일관성을 보장하는 최소 기준선 |
| **가중치 변경 시 항상 전체 재채점** | 이전 기사의 verdict가 새 기준으로 달라질 수 있음 |
| **모델은 여전히 패턴 경고 시스템** | gold_label 정확도를 높이더라도 "사실판별" 주장은 금지 |
| **피드백 데이터도 git에서 버전 관리** | `invariant_pairs.jsonl`처럼 `gold_labels.jsonl`을 별도 관리 |
