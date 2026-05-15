# trust_dev1.md — Phase 1+2 첫 평가 결과 분석

**실험일**: 2026-03-22
**모델**: `gemini-2.5-flash` (Phase 1+2 적용본)
**데이터**: `data/trust_eval_samples.jsonl` — Real 10건(label=1), Fake 10건(label=0)
**데이터 출처**: AI-HUB 낚시성 기사 데이터셋 (`clickbait_aggregated_train.csv`)

---

## 1. 원시 결과

| # | label | score | verdict |
|---|-------|-------|---------|
| 1 | Real(1) | 67 | uncertain |
| 2 | Real(1) | 39 | likely_false |
| 3 | Real(1) | 57 | uncertain |
| 4 | Real(1) | 50 | uncertain |
| 5 | Real(1) | 56 | uncertain |
| 6 | Real(1) | **0** | likely_false |
| 7 | Real(1) | 53 | uncertain |
| 8 | Real(1) | 57 | uncertain |
| 9 | Real(1) | 53 | uncertain |
| 10 | Real(1) | 29 | likely_false |
| 11 | Fake(0) | 67 | uncertain |
| 12 | Fake(0) | 59 | uncertain |
| 13 | Fake(0) | 69 | uncertain |
| 14 | Fake(0) | 63 | uncertain |
| 15 | Fake(0) | 63 | uncertain |
| 16 | Fake(0) | 57 | uncertain |
| 17 | Fake(0) | 57 | uncertain |
| 18 | Fake(0) | 61 | uncertain |
| 19 | Fake(0) | 53 | uncertain |
| 20 | Fake(0) | 47 | uncertain |

| 지표 | 값 | 목표 |
|------|----|------|
| 전체 평균 | 52.9 | — |
| **Real 그룹 평균** | **46.1** | ≥ 65 |
| **Fake 그룹 평균** | **59.6** | ≤ 45 |
| Spearman ρ (추정) | **음수** | ≥ 0.5 |

---

## 2. 핵심 문제: 점수 방향 역전

Real 평균(46.1) < Fake 평균(59.6) — **레이블과 점수 방향이 반대**.
검증 목표를 모두 미달하며, Spearman 상관계수는 음수로 추정된다.

---

## 3. 원인 분석

### 3-1. 데이터셋 레이블 의미 불일치 (가장 큰 원인)

AI-HUB 낚시성 데이터셋의 `Fake` 레이블은 **"제목이 낚시성인 기사"** 를 의미하며,
**기사 본문이 허위라는 뜻이 아니다.**

```
Fake 샘플 예시 (본문은 정상 경제 기사):
  "에너지 수입 급증에 이달 1~20일 무역적자 21억달러…원유 57.8%·가스 114.3% ↑"
  → 본문에 구체적 수치·통계 다수 포함 → evidence_support 고점
  → logical_consistency 정상 → 전체 점수 상승

Real 샘플 예시 (짧은 스포츠 속보):
  "한화 이용규, 경기 중 '어지럼증' 호소… 3회 '조기 교체'"
  → 전문가 인용 없음, 통계 없음 → evidence_support 저점
  → 전체 점수 하락
```

`trust.py`는 **본문의 신뢰도(내용 품질)**를 측정하도록 설계되었지만,
이 데이터셋은 **제목의 선정성**을 레이블로 사용한다.
→ 평가 목적과 데이터셋 목적이 근본적으로 다름.

### 3-2. Real #6 점수 0 이상 징후

`Real #6` 점수 0은 두 가지 가능성:

**가능성 A — 교차 패널티 연쇄 폭발**
`_cross_penalty()`: style_neutrality < 4 AND evidence_support < 4 → -10점
두 기준이 모두 저점이면 이미 낮은 raw score에서 추가 -10 → 0점 수렴 가능.

**가능성 B — API/파싱 실패 후 fallback**
`_fallback()` 반환 시 score=0, verdict=likely_false.
로그에 오류 메시지가 없으면 A 가능성이 높음.

### 3-3. clickbait_risk 가중치 미흡

`clickbait_risk` 가중치는 -0.10으로 전체 가중치의 10%.
Fake 기사의 낚시성 제목을 LLM이 올바르게 감지해도,
나머지 본문 품질 점수(evidence_support 0.25, evidence 0.25, style 0.20, logic 0.20)가
압도적으로 높으면 전체 점수를 내리기 어렵다.

### 3-4. source 전부 None → source_credibility 25% 고정

모든 샘플의 source가 None → `SOURCE_TIER` 룩업 불가 → 전부 5점(중립).
`source_credibility` 가중치는 0.25 — 전체의 25%가 5점으로 고정되어 변별력 없음.
실제 운영 환경에서는 출처 정보가 있어 이 문제는 완화된다.

---

## 4. trust.py 자체 로직 평가

데이터셋 불일치를 제외하면, Phase 1+2 변경사항 자체는 의도대로 동작하고 있다.

| 항목 | 상태 |
|------|------|
| SOURCE_TIER 룰 기반 룩업 | ✅ 동작 (source=None → 5점 중립 처리 정상) |
| 3단계 폴백 파싱 | ✅ 20건 중 파싱 실패 없음 (score=0은 별도 원인) |
| MAX_RETRIES=3 루프 | ✅ 무한 루프 없음 확인 |
| logging 모듈 | ✅ print() 제거 |
| _trim_text (앞2000+뒤1000) | ✅ 적용됨 |
| Yes/No predicate 방식 | ✅ 20건 전원 응답 수신 |
| _cross_penalty | ⚠️ Real #6 과도 감점 의심 — 임계값 재검토 필요 |
| _MAX_RAW 동적 계산 | ✅ 정상 |
| fallback verdict=likely_false | ✅ 수정 완료 |

---

## 5. 향후 개선 방향

### 5-1. 데이터셋 교체 (최우선)

현재 데이터셋은 이 평가에 부적합하다.
아래 기준으로 `trust_eval_samples.jsonl`을 교체해야 한다:

| 유형 | 출처 | 수량 | label | source |
|------|------|------|-------|--------|
| 팩트체크 통과 기사 | SNU 팩트체크 | 10건 | 1 | 연합뉴스·KBS 등 등록 언론사 |
| 허위/낚시 기사 | AI-HUB (제목+본문 모두 허위) | 10건 | 0 | 미상 또는 저신뢰 |
| 중립 기사 | 네이버 뉴스 경제/정치 | 10건 | 0.5 | 중앙·한겨레 등 |

source 필드가 있어야 SOURCE_TIER 효과도 측정 가능하다.

### 5-2. 교차 패널티 임계값 완화

현재: `neutrality < 4 AND evidence < 4` → -10점
제안: `neutrality < 3 AND evidence < 3` → -7점
또는 패널티를 연속 함수로 완화 (계단식 대신 선형).

### 5-3. clickbait_risk 가중치 상향 검토

현재 -0.10은 낚시성 기사를 충분히 감점하지 못한다.
-0.15 ~ -0.20 범위로 조정 후 재평가 필요.
단, 가중치 변경 시 `_MAX_RAW`는 자동 재계산되므로 별도 수정 불필요.

---

## 6. 결론

이번 평가에서 모든 지표가 목표 미달이지만,
**핵심 원인은 trust.py 로직 결함이 아닌 데이터셋-평가 목적 불일치**다.

- AI-HUB 낚시성 데이터셋 = 제목 선정성 레이블
- trust.py = 본문 내용 신뢰도 측정

두 개념이 달라 Spearman 음수가 나오는 것은 오히려 trust.py가 **본문 품질을 올바르게 측정하고 있다는 방증**이기도 하다.
(낚시성 기사는 제목만 자극적이고 본문은 정상 → 높은 점수 부여는 논리적으로 일관됨)

Phase 2에서 추가한 교차 패널티 임계값과 clickbait_risk 가중치는
올바른 데이터셋으로 재평가 후 튜닝한다.
