# trust_issue.md — trust.py 현재 이슈 정리

**작성일**: 2026-03-22
**기준 코드**: `backend/services/trust.py` (Phase 1+2 적용본)
**발견 경로**: `eval_trust.py` 첫 실행 결과 분석 (`docs/trust_dev1.md`)

---

## 이슈 분류 요약

| # | 이슈 | 심각도 | 해결 가능 여부 |
|---|------|--------|---------------|
| I-1 | 검증용 한국어 레이블 데이터 없음 | 🔴 구조적 | 단기 해결 어려움 |
| I-2 | AI-HUB 데이터셋 레이블 불일치 | 🔴 구조적 | 데이터셋 교체 필요 |
| I-3 | 교차 패널티 과도 감점 | 🟡 로직 | 코드 수정 가능 |
| I-4 | clickbait_risk 가중치 낮음 | 🟡 로직 | 데이터 없이 튜닝 어려움 |
| I-5 | source 없는 기사 변별력 없음 | 🟢 운영 | 크롤러 개선으로 완화 |
| I-6 | 가중치·임계값 실증 검증 불가 | 🔴 구조적 | 데이터 확보 전 보류 |

---

## I-1. 검증용 한국어 레이블 데이터 없음

### 문제
한국어 뉴스 신뢰도 레이블(fake/real/신뢰도 점수)이 붙은 공개 데이터셋이 사실상 존재하지 않는다.

- **SNU 팩트체크**: 서비스 종료, 공식 데이터셋 배포 없음
- **TELLER (ACL 2024) GitHub**: 데이터셋이 전부 영어 (LIAR, PolitiFact, GossipCop, Constraint)
- **AI-HUB**: 등록 필요 + 레이블 의미 불일치 (→ I-2)
- **Hugging Face**: 한국어 신뢰도 레이블 데이터셋 없음
- **KoPolitic (GitHub)**: 정치 성향 레이블, 신뢰도 아님

### 영향
`eval_trust.py`의 Spearman ρ, F1, 그룹 평균 등 모든 수치 검증 지표를 실행할 수 없다.
가중치(`WEIGHTS`), verdict 임계값(70/40), 교차 패널티 임계값(4)을 데이터로 튜닝하거나 검증하는 것이 불가능하다.

### 현실적 대응
- 데이터 기반 튜닝 보류
- 불변 조건 테스트(I-1-A)와 섭동 테스트(I-1-B)로 방향성 검증만 수행
- 운영 중 이상 케이스 누적 후 판단

```
I-1-A 불변 조건 테스트 (Invariant Test):
  score_trust(연합뉴스 팩트 기사) > score_trust(제목-본문 불일치 기사)
  score_trust(통계 인용 기사) > score_trust(주장만 있는 기사)
  → 절대 점수 아닌 순위만 검증

I-1-B 섭동 테스트 (Perturbation Test):
  원본 기사에서 전문가 인용 제거 → evidence_support 점수 감소 확인
  원본 기사의 제목을 자극적으로 교체 → clickbait_risk 점수 증가 확인
```

---

## I-2. AI-HUB 데이터셋 레이블 의미 불일치

### 문제
`clickbait_aggregated_train.csv`의 레이블:
- `Fake` = **제목이 낚시성인 기사** (본문이 허위라는 의미가 아님)
- `Real` = 정상 제목의 기사 (본문 신뢰도와 무관)

`trust.py`는 본문 내용 품질(증거, 논리, 중립성)을 평가하도록 설계됐으나,
데이터셋은 제목 선정성을 레이블로 사용한다.

### 실제 관측 결과 (2026-03-22 eval)
```
Real 그룹 평균:  46.1점  (목표 ≥ 65 → 미달)
Fake 그룹 평균:  59.6점  (목표 ≤ 45 → 반대 방향)
Spearman ρ:      음수 추정
```

Fake 기사가 Real보다 높은 점수를 받은 이유:
- Fake(낚시성 제목) 기사는 본문이 정상 경제·사회 기사 → evidence_support 고점
- Real 기사 중 스포츠 속보·단신 등 근거 없는 짧은 기사 → evidence_support 저점

### 영향
`eval_trust.py`로 측정한 모든 지표가 무의미하다.
데이터셋을 교체하기 전까지 Phase 3 검증은 실질적으로 불가능하다.

### 대응 방향
- AI-HUB 데이터로는 **`clickbait_risk` 기준 단독 F1 측정만** 유효
  (낚시성 제목 여부 판별에 한해 레이블이 의미 있음)
- 전체 신뢰도 검증은 아래 조합으로 소규모 수작업 데이터 구성 필요:

  | 유형 | 조건 | label |
  |------|------|-------|
  | 신뢰 기사 | 등록 언론사 + 통계·전문가 인용 있음 | 1 |
  | 낚시성 기사 | 제목-본문 불일치 + 근거 없음 | 0 |
  | 중립 기사 | 단신·속보 (근거는 없으나 조작 아님) | 0.5 |

---

## I-3. 교차 패널티 과도 감점

### 문제
```python
# _cross_penalty() 현재 구현
if neutrality < 4 and evidence < 4:
    return -10.0
```

두 기준이 모두 4 미만일 때 일괄 -10점 적용.
스포츠 속보·단신처럼 근거는 없지만 조작이 아닌 기사도 대상이 된다.

### 실제 관측
`eval` 결과 Real #6 score=0 — 정상 기사가 0점 수렴.
교차 패널티가 이미 낮은 raw score에 추가 감점을 주면서 0점으로 수렴한 것으로 추정.

### 수정 방향 (데이터 확보 후 검토)
```python
# 안 1: 임계값 강화 (더 극단적인 경우만 적용)
if neutrality < 3 and evidence < 3:
    return -7.0

# 안 2: 선형 패널티 (계단식 제거)
if neutrality < 4 and evidence < 4:
    severity = (4 - neutrality + 4 - evidence) / 8  # 0~1
    return -10.0 * severity
```

**현재는 보류** — 올바른 데이터셋 없이 수정하면 잘못된 방향으로 과적합될 수 있음.

---

## I-4. clickbait_risk 가중치 낮음

### 문제
```python
WEIGHTS = {
    "source_credibility":  0.25,
    "evidence_support":    0.25,
    "style_neutrality":    0.20,
    "logical_consistency": 0.20,
    "clickbait_risk":     -0.10,   # 전체의 10%
}
```

낚시성 기사에서 `clickbait_risk`가 높게(위험하게) 평가되더라도,
나머지 4개 기준이 본문 품질로 인해 높게 평가되면 전체 점수를 내리기에 역부족이다.

### 현황
- 이론적으로는 -0.15 ~ -0.20 범위가 더 적절할 수 있음
- 단, 가중치 변경은 실증 데이터 없이 하면 임의 조정이 됨
- `_MAX_RAW`는 동적 계산이므로 가중치 변경 시 자동 반영됨 (코드 수정 최소)

### 대응
AI-HUB 데이터로 `clickbait_risk` 단독 F1을 먼저 측정한 뒤,
성능 확인 후 가중치 조정 여부 결정.

---

## I-5. source 없는 기사 변별력 없음

### 문제
크롤러에서 언론사명(source)을 가져오지 못하거나 None으로 들어오면
`_rule_source_score()` → 전부 5점(중립) 반환.
`source_credibility` 가중치 0.25가 사실상 상수가 된다.

### 영향
- `eval` 전체 20건이 source=None → 변별력 없음 확인됨
- 실제 운영 환경에서는 크롤러가 출처를 수집하므로 일부 완화됨

### 확인 필요 사항
```python
# backend/services/crawl.py — source 필드가 실제로 채워지는지 확인
# repo.py — articles 테이블 source 컬럼 null 비율 확인
```

SOURCE_TIER에 등록된 언론사(연합뉴스·KBS·조선일보 등)를 실제 크롤링하고 있다면
운영 환경에서는 문제가 줄어든다.

---

## I-6. 가중치·임계값 실증 검증 불가 (구조적 한계)

### 문제
현재 시스템의 숫자들은 모두 이론/직관에 근거한 값이다:

| 파라미터 | 현재 값 | 근거 |
|----------|---------|------|
| WEIGHTS | 0.25/0.25/0.20/0.20/-0.10 | 직관적 설계 |
| verdict 임계값 | ≥70 / ≥40 | 직관적 설계 |
| 교차 패널티 임계값 | <4 | 직관적 설계 |
| 교차 패널티 크기 | -10 | 직관적 설계 |

이를 검증하려면 신뢰도 레이블이 있는 데이터가 필요하지만 현재 없다.

### 결론
trust.py의 신뢰도 점수는 **"LLM이 TELLER 기준으로 판단한 상대적 본문 품질 지표"** 이며,
실증적으로 검증된 팩트체크 점수가 아니다.

이 한계는 UI/서비스에서 명시적으로 안내하는 방향이 현실적으로 올바르다.

---

## 단기 실행 가능한 검증 계획

데이터 없이 지금 할 수 있는 검증:

```
1. clickbait_risk 단독 F1 측정
   - AI-HUB Fake/Real → clickbait_risk 점수 이진 분류 성능 확인
   - eval_trust.py의 clickbait_risk 리포트 섹션 활용

2. Invariant 쌍 테스트 (수작업 20쌍)
   - 쌍 설계 → score_trust() 두 번 호출 → 순위 방향 확인
   - 가중치 튜닝이 아닌 로직 방향성 검증 목적

3. 운영 환경 source 채워진 기사 대상 재실행
   - crawl.py로 등록 언론사(연합뉴스 등) 기사 수집 후 평가
   - SOURCE_TIER 효과 실측
```

---

## 관련 문서

- `docs/trust_revise.md` — Phase 1+2 리팩토링 지침
- `docs/trust_dev1.md` — 첫 eval 결과 상세 분석
- `backend/services/trust.py` — 현재 구현
- `backend/services/eval_trust.py` — 검증 스크립트
- `data/trust_eval_samples.jsonl` — 현재 평가 데이터 (AI-HUB 샘플 20건)
