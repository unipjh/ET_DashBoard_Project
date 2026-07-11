# 추천 시스템 진단 및 고도화 로드맵

> 작성일: 2026-07-05
> 배경: "장단기 추천 성능을 어떻게 확보할 것인가"라는 4가지 질문에 대한 진단과 그에 따른 개선 구현 기록

---

## 0. 네 가지 질문에 대한 결론 요약

| 질문 | 결론 |
|------|------|
| 좋은 추천이란? (장기 프로필 vs 단기 세션) | 뉴스 도메인은 **단기(세션 의도) 우선, 장기(계정 취향) 보조**. 단기 65% : 장기 35% 블렌딩으로 구현 |
| 단기 추천이 뜬금없지 않으려면? | recency 가중 역전 버그 수정 + 관련성 하한(0.35) + 카테고리 다양성 캡 + 탐색 슬롯 명시(rec_source='explore') |
| 테스트는 어떻게 정량화? | 오프라인: 로그 리플레이 평가(HitRate@K/MRR/NDCG). 온라인: rec_source별 CTR (impression 로깅을 이번에 추가해 분모 확보) |
| 목적 함수(튜닝 방향)는? | **주 지표 = 추천 섹션 CTR**, 오프라인 대리 지표 = HitRate@5/MRR, 가드레일 = 카테고리 다양성 + trust_score(신뢰 기반 강등) |

---

## 1. 진단: 기존 시스템의 문제점

### 1-1. 장/단기 구분 부재
`get_recent_user_history`가 `session_id OR user_id`를 한 쿼리로 섞어 단일 히스토리를 만들고,
단일 평균 벡터 하나로 검색했다. 로그인 사용자의 몇 주 전 관심사와 "지금 이 세션"의 의도가
같은 무게로 섞여, 세션 의도가 희석되는 구조였다.

### 1-2. recency 가중 역전 버그 (치명적)
`get_recent_user_history`는 **최신순(DESC)** 으로 반환하는데, `build_profile_vector`는
`weight = 1.0 + index/len` 로 **리스트 뒤쪽(=가장 오래된 기사)에 더 큰 가중치**를 주고 있었다.
문서("최근 기사일수록 더 높은 가중치")와 정반대로 동작 — 단기 추천이 뜬금없게 느껴지는
직접적 원인 후보였다.

### 1-3. 관련성 하한 부재
Stage 2 벡터 검색이 `min_score=0.0`으로 호출되어, 히스토리가 빈약하거나 프로필이 흐릿해도
"가장 덜 무관한" 기사를 무조건 반환했다. 유사도가 낮은 결과는 사용자에게 무작위로 보인다.

### 1-4. 추천 품질을 잴 수단이 없었음
- 추천 섹션 클릭은 `source: personalized_recommendation`으로 로깅됐지만 **노출(impression)은 로깅되지 않아 CTR의 분모가 없었다.**
- 오프라인 지표는 학습 시 val_auc뿐 (합성 데이터 기준 0.53).
- 어떤 폴백 단계(encoder/profile/category/latest)가 실제로 응답했는지 응답에 남지 않아 원인 분석 불가.

### 1-5. 목적 함수 미정의
학습은 암묵적으로 클릭 예측(cross-entropy)이었지만, 서빙/튜닝 단계에서 무엇을 개선해야
"좋아졌다"고 판단할지 정의가 없었다. 신뢰도 플랫폼이라는 정체성(trust_score)도 추천에 반영되지 않았다.

---

## 2. 이번에 구현한 것

### 2-1. 서빙 로직 (`backend/services/recommend.py`)
1. **recency 버그 수정** — 히스토리를 시간순으로 정규화한 뒤 지수 감쇠 가중치 적용
   (`0.5^(age/5)`: 최근 5개 이전 기사는 가중치 절반).
2. **장/단기 이중 프로필** — 세션 프로필 65% + 계정 프로필 35% 블렌딩 (`SHORT_TERM_WEIGHT`).
   게스트는 세션만, 한쪽이 비면 다른 쪽으로 자동 폴백.
3. **관련성 하한** — `RELEVANCE_FLOOR=0.35` 미달 시 벡터 결과를 버리고 카테고리 폴백.
4. **카테고리 다양성 캡** — 한 카테고리가 limit의 60%를 초과하지 않게 재배열 (도배 방지).
5. **신뢰 가드레일** — trust_score 1~39(likely_false 구간) 기사는 목록 후순위로 강등.
   ET의 정체성("신뢰할 수 있는 뉴스")을 추천 목적 함수에 반영.
6. **탐색 슬롯** — limit≥5일 때 마지막 1개를 히스토리 밖 카테고리의 최신 기사로 교체
   (`rec_source='explore'`). 필터버블 완화 + 학습 데이터 다양성 확보.
7. **rec_source/rec_score 태깅** — 모든 응답 기사에 어느 단계가 추천했는지
   (`encoder|profile|category|latest|explore`)와 유사도 점수를 포함.
8. 어텐션 인코더에는 시간순 시퀀스를 전달하도록 수정.

### 2-2. 측정 인프라
- **프론트** ([MainPage.tsx](../frontend/src/pages/MainPage.tsx)): 추천 섹션 impression 로깅 추가
  (`source: personalized_recommendation`, rec_sources 포함) → CTR 분모 확보.
  클릭 이벤트에 `rec_source` 포함. explore 추천에는 "새로운 발견" 배지 표시.
- **온라인 지표**: `GET /api/admin/recommendation-metrics?days=14` —
  노출/클릭/CTR + rec_source별 클릭 분해 (`repo.get_recommendation_funnel`).
- **오프라인 지표**: `python -m backend.training.evaluate_offline` —
  실제 event_logs를 리플레이하며 profile/category/latest/random 전략의
  HitRate@K, MRR, NDCG@K를 비교. **torch 불필요** (서빙 환경에서도 실행 가능).
- 학습·평가가 공유하는 샘플 빌더를 `backend/training/samples.py`로 분리 (torch-free).

---

## 3. 목적 함수 정의 (튜닝의 방향성)

```
최대화: 추천 섹션 CTR  (온라인, rec_source별로 분해 관측)
대리:   HitRate@5, MRR (오프라인 리플레이 — 배포 전 회귀 검증용)
제약:   ① 한 카테고리 ≤ 60% (다양성)
        ② trust_score < 40 기사는 후순위 (신뢰 가드레일)
        ③ 탐색 슬롯 1개 유지 (장기 데이터 품질을 위한 탐색-활용 균형)
```

**튜닝 절차** (파라미터: `SHORT_TERM_WEIGHT`, `RECENCY_HALF_LIFE`, `RELEVANCE_FLOOR`):
1. 파라미터 변경 → `evaluate_offline`으로 HitRate@5/MRR이 나빠지지 않는지 확인
2. 배포 → 1~2주 후 `recommendation-metrics`에서 CTR 비교
3. explore 슬롯의 CTR이 profile과 비슷해지면 RELEVANCE_FLOOR·프로필이 과협소하다는 신호

---

## 4. 남은 로드맵

| 단계 | 내용 | 조건 |
|------|------|------|
| 단기 | 실제 로그 리플레이로 profile vs category 정량 비교 (`evaluate_offline`) | 클릭 로그 수십 건이면 시작 가능 |
| 단기 | AdminPage에 recommendation-metrics 시각화 | 백엔드 완료, UI만 필요 |
| 중기 | 실제 데이터로 어텐션 인코더 재학습 (`train_user_encoder`, --synthetic 제거) | 클릭 ≥100, impression ≥1,000 |
| 중기 | 세션 내 "즉시 반응" — 상세 페이지 이탈 시 추천 무효화(refetch)는 이미 동작, 클릭 직후 프로필 반영 지연 측정 | — |
| 장기 | A/B 테스트: 세션을 해시로 2그룹 분할, 파라미터 변형 비교 | 일 방문자 수백 이상 |
| 장기 | Render 유료 전환 또는 ONNX 변환으로 Stage 1(인코더) 프로덕션 활성화 | 인프라 결정 |

---

## 5. 관련 문서
- [personalized_recommendation_2026-06-25.md](personalized_recommendation_2026-06-25.md) — 최초 설계
- [recommendation_flowchart_2026-07-03.md](recommendation_flowchart_2026-07-03.md) — 배포 상태 플로우차트
  (주의: Stage 2의 `min_score`는 이번 변경으로 0.35가 되었고, 장/단기 블렌딩·다양성 캡·탐색 슬롯이 추가됨)
