# ET 핵심 모듈 성능 최적화 일지

## 성능 목표와 채택 기준

### 신뢰도

- 방향성 불변쌍 통과율 ≥ 80%, 목표 9/9.
- 서울대 팩트체크 5단계 레이블 Spearman ρ와 극단 레이블 ROC-AUC를 기존 W3보다 개선.
- 근거 점수 4 미만 기사가 `likely_true`가 되지 않을 것.
- API 실패를 0점으로 처리하지 않을 것.

### 추천

- 시간 순서를 보존한 실제 로그에서 hybrid가 profile 대비 MRR·NDCG@3를 개선.
- 인간 직관 페르소나 8종, seed 5개 모두에서 semantic-only 대비 개선.
- Top-3 저신뢰 노출률을 증가시키지 않을 것.
- 추천 API·회귀 테스트를 깨지 않을 것.

## 실험 기록

### R-001 — 기존 실제 로그 기준선 감사

- 가설: 기존 리플레이 수치가 실제 개인화 품질을 나타낸다.
- 결과: profile MRR 0.9347, NDCG@5 0.9356.
- 발견: 30초 내 동일 기사 click/view 중복 967건, 클릭 이후 impression 혼입, 현재 DB에 없는 후보의 `-1` 처리, 동률 시 positive-first 유지.
- 결정: **기각**. 성능 수치가 누수로 과대평가되어 기준선으로 사용 불가.

### R-002 — 시간 순서 보존 샘플러

- 변경: 클릭 이전의 실제 impression만 negative로 사용, 30초 내 동일 기사 positive 중복 제거, 현재 유효 임베딩 기사만 포함, catalog 무작위 fallback 제거.
- 평가 설정: 실제 로그 118건, positive 1 + observed negative 3, K=3.
- 정정 기준선:
  - profile: HitRate@3 0.8729, MRR 0.6942, NDCG@3 0.7160.
  - category: HitRate@3 0.8729, MRR 0.7239, NDCG@3 0.7383.
  - latest: HitRate@3 0.7373, MRR 0.6631, NDCG@3 0.6329.
  - random: HitRate@3 0.7797, MRR 0.5593, NDCG@3 0.5741.
- 결정: **채택**. 이후 모든 실제 로그 평가는 temporal 모드만 사용.

### R-003 — 프로필 최근성 파라미터 탐색

- 탐색: history 1/2/3/5/10/20, half-life 0.5/1/2/3/5/8/12/∞.
- 결과: 현재 history 20, half-life 5가 시간 분할 tune 구간에서 최상이며 validation에서도 대체 조합이 개선하지 못함.
- 결정: **현재값 유지**. 파라미터 변경 기각.

### R-004 — 인간 직관 페르소나 재정렬

- 데이터: 8개 페르소나 × 20명 × 사용자당 11개 평가 시점 = seed당 1,760건.
- 분리: 사용자 단위 train 1,232 / validation 528.
- 반복: seed 42~46.
- 채택 조합: semantic 0.10 / category 0.60 / trust 0.10 / freshness 0.20.
- 5-seed 검증 평균:
  - NDCG@3: semantic 0.5231 → hybrid 0.6310 (**+20.6%**).
  - MRR: semantic 0.5420 → hybrid 0.6193 (**+14.3%**).
  - 저신뢰 Top-3 비율: 1.439% → 0.922% (**36.0% 감소**).
- 실제 temporal 로그:
  - profile → hybrid MRR: 0.6942 → 0.7458 (**+7.4%**).
  - profile → hybrid NDCG@3: 0.7160 → 0.7584 (**+5.9%**).
  - profile → hybrid HitRate@3: 0.8729 → 0.8814 (**+0.85%p**).
- 결정: **채택**. 임베딩 검색 후보에만 경량 재정렬 적용.

### R-005 — 임베딩 공간 분리

- 문제: 프로필 벡터는 이력의 `learned_embedding`을 우선 사용하지만, 후보 검색은 raw Gemini `article_chunks.embedding` 공간에서 수행하고 있었다.
- 공간 정합성(동일 기사 21건): learned/raw 코사인 평균 0.173, 중앙값 0.166.
- 자기 기사 검색:
  - learned → raw chunks: Top-1 0.2381, Top-5 0.6190, MRR 0.4304.
  - raw → raw chunks: Top-1 1.0000, Top-5 1.0000, MRR 1.0000.
- 실제 temporal 로그 118건 프로필 리플레이:
  - HitRate@3: mixed 0.8390 → raw-only 0.8729 (**+3.39%p**).
  - MRR: mixed 0.5530 → raw-only 0.6942 (**+25.5%**).
  - NDCG@3: mixed 0.5957 → raw-only 0.7160 (**+20.2%**).
- 변경: stage-2 프로필 검색은 `embed_summary`만 사용한다. Attention 인코더 검색은 `encode_user`와 `learned_embedding` 공간 안에서 별도로 유지한다.
- 결정: **채택**. 서로 다른 임베딩 공간을 한 코사인 검색에서 혼합하지 않는다.

### T-001 — 근거 점수 포화 해소

- 문제: P8 원본과 근거 강화 기사 모두 evidence 10점으로 동률.
- 변경: 독립 출처 2개 이상, 추적 가능한 1차 자료·연구명을 evidence predicate에 추가. JSON 예시를 질문 수에 맞춰 자동 생성.
- 결과: P8 76 → 82로 기대 방향 회복.
- 결정: **채택**.

### T-002 — 논리 불변쌍 인간 직관 보정

- 문제: 기존 P5 삽입문은 “전문가들의 반대 의견”이어서 인간 관점에서도 기사 자체의 모순이 아니었음.
- 변경: 기사 서술자가 해결을 직접 단정한 뒤 미해결이라고 부정하는 명백한 자기모순으로 fixture 수정. 상반된 인용과 자기모순을 구분하는 predicate 추가.
- 결과: 원본 64, 모순 삽입본 42로 기대 방향 회복.
- 결정: **테스트 fixture와 predicate 모두 채택**.

### T-003 — 가중치 다목적 탐색

- 불변쌍: 새 criterion + 수정 P5에서 W2와 근거 상한 적용 시 9/9.
- 서울대 팩트체크 30건 동일 criterion 결과 재가중치:

| 조합 | Spearman ρ | 고·저신뢰 평균 격차 | 극단 레이블 AUC |
|---|---:|---:|---:|
| W0 | 0.329 | 9.2 | 0.730 |
| W1 | 0.348 | 13.5 | 0.745 |
| W2 | **0.391** | **15.7** | **0.775** |
| W3(기존) | 0.308 | 10.7 | 0.710 |
| W4 | 0.316 | 10.7 | 0.695 |

- 채택: W2 = source 0.20, evidence 0.20, style 0.25, logic 0.15, clickbait -0.20.
- 추가 규칙: evidence < 4이면 최대 69점.
- 기존 W3 대비 Spearman **+26.9%**, 극단 AUC **+9.2%**.
- 결정: **W2 + evidence gate 채택**.

### T-004 — 부적합 데이터셋 식별

- `trust_eval_samples.jsonl`은 기존 프로젝트 문서에도 AI-HUB 낚시성 기사로 사실성 평가에 부적합하다고 기록된 세트.
- 실제 실행에서도 신뢰 평균 53.9, 불신 평균 65.5로 역전.
- 결정: **사실성 성능 평가에서 제외**. 문체/낚시성 별도 태스크에만 사용.

### T-005 — Google Search 외부 주장 교차검증

- 평가 단위 수정: 서울대 레이블은 기사 전체가 아니라 `_fc_title`의 검증 주장에 붙으므로, 해당 주장을 검색하고 본문은 문맥으로만 사용했다.
- 30건 grounding 결과와 검색 출처를 `trust_grounding.jsonl`에 캐시했다. 운영 비용을 막기 위해 `--live` 없이는 외부 호출하지 않는다.
- 전체 30건 진단:
  - W2 단독: Spearman 0.391, 극단 AUC 0.775.
  - W2 80% + grounding 20%: Spearman 0.552, 극단 AUC 0.860.
  - W2 70% + grounding 30%: Spearman 0.569, 극단 AUC 0.860.
- 5-fold에서 각 train fold만으로 혼합 비율을 고른 out-of-fold 결과:
  - Spearman 0.391 → 0.462 (**+18.1%**).
  - 극단 AUC 0.775 → 0.800 (**+3.2%**).
  - 선택 alpha 평균 0.28.
- 해석: 전체 데이터 진단에서는 목표 ρ≥0.5를 넘었지만, 더 보수적인 out-of-fold 값은 0.462이므로 목표 달성을 확정하지 않는다.
- 결정: **오프라인 선택 기능으로 채택, 기본 운영 점수 반영은 보류**. 검색 비용·지연, 역사적 주장에 대한 현재 시점 자료 혼입, 출처 품질을 더 검증해야 한다.

## 회귀검증

- `backend.tests.test_trust_scoring`
- `backend.tests.test_grounded_trust`
- `backend.tests.test_recommendation_samples`
- `backend.tests.test_evaluate_offline`
- `backend.tests.test_recommendation_attention`
- `backend.tests.test_smoke`
- 외부 grounding 파서 및 임베딩 공간 분리 후 총 36개 테스트 통과.
- 신뢰도 불변쌍 W2: 9/9 통과.
- 실제 로그 hybrid: profile·category·latest·random 기준선보다 MRR/NDCG@3 우수.

## 남은 위험과 다음 실험

1. 외부 검색 혼합은 전체 진단 ρ=0.552(alpha=0.2)지만 5-fold OOF는 0.462다. 100건 이상 시간 분할 holdout에서 0.5 이상을 재현하기 전 기본 운영 점수에는 넣지 않는다.
2. 임베딩 공간 분리는 완료했다. 이후 재학습 모델을 배포할 때 raw 공간과의 정렬 손실을 학습하지 않는 한 두 경로를 다시 혼합하지 않아야 한다.
3. Render에서 `rec_source`별 노출·클릭과 p95 latency를 수집해 hybrid 효과를 온라인 검증해야 한다.
4. 실제 temporal 리플레이를 300건 이상으로 늘리고 사용자 단위 bootstrap 신뢰구간을 추가해야 한다.
5. 팩트체크 평가를 100건 이상으로 확대하고 모델·프롬프트·코드 지문을 캐시에 함께 저장해야 한다.
