# trust_fix_plan.md — 한계 보수적 수정 및 검증 계획

**작성일**: 2026-04-05
**기준**: `docs/trust_limitations.md` L1~L8, O1~O3
**원칙**: 각 수정은 독립적으로 적용하고, 적용 전후 invariant_pairs 통과율이 **낮아지지 않아야** 채택한다.

---

## 0. 보수적 접근 원칙

1. **건드리지 않는 것 먼저 결정한다** — 고치려다 기존 작동을 망가뜨리는 게 더 위험
2. **한 번에 하나씩** — 변경 1개 → 검증 → 통과 → 다음으로
3. **통과율 기준선 유지** — 현재 7/9(77.8%). 수정 후 이 수치가 내려가면 롤백
4. **LLM 호출은 최소화** — 검증은 `--from-raw` 캐시 재활용 우선, 새 호출 불가피한 경우에만 진행
5. **설계 범위 밖은 손대지 않는다** — L1(잘 쓰인 오보), L8(시계열 맥락)은 수정 대상 제외

---

## 1. 수정 대상 분류

| 항목 | 대상 여부 | 이유 |
|------|-----------|------|
| L1 잘 쓰인 오보 | ❌ 제외 | 설계 범위 밖, 수정 불가 |
| L2 논리 모순 감지 약함 | ✅ Tier B | 프롬프트 수정, 검증 필요 |
| L3 P8 출처 상쇄 | ✅ Tier B | 가중치 조정 실험 |
| L4 SOURCE_TIER 편향 | ✅ Tier A | 목록 추가만, 점수 로직 불변 |
| L5 텍스트 슬라이싱 | ✅ Tier B | 파라미터 조정, 검증 필요 |
| L6 uncertain 편향 | ✅ Tier B | 점수 변환 로직 조정 |
| L7 단신 불이익 | ✅ Tier A | 길이 감지 후 안내만, 점수 불변 |
| L8 맥락 부재 | ❌ 제외 | 현재 구조에서 구현 불가 |
| O1 fallback verdict | ✅ Tier A | 코드 1줄, 리스크 없음 |
| O2 통과율 상한 77.8% | ✅ Tier B | L2/L3 해결 시 개선 기대 |
| O3 캐시 시간 불일치 | ✅ Tier C | 캐시 재생성 절차 문서화 |

- **Tier A**: 즉시 적용 가능, 검증 부담 없음
- **Tier B**: 적용 전후 invariant_pairs 검증 필수
- **Tier C**: 운영 절차 수준, 코드 변경 없음

---

## 2. 수정 순서 및 상세

### [Tier A-1] O1 — fallback verdict 수정

**현재**:
```python
def _fallback(error_msg):
    return {"score": 0, "verdict": "likely_false", ...}
```

**변경**:
```python
def _fallback(error_msg):
    return {"score": None, "verdict": "unanalyzed", ...}
```

**이유**: API 장애 시 멀쩡한 기사를 "가짜뉴스 패턴"으로 표시하는 것이 가장 큰 사용자 오해 원인.

**검증**: invariant_pairs 무관 (fallback은 정상 흐름에서 호출되지 않음). 프론트엔드에서 `verdict === "unanalyzed"` 처리 추가 필요.

**rollback 조건**: 없음 (DB 저장값에만 영향, 기존 분석 기사는 재분석 필요 없음)

---

### [Tier A-2] L4 — tier 3 저신뢰 매체 목록 추가

**현재**:
```python
3: set(),  # 비어 있음
```

**변경 (추가만, 기존 tier 불변)**:
```python
3: {"온라인 커뮤니티", "카카오스토리", "네이버 블로그", "티스토리",
    "인스타그램", "페이스북", "유튜브"},
```

**이유**: 현재 미등록 매체는 전부 5점(중립). SNS/블로그 출처는 구조적으로 3점이 더 적절.

**검증**:
```bash
# P7 쌍: 조선일보(8점) vs 블로그뉴스
# 블로그뉴스가 tier 3에 없으면 5점 → 추가 후에도 P7 PASS 유지해야 함
python eval_trust.py --from-raw ../../data/eval_cache.jsonl --pairs-ref ../../data/invariant_pairs.jsonl --weights W3
```
기대: P7 여전히 PASS (조선일보 8점 > 블로그뉴스 3점, 현재 5점보다 차이 더 커짐)

**rollback 조건**: P7 FAIL로 바뀌면 롤백

---

### [Tier A-3] L7 — 단신 속보 길이 감지 (점수 불변)

**변경 방향**: 점수는 그대로, 반환값에 `short_article` 플래그 추가.

```python
# score_trust() 반환값에 추가
"flags": {
    "short_article": len(text) < 300  # 단신 속보 가능성
}
```

프론트엔드에서 해당 플래그가 True면 "단신 기사는 근거 지지도가 구조적으로 낮을 수 있습니다" 안내 표시.

**이유**: 점수 로직 무변경, 사용자에게 맥락 제공. 리스크 없음.

**검증**: invariant_pairs 점수 불변 확인 (플래그만 추가이므로 --from-raw 결과 동일해야 함)

---

### [Tier B-1] L2 — logical_consistency 프롬프트 강화

**현재 문제**: 미묘한 모순을 Gemini가 "uncertain"으로 회피.

**변경안**:
```
# 현재 질문
"기사 내에 서로 모순되는 주장이 없는가?"

# 변경
"기사 앞부분의 주장과 뒷부분 결론이 서로 일치하는가?
 만약 '전문가는 A라고 말했다'고 쓴 뒤 결론에서 'B가 맞다'고 쓴다면 no로 답하라."
```

**검증 절차**:
1. 프롬프트 변경 후 `--save-raw` 재실행 (LLM 17회 호출)
2. 새 캐시로 W3 통과율 확인
3. 기준: 통과율 ≥ 7/9 (현재 수준 유지) + P5 PASS 달성 시 채택

**rollback 조건**: 통과율 7/9 미만이면 프롬프트 원복

---

### [Tier B-2] L3/P8 — source_credibility 가중치 하향 실험

**현재 W3**: `source_credibility: 0.20`

**실험안 W4**:
```python
"W4": {
    "source_credibility":  0.15,  # -0.05 추가 하향
    "evidence_support":    0.30,  # +0.05 상향 (본문 품질 강조)
    "style_neutrality":    0.25,
    "logical_consistency": 0.20,
    "clickbait_risk":     -0.10,
}
```

**검증 절차**:
1. `eval_cache.jsonl` 재활용 (LLM 호출 없음)
2. W4 추가 후 `--from-raw --weights W4` 실행
3. 기준: P8 PASS + 전체 통과율 ≥ 7/9

**rollback 조건**: 기존에 PASS하던 쌍이 FAIL로 바뀌면 기각

---

### [Tier B-3] L6 — uncertain 점수 처리 조정

**현재**: `uncertain = 0.5`

**변경안**: `uncertain = 0.3` (명확한 판단 유도, 중간 쏠림 완화)

**이유**: 0.5는 사실상 "모르겠음 → 중립"이지만, 신뢰도 맥락에서 "모르겠다"는 약간 부정적 신호에 가까움.

**검증 절차**:
1. `eval_cache.jsonl` 재활용 (LLM 호출 없음)
2. `_predicate_to_score` 함수에서 uncertain 값 0.3으로 변경
3. 전체 9쌍 점수 재계산, PASS/FAIL 변화 확인
4. 기준: 통과율 ≥ 7/9 유지

**rollback 조건**: 통과율 하락 시 0.4로 재시도, 그래도 하락이면 원복

---

### [Tier B-4] L5 — 텍스트 슬라이싱 파라미터 조정

**현재**: `front=2000, tail=1000`

**변경안**: `front=2500, tail=1500` (중간 누락 범위 축소)

**이유**: Gemini 2.5 Flash는 긴 컨텍스트 처리 가능 → 슬라이싱 확대 부담 낮음.

**검증 절차**:
1. 변경 후 `--save-raw` 재실행 (LLM 재호출 필요)
2. 통과율 ≥ 7/9 유지 확인
3. API 비용 모니터링 (토큰 증가분 확인)

**rollback 조건**: 통과율 하락 또는 비용 과다 시 원복

---

### [Tier C] O3 — 캐시 무효화 절차

코드 변경 없음. 운영 절차로 관리.

```
캐시 재생성이 필요한 시점:
  - Gemini 모델 버전 변경 시 (TRUST_MODEL 상수 변경)
  - invariant_pairs.jsonl 내용 수정 시
  - 6개월 경과 시 (정기 재검증)

재생성 명령:
  cd backend/services
  python eval_trust.py --data ../../data/invariant_pairs.jsonl \
      --save-raw --cache-out ../../data/eval_cache.jsonl
```

---

## 3. 전체 실행 순서

```
Phase 1 (Tier A — 리스크 없음, 즉시 가능)
  ├── A-1: fallback verdict → "unanalyzed"
  ├── A-2: tier 3 매체 목록 추가 + --from-raw 검증
  └── A-3: short_article 플래그 추가 + 프론트엔드 안내

Phase 2 (Tier B — 캐시 재활용, LLM 호출 없음)
  ├── B-2: W4 가중치 실험 (eval_cache 재활용)
  └── B-3: uncertain=0.3 실험 (eval_cache 재활용)

Phase 3 (Tier B — LLM 재호출 필요)
  ├── B-1: logical_consistency 프롬프트 강화 → --save-raw 재실행
  └── B-4: 슬라이싱 파라미터 확대 → --save-raw 재실행

Phase 4 (Tier C — 운영 절차)
  └── C: 캐시 재생성 주기 운영 문서화
```

---

## 4. 채택 판정 기준 (공통)

| 결과 | 조치 |
|------|------|
| 통과율 ≥ 8/9 | 즉시 채택, trust.py 반영 |
| 통과율 = 7/9 (현 수준 유지) | 채택 (기존보다 나쁘지 않음) |
| 통과율 ≤ 6/9 | 기각, 원복 |
| 기존 PASS 쌍이 FAIL로 전환 | 기각 (개선이 퇴보를 만들면 안 됨) |

---

## 5. 현재 기준선 (수정 전 참조값)

```
W3 기준 (2026-04-05 eval_cache.jsonl):

P1 출처 신호:   score(A)=75 vs score(A')=57 → PASS
P2 근거 신호:   score(A)=62 vs score(A')=0  → PASS
P3 감정 신호:   score(A)=70 vs score(A')=52 → PASS
P4 제목 신호:   score(A)=58 vs score(A')=54 → PASS
P5 논리 신호:   score(A)=45 vs score(A')=45 → FAIL  ← B-1 목표
P6 복합 신호:   score(A)=90 vs score(A')=0  → PASS
P7 출처 등급:   score(A)=72 vs score(A')=66 → PASS
P8 근거 강도:   score(A)=80 vs score(A')=66 → FAIL  ← B-2 목표
P9 단신 속보:   score=62  범위=[40,69]       → PASS

통과율: 7/9 (77.8%)
```
