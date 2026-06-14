# trust_dev2.md — 진행 현황 및 다음 단계

**작성일**: 2026-03-29
**기준 브랜치**: `uidev`

---

## 1. 완료된 작업

### Phase 1 — trust.py 안정성 개선 (완료)

| 항목 | 내용 |
|------|------|
| `SOURCE_TIER` 룰 기반 점수 | 언론사 딕셔너리 룩업 → `source_credibility` 결정적 처리 |
| `_trim_text()` | 앞 2000자 + 뒤 1000자 슬라이싱 (토큰 초과 방지) |
| `_safe_json_parse()` | 직접 파싱 → fence 제거 → regex 블록 3단계 폴백 |
| `MAX_RETRIES=3` | `while True` → `for attempt in range(3)` |
| `logging` 모듈 | `print()` 전면 교체 |
| `_fallback()` | verdict `"uncertain"` → `"likely_false"` 수정 |

### Phase 2 — 평가 방식 개선 (완료)

| 항목 | 내용 |
|------|------|
| Yes/No predicate 분해 | 기준별 서술형 질문 → 이진 질문 집합으로 변환 |
| `_predicate_to_score()` | Yes 개수 → 0~10점 변환 |
| `_cross_penalty()` | `neutrality < 4 AND evidence < 4` → -10점 |
| `_MAX_RAW` 동적 계산 | `sum(abs(w) for w in WEIGHTS.values()) * 10` |
| 프롬프트 전면 교체 | 4개 기준 Yes/No 질문 구조 |

### Phase 3 — eval_trust.py 스켈레톤 (완료)

`backend/services/eval_trust.py` 작성:
- `--data` 인자로 jsonl 경로 받음
- `score_trust()` 순차 호출
- Spearman ρ, F1, 박스플롯 출력 구조 (scipy/sklearn/matplotlib 없으면 기본 통계 폴백)

### 첫 평가 실행 및 분석 (완료)

- **데이터**: `data/trust_eval_samples.jsonl` — AI-HUB 낚시성 기사 20건 (Real 10 / Fake 10)
- **결과**: Real 평균 46.1점, Fake 평균 59.6점 — **점수 역전**
- **원인 확정**: 데이터셋-평가 목적 불일치 (레이블=제목 선정성 vs trust.py=본문 신뢰도)
- **상세**: `docs/trust_dev1.md` 참조

### 이슈 정리 (완료)

`docs/trust_issue1.md`에 6개 이슈 문서화:
- I-1: 한국어 레이블 데이터 없음 (구조적)
- I-2: AI-HUB 레이블 불일치 (구조적)
- I-3: 교차 패널티 과도 감점 (로직, 보류)
- I-4: clickbait_risk 가중치 낮음 (로직, 보류)
- I-5: source=None 변별력 없음 (운영)
- I-6: 가중치·임계값 실증 검증 불가 (구조적)

### SNU 팩트체크 크롤 파이프라인 구축 (완료, 부분 성공)

`crawl_exp/` 폴더:

| 파일 | 역할 |
|------|------|
| `01_fetch_data.py` | GitHub에서 `snu_factcheck.csv` 다운로드 |
| `02_parse_sources.py` | source 필드 → TYPE_A/B/C/D/SKIP 분류 |
| `03_search_and_crawl.py` | Naver Search API + 기사 본문 크롤 |
| `04_build_eval.py` | 크롤 결과 → eval jsonl 조립 |

**source 분류 결과**: TYPE_A=178 / TYPE_B=358 / TYPE_C=531 / TYPE_D=571 / SKIP=3,124

**크롤 성공률** (TYPE_A 20건 샘플 기준): 60% (12/20)
- 기존 HTML 스크래핑 방식 실패 → Naver Search API(CLIENT_ID/SECRET)로 교체 후 해결
- `data/snu_crawled.jsonl`에 누적 저장, 재실행 시 자동 스킵(resumable)
- 최근 코드 수정: `sim` 값을 record에 저장하도록 변경 (나중에 유사도 기반 필터링 가능)

---

## 2. 현재 상태

```
data/snu_crawled.jsonl   — TYPE_A 일부 크롤 완료 (재실행으로 sim 추가 예정)
data/trust_eval_samples.jsonl — AI-HUB 샘플 20건 (AI-HUB, 평가 부적합)
```

**현재 TYPE_A 전체 크롤 실행 중** (`python crawl_exp/03_search_and_crawl.py --types TYPE_A`)
- sim 저장 코드 반영을 위해 완료 후 `snu_crawled.jsonl` 삭제 후 재실행 필요

---

## 3. 다음 단계

### Step 1. TYPE_A 크롤 완료 후 재실행 (즉시)

```bash
# 현재 실행 완료 후
rm data/snu_crawled.jsonl
python crawl_exp/03_search_and_crawl.py --types TYPE_A
```

목표: sim 값 포함된 TYPE_A 기사 확보 (~100건 예상)

### Step 2. 크롤 결과 품질 검토 후 범위 결정

TYPE_A 완료 후 성공률 확인:
- 성공률 ≥ 50% → TYPE_B까지 확장
- 성공률 < 50% → TYPE_A만으로 eval 구성

sim 임계값 결정: `sim >= 0.30` 이상만 사용 권장 (sim=0.10 수준은 엉뚱한 기사 가능성 높음)

### Step 3. eval jsonl 구성

```bash
python crawl_exp/04_build_eval.py --min-sim 0.30 --output data/trust_eval_snu.jsonl
```

SNU 팩트체크 레이블 매핑 확인 필요:
- `사실` / `대체로 사실` → label=1
- `절반의 사실` → label=0.5
- `대체로 사실 아님` / `사실 아님` / `전혀 사실 아님` → label=0

### Step 4. eval_trust.py로 재평가

```bash
python backend/services/eval_trust.py --data data/trust_eval_snu.jsonl
```

이번에는 올바른 레이블(본문 신뢰도와 직접 관련)이므로 Spearman ρ 측정이 유효함.
목표: ρ ≥ 0.3 (약한 양의 상관), Real 평균 ≥ Fake 평균

### Step 5. 결과 분석 → 파라미터 조정 여부 결정

평가 결과에 따라:
- ρ < 0 → 로직 문제 → I-3(교차 패널티), I-4(clickbait 가중치) 수정 검토
- ρ ≥ 0.3 → 현행 유지, 운영 환경 source 채워진 기사로 추가 검증
- source=None 비율 높으면 I-5 대응 (crawl.py source 수집 개선)

---

## 4. 보류 중인 사항

| 항목 | 보류 이유 | 재개 조건 |
|------|-----------|-----------|
| 교차 패널티 임계값 완화 (I-3) | 데이터 없이 조정 시 방향 불확실 | Step 4 결과 이후 |
| clickbait_risk 가중치 상향 (I-4) | 동일 | Step 4 결과 이후 |
| TYPE_B/C/D 크롤 확장 | TYPE_A 결과 검토 후 결정 | Step 2 이후 |
| Invariant 쌍 테스트 | 현재 SNU 데이터 확보 우선 | Step 3 이후 언제든 |

---

## 5. 관련 문서 및 파일

| 파일 | 내용 |
|------|------|
| `docs/trust_dev1.md` | Phase 1+2 첫 평가 상세 분석 |
| `docs/trust_issue1.md` | 이슈 6개 전체 목록 |
| `docs/trust_revise.md` | Phase 1+2+3 리팩토링 원본 지침 |
| `backend/services/trust.py` | 현재 구현 (Phase 1+2 적용) |
| `backend/services/eval_trust.py` | 평가 스크립트 (Phase 3) |
| `crawl_exp/` | SNU 팩트체크 크롤 파이프라인 |
| `data/snu_crawled.jsonl` | 크롤 결과 누적 파일 |
| `data/trust_eval_samples.jsonl` | AI-HUB 샘플 (평가 부적합, 참고용) |
