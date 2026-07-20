# CLAUDE.md — trust.py 리팩토링 작업 지침

## 작업 개요

`backend/services/trust.py`를 3개 Phase로 단계적으로 개선한다.
각 Phase는 독립 배포 가능한 단위이며, Phase 순서를 반드시 지킨다.

---

## 배경 및 수정 계기

### 현재 구조의 문제 (왜 바꾸는가)

**M2 — source_credibility를 LLM이 판단하는 구조**
- 동일 언론사에 대해 호출마다 점수가 달라지는 비결정적 동작 발생
- LLM이 기사 본문에서 출처를 역추론하거나 학습 편향이 개입할 여지 있음
- 출처 정보 미상일 때 무조건 0점 → 전체 신뢰도의 25%가 날아감
- 해결: 룰 기반 사전(SOURCE_TIER)으로 완전 교체, LLM 프롬프트에서 제거

**M3 — 텍스트 3000자 하드컷**
- 한국어 기사는 결론/핵심이 후반부에 오는 구조가 많음
- logical_consistency 평가 시 후반부 유실로 오평가 가능
- 해결: 앞 2000자 + 뒤 1000자 슬라이싱으로 교체

**M3 — 0~10 점수를 LLM이 직접 생성**
- LLM의 숫자 생성 편향 존재 (7점 선호, 극단값 회피 등)
- calibration 없이 점수 신뢰 어려움
- TELLER 원논문(ACL 2024)에서도 Yes/No 이진 판단 후 집계 방식 사용
- 해결: 각 기준을 Yes/No 질문 여러 개로 분해, Yes 비율로 점수 산출

**M3 — 루브릭이 0/5/10 이산값인데 실제 응답은 중간값**
- 루브릭과 실제 채점 행동 불일치
- 해결: 0/3/5/7/10 세분화 + 각 단계별 행동 기술 추가

**M4 — 파싱 실패와 API 실패가 동일 처리**
- 어떤 원인으로 실패했는지 로그에서 구분 불가
- Gemini가 가끔 ```json 마크다운 펜스로 감싸서 반환 → json.loads 폭발
- 해결: 3단계 폴백 파싱 + 실패 유형 플래그 분리

**M5 — 가중치 하드코딩 + 매직 넘버 9.0**
- 9.0은 clickbait_risk=0일 때의 최대 raw값을 역산한 것
- 가중치 변경 시 9.0도 같이 바꿔야 하는데 연결이 명시적이지 않음
- 교차 패널티 없음: 감정 과잉 + 근거 부족 조합(가짜뉴스 핵심 패턴)을 단순합산으로 못 잡음
- 해결: MAX_RAW 동적 계산 + 교차 패널티 함수 추가

**M6 — 무한 재시도 루프 + print 로깅**
- 429가 계속 오면 영원히 돌아감
- print()는 프로덕션에서 추적 불가
- fallback의 score:0이 verdict:"uncertain"인데 논리적으로 "likely_false"여야 함
- 해결: MAX_RETRIES=3 상한 + logging 모듈 교체 + fallback verdict 수정

---

## 수정 범위 및 제약

- **수정 대상**: `backend/services/trust.py` 단독
- **출력 스키마 유지 필수**: 반환 dict의 키 구조(`score`, `verdict`, `reason`, `per_criteria`)는 변경 금지. 대시보드 연동이 이 스키마에 의존함
- **per_criteria 키 유지 필수**: `source_credibility`, `evidence_support`, `style_neutrality`, `logical_consistency`, `clickbait_risk` 5개 키는 그대로 유지
- **Gemini API 유지**: LLM 벤더 변경 없음, `gemini-2.5-flash` 모델 유지
- **Phase 순서 준수**: Phase 1 완료 후 Phase 2 진행. Phase 3은 별도 파일로 분리

---

## Phase 1 — 구조 안정화

> 출력 형식 변경 없음. 기존 동작 유지하면서 안정성만 높인다.

### 1-1. SOURCE_TIER 사전 구축 (M2)

`source_credibility`를 LLM 평가에서 완전히 제거하고 룰 기반으로 교체한다.

```python
# 파일 상단 상수로 정의
SOURCE_TIER: dict[int, set[str]] = {
    10: {"연합뉴스", "KBS", "MBC", "SBS", "YTN"},
    8:  {"조선일보", "중앙일보", "동아일보", "한겨레", "경향신문",
         "한국일보", "국민일보", "서울신문", "세계일보", "문화일보"},
    6:  {"머니투데이", "뉴시스", "아시아경제", "뉴스1", "헤럴드경제",
         "이데일리", "파이낸셜뉴스", "아주경제", "데일리안", "매일경제"},
    3:  set(),   # 추후 저신뢰 매체 추가용 슬롯
}
SOURCE_DEFAULT_SCORE = 5  # 미상 또는 사전 미등록 → 중립 처리
```

```python
def _rule_source_score(source: str) -> dict:
    """
    출처명을 SOURCE_TIER 사전에서 룩업하여 점수 반환.
    - 완전 일치 우선, 부분 포함 차선
    - 미상/미등록: SOURCE_DEFAULT_SCORE(5점) 반환
    - LLM 호출 없음
    """
    if not source or source == "미상":
        return {"score": SOURCE_DEFAULT_SCORE, "reason": "출처 미상 — 중립 처리"}
    
    for score, names in sorted(SOURCE_TIER.items(), reverse=True):
        if source in names:
            return {"score": score, "reason": f"등록 언론사: {source}"}
    
    # 부분 일치 (예: "조선일보 경제" → "조선일보" 매칭)
    for score, names in sorted(SOURCE_TIER.items(), reverse=True):
        for name in names:
            if name in source:
                return {"score": score, "reason": f"부분 일치: {name}"}
    
    return {"score": SOURCE_DEFAULT_SCORE, "reason": f"미등록 언론사: {source} — 중립 처리"}
```

LLM 프롬프트(`_PROMPT_TEMPLATE`)에서 `source_credibility` 기준을 제거하고,
`score_trust()` 함수에서 `per_criteria["source_credibility"]`를 `_rule_source_score(source_str)` 결과로 직접 채운다.
프롬프트는 나머지 4개 기준만 평가하도록 수정한다.

### 1-2. 텍스트 슬라이싱 개선 (M3)

```python
def _trim_text(text: str, front: int = 2000, tail: int = 1000) -> str:
    """앞 + 뒤 슬라이싱으로 결론부 보존."""
    if len(text) <= front + tail:
        return text
    return text[:front] + "\n\n...(중략)...\n\n" + text[-tail:]
```

`score_trust()` 내부의 `text[:3000]`을 `_trim_text(text)`로 교체한다.

### 1-3. 3단계 폴백 파싱 (M4)

```python
import re

def _safe_json_parse(raw: str) -> dict | None:
    """
    3단계 폴백 파싱.
    1차: 직접 json.loads
    2차: ```json ... ``` 펜스 제거 후 파싱
    3차: 정규식으로 { ... } 블록 추출 후 파싱
    전부 실패 시 None 반환
    """
    # 1차
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # 2차: 마크다운 펜스 제거
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # 3차: 중괄호 블록 추출
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return None
```

`_parse_criterion` 반환값에 `_parse_failed` 플래그 추가:
```python
def _parse_criterion(item: object) -> dict:
    if not isinstance(item, dict):
        return {"score": 0, "reason": "", "_parse_failed": True}
    # ... 기존 로직 유지 ...
    return {"score": score, "reason": reason}  # 정상 파싱 시 플래그 없음
```

### 1-4. MAX_RETRIES + logging 교체 (M6)

```python
import logging
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
```

`score_trust()` 내부 while True 루프를 for 루프로 교체:
```python
for attempt in range(MAX_RETRIES):
    try:
        response = model.generate_content(prompt)
        raw_json = _safe_json_parse(response.text)
        if raw_json is None:
            raise ValueError("JSON 파싱 전부 실패")
        break
    except Exception as e:
        err = str(e)
        if "429" in err or "Quota exceeded" in err:
            if attempt == MAX_RETRIES - 1:
                logger.error("[trust] 할당량 초과 — 재시도 소진 (%d회)", MAX_RETRIES)
                return _fallback("할당량 초과")
            wait = random.randint(30, 60)
            logger.warning("[trust] 할당량 초과, %d초 후 재시도 (%d/%d)", wait, attempt+1, MAX_RETRIES)
            time.sleep(wait)
        else:
            logger.error("[trust] Gemini 호출 실패: %s", e)
            return _fallback(err)
```

`_fallback` verdict 수정:
```python
def _fallback(error_msg: str) -> dict:
    return {
        "score": 0,
        "verdict": "likely_false",   # 0점이면 likely_false가 논리적으로 맞음
        "reason": f"신뢰도 분석 실패: {error_msg}",
        "per_criteria": {k: {"score": 0, "reason": "분석 실패"} for k in WEIGHTS},
    }
```

### 1-5. 동적 정규화 (M1/M5)

```python
# WEIGHTS에서 MAX_RAW를 동적으로 계산
# clickbait_risk 가중치는 음수이므로 abs() 처리
_MAX_RAW = sum(abs(w) for w in WEIGHTS.values()) * 10  # 모든 기준 최고점일 때

def _weighted_sum_score(criteria: dict) -> int:
    raw = sum(criteria[k]["score"] * w for k, w in WEIGHTS.items())
    return int(max(0, min(100, raw / _MAX_RAW * 100)))
```

---

## Phase 2 — 평가 로직 고도화

> Phase 1 완료 후 진행. 점수 분포가 달라지므로 Phase 3 검증 병행 권장.

### 2-1. Yes/No Predicate 분해 (M3)

각 평가 기준을 Yes/No 질문 묶음으로 분해한다.
LLM은 숫자 점수를 직접 생성하지 않고, 각 질문에 "yes"/"no"/"uncertain"만 답한다.
Yes 비율을 점수로 변환하여 LLM 숫자 편향을 제거한다.

```python
PREDICATES: dict[str, list[str]] = {
    "evidence_support": [
        "기사에 구체적인 수치나 통계가 포함되어 있는가?",
        "실명의 전문가 또는 공식 기관이 인용되어 있는가?",
        "인용된 자료의 출처(기관명, 보고서명 등)가 명시되어 있는가?",
    ],
    "style_neutrality": [
        "기사 제목에 감정적이거나 자극적인 표현이 없는가?",
        "본문에 선동적이거나 편향된 언어가 사용되지 않았는가?",
        "사실과 의견이 명확히 구분되어 서술되어 있는가?",
    ],
    "logical_consistency": [
        "기사의 전후 문맥이 일관되게 이어지는가?",
        "기사 내에 서로 모순되는 주장이 없는가?",
        "제목에서 제시한 주장이 본문에서 뒷받침되는가?",
    ],
    "clickbait_risk": [
        "제목이 본문 내용을 과장하거나 왜곡하고 있는가?",   # 역방향: Yes → 위험
        "독자의 클릭을 유도하기 위한 자극적 표현이 제목에 있는가?",  # 역방향
        "기사 내용이 제목이 암시하는 것과 실질적으로 다른가?",  # 역방향
    ],
}

NEGATIVE_PREDICATES = {"clickbait_risk"}  # Yes가 나쁜 방향인 기준
```

프롬프트를 아래 구조로 전면 교체:

```
[평가 방식]
각 질문에 반드시 "yes", "no", "uncertain" 중 하나만 답하세요.
이유(reason)를 먼저 1문장으로 쓴 뒤 answer를 결정하세요.

[질문 목록]
evidence_support:
  Q1: 기사에 구체적인 수치나 통계가 포함되어 있는가?
  Q2: ...
  
[반환 형식 — JSON만, 설명 텍스트 없이]
{
  "evidence_support": [
    {"reason": "...", "answer": "yes"},
    {"reason": "...", "answer": "no"},
    ...
  ],
  ...
  "overall_reason": "신뢰도를 낮추는 핵심 요인 1~2개를 한국어 2문장으로 서술"
}
```

점수 변환 함수:
```python
def _predicate_to_score(answers: list[dict], is_negative: bool = False) -> dict:
    """
    Yes/No 답변 목록을 0~10 점수로 변환.
    - 정방향 기준: Yes 비율이 높을수록 고점
    - 역방향 기준(clickbait_risk 등): Yes 비율이 높을수록 저점
    - uncertain은 0.5로 처리
    """
    if not answers:
        return {"score": 5, "reason": "답변 없음"}
    
    values = []
    for a in answers:
        ans = str(a.get("answer", "")).lower()
        if ans == "yes":
            values.append(1.0)
        elif ans == "no":
            values.append(0.0)
        else:  # uncertain
            values.append(0.5)
    
    ratio = sum(values) / len(values)
    
    if is_negative:
        score = int((1 - ratio) * 10)  # 역방향: Yes 많을수록 낮은 점수
    else:
        score = int(ratio * 10)
    
    reasons = [a.get("reason", "") for a in answers if a.get("reason")]
    return {"score": max(0, min(10, score)), "reason": " / ".join(reasons[:2])}
```

### 2-2. 교차 패널티 (M5)

```python
def _cross_penalty(per_criteria: dict) -> float:
    """
    감정 과잉 + 근거 부족 조합 패널티.
    가짜뉴스의 핵심 패턴: 자극적인데 근거가 없음.
    style_neutrality 낮음(감정적) AND evidence_support 낮음(근거 없음) → -10점
    """
    neutrality = per_criteria["style_neutrality"]["score"]
    evidence   = per_criteria["evidence_support"]["score"]
    
    if neutrality < 4 and evidence < 4:
        logger.debug("[trust] 교차 패널티 적용: neutrality=%d, evidence=%d", neutrality, evidence)
        return -10.0
    return 0.0
```

`_weighted_sum_score`에 패널티 반영:
```python
def _weighted_sum_score(criteria: dict) -> int:
    raw = sum(criteria[k]["score"] * w for k, w in WEIGHTS.items())
    raw_penalized = raw + _cross_penalty(criteria)
    return int(max(0, min(100, raw_penalized / _MAX_RAW * 100)))
```

### 2-3. 루브릭 세분화 및 overall_reason 지침 (M3)

Phase 2-1의 Yes/No 방식 전환 후에는 기존 0/5/10 루브릭이 불필요해진다.
대신 `overall_reason` 생성 지침을 프롬프트에 명시 추가:

```
overall_reason 작성 규칙:
- 위 평가 기준 중 "no" 답변이 가장 많은 기준을 중심으로 서술
- 신뢰도를 낮추는 핵심 요인 1~2개를 구체적으로 지목
- 한국어 2문장 이내
- 예시: "근거 지지도가 낮습니다. 전문가 인용이나 통계 수치 없이 주관적 주장으로만 구성되어 있습니다."
```

---

## Phase 3 — 검증 레이어 (별도 파일)

> `trust.py` 미변경. `backend/services/eval_trust.py`로 분리 작성.

### eval_trust.py 구성

```python
"""
backend/services/eval_trust.py
trust.py 점수의 타당성 검증 스크립트.

사용법:
    python -m backend.services.eval_trust --data data/trust_eval_samples.jsonl
    
입력 형식 (jsonl, 한 줄 = 한 기사):
    {"text": "...", "source": "연합뉴스", "label": 1}
    # label: 1=신뢰, 0=불신, 0.5=보통

출력:
    - Spearman 상관계수 (전체 점수 vs 레이블)
    - 기준별 점수 분포 (boxplot)
    - verdict 혼동 행렬
    - clickbait_risk F1 (이진 분류 기준)
"""
```

검증 통과 기준 (이 수치를 목표로 설정):
```
Spearman ρ ≥ 0.5        — 점수가 레이블 방향과 일치
clickbait_risk F1 ≥ 0.6  — AI-HUB 낚시성 레이블 대비
신뢰 그룹 평균 ≥ 65점    — 팩트체크 기사 10건
불신 그룹 평균 ≤ 45점    — 낚시성 기사 10건
```

### 검증용 데이터 수집 가이드

`data/trust_eval_samples.jsonl` 파일을 아래 기준으로 수동 구성:

| 유형 | 출처 | 목표 수량 | 레이블 |
|------|------|-----------|--------|
| 팩트체크 통과 기사 | SNU 팩트체크 (snufactcheck.snu.ac.kr) | 10건 | 1 |
| 낚시성 기사 | AI-HUB 샘플 데이터 | 10건 | 0 |
| 중립 기사 | 네이버 뉴스 정치/경제 임의 수집 | 10건 | 0.5 |

AI-HUB 낚시성 기사 데이터셋 신청 URL:
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71338
(승인 전에도 샘플 데이터 다운로드 가능 — 샘플로 먼저 검증 진행)

---

## 파일 구조 (최종)

```
backend/services/
├── trust.py          # Phase 1, 2 적용 대상
├── eval_trust.py     # Phase 3 — 신규 작성
└── config.py         # 변경 없음

data/
└── trust_eval_samples.jsonl  # 수동 구성 검증 데이터
```

---

## 작업 순서 요약

```
1. Phase 1 전체 적용 (trust.py 수정)
   1-1 SOURCE_TIER + _rule_source_score
   1-2 _trim_text
   1-3 _safe_json_parse + _parse_criterion 플래그
   1-4 MAX_RETRIES + logging
   1-5 _MAX_RAW 동적 계산 + _fallback verdict 수정

2. Phase 1 동작 확인
   - 기존 기사 3건 이상으로 score_trust() 호출
   - 반환 스키마 키 구조 변경 없음 확인
   - source_credibility가 룰 기반으로 결정됨 확인

3. Phase 2 전체 적용 (trust.py 수정)
   3-1 PREDICATES 상수 정의
   3-2 프롬프트 전면 교체 (Yes/No 방식)
   3-3 _predicate_to_score 함수 추가
   3-4 _cross_penalty 함수 추가
   3-5 score_trust() 내 per_criteria 조립 로직 수정

4. Phase 3 작성 (eval_trust.py 신규)
   - argparse로 --data 경로 받기
   - jsonl 로드 → score_trust() 순차 호출
   - Spearman, F1, 분포 시각화 출력
```

---

## 주의사항

- `per_criteria` 딕셔너리의 5개 키는 어떤 Phase에서도 제거하거나 이름 변경 금지
- Phase 2에서 Yes/No 방식으로 바꾸더라도 최종 반환값의 `score`(0~10 int)와 `reason`(str) 구조는 유지
- `clickbait_risk`의 역방향 처리는 `_predicate_to_score(is_negative=True)`로 일관되게 처리
- SOURCE_TIER는 이후 운영 중 언론사 추가/수정이 쉽도록 상수로 분리 유지
- eval_trust.py는 trust.py를 import해서 사용 — trust.py 내부 private 함수(_로 시작)는 직접 호출하지 않음