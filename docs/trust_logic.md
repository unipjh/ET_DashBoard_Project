# 신뢰도 분석 모듈 설계 문서

**파일**: `backend/services/trust.py`
**모델**: Gemini 2.5 Flash (`gemini-2.5-flash`)
**방식**: TELLER 논문 기반 + CoT(Chain-of-Thought) + Weighted Sum Rule

---

## 1. 평가 기준 및 가중치

| 기준 키 | 한국어 명칭 | 가중치 | 방향 |
|---|---|---|---|
| `source_credibility` | 출처 신뢰성 | +0.25 | 높을수록 좋음 |
| `evidence_support` | 근거 지지도 | +0.25 | 높을수록 좋음 |
| `style_neutrality` | 문체 중립성 | +0.20 | 높을수록 좋음 |
| `logical_consistency` | 논리 일관성 | +0.20 | 높을수록 좋음 |
| `clickbait_risk` | 어뷰징 위험도 | **-0.10** | **높을수록 나쁨 (역방향)** |

---

## 2. 채점 루브릭

### source_credibility — 출처 신뢰성

| 점수 | 기준 |
|---|---|
| 10점 | 메이저 공식 언론사(조선·중앙·한겨레 등), 정부/공공기관, 검증된 전문 매체 |
| 5점 | 중소 언론사 또는 출처가 부분적으로 확인됨 |
| 0점 | 출처 불분명한 블로그, 유언비어, 찌라시 |

### evidence_support — 근거 지지도

| 점수 | 기준 |
|---|---|
| 10점 | 구체적 수치·통계, 실명 전문가 인용, 공식 자료 다수 포함 |
| 5점 | 일부 인용이나 근거가 있으나 불충분 |
| 0점 | 근거 없는 주관적 주장만 있음 |

### style_neutrality — 문체 중립성

| 점수 | 기준 |
|---|---|
| 10점 | 건조하고 사실 중심의 객관적 문체, 감정 표현 없음 |
| 5점 | 일부 감정적 표현이 있으나 전반적으로 중립 |
| 0점 | 매우 감정적·선동적·편향된 표현이 지배적 |

### logical_consistency — 논리 일관성

| 점수 | 기준 |
|---|---|
| 10점 | 전후 문맥이 완벽히 이어지고 주장이 일관됨 |
| 5점 | 일부 논리적 비약이 있으나 전체 흐름은 파악 가능 |
| 0점 | 스스로 모순되거나 논리적 비약이 심해 신뢰 불가 |

### clickbait_risk — 어뷰징 위험도 ⚠️ 역방향

> **주의**: 이 항목은 점수가 높을수록 위험(나쁨)입니다.
> "어뷰징 위험이 낮다" → 낮은 점수(0~3), "어뷰징 위험이 높다" → 높은 점수(7~10)

| 점수 | 기준 |
|---|---|
| 10점 | 제목과 내용이 완전 불일치하거나 극도로 자극적인 어뷰징 기사 |
| 5점 | 제목이 다소 과장되었으나 내용과 연결은 됨 |
| 0점 | 제목이 본문 내용을 정직하고 정확하게 담고 있음 |

---

## 3. 점수 산출 공식

### 가중 합산 (Weighted Sum)

```
raw = source_credibility×0.25 + evidence_support×0.25
    + style_neutrality×0.20  + logical_consistency×0.20
    + clickbait_risk×(-0.10)

최대 raw = (0.25+0.25+0.20+0.20)×10 = 9.0  (clickbait=0일 때)
최소 raw = (0.25+0.25+0.20+0.20)×0 + (-0.10)×10 = -1.0

score = clamp(raw / 9.0 × 100, 0, 100)
```

### 판정 (verdict)

| score 범위 | verdict | 의미 |
|---|---|---|
| 70 이상 | `likely_true` | 신뢰 가능 |
| 40~69 | `uncertain` | 불확실 |
| 39 이하 | `likely_false` | 신뢰 불가 |

---

## 4. CoT 적용 방식

기존 프롬프트는 LLM이 score를 먼저 결정하고 reason을 끼워 맞추는 경향이 있었습니다.
이를 방지하기 위해 다음 두 가지를 적용했습니다.

### ① 프롬프트 지시문
```
각 기준을 평가할 때 다음 순서를 엄수하세요:
1. 기사에서 해당 기준과 관련된 근거를 먼저 찾는다.
2. 찾은 근거를 바탕으로 reason을 1~2문장으로 작성한다.
3. reason을 확인한 뒤 score를 결정한다.
절대로 score를 먼저 정하고 reason을 끼워 맞추지 마세요.
```

### ② JSON 키 순서 강제
```json
// 변경 전 (score 먼저)
{"score": 7, "reason": "..."}

// 변경 후 (reason 먼저 → CoT 유도)
{"reason": "...", "score": 7}
```

JSON 키 순서를 `reason → score`로 강제함으로써 LLM이 근거를 먼저 생성하도록 유도합니다.

---

## 5. 파싱 로직

### `_parse_criterion(item)` 헬퍼

```python
def _parse_criterion(item: object) -> dict:
    if not isinstance(item, dict):
        return {"score": 0, "reason": ""}
    try:
        score = int(float(item.get("score", 0)))  # "7.5" 같은 float string 처리
        score = max(0, min(10, score))             # 0~10 범위 강제
    except (TypeError, ValueError):
        score = 0
    return {
        "score": score,
        "reason": str(item.get("reason", "") or ""),
    }
```

**처리하는 예외 케이스**:
- 해당 키가 JSON에 없는 경우 (`None` → 기본값 반환)
- score가 `"7.5"` 같은 float string인 경우 (`int(float(...))` 처리)
- score가 범위 초과인 경우 (0~10으로 clamp)
- item이 dict가 아닌 경우 (타입 가드)

---

## 6. 반환 포맷

```python
{
    "score":        int,   # 0~100
    "verdict":      str,   # "likely_true" | "uncertain" | "likely_false"
    "reason":       str,   # 종합 판단 근거 (overall_reason)
    "per_criteria": {
        "source_credibility":  {"score": int, "reason": str},
        "evidence_support":    {"score": int, "reason": str},
        "style_neutrality":    {"score": int, "reason": str},
        "logical_consistency": {"score": int, "reason": str},
        "clickbait_risk":      {"score": int, "reason": str},
    }
}
```

반환 포맷은 변경 전과 동일합니다 (하위 호환 유지).

---

## 7. 오류 처리

| 상황 | 동작 |
|---|---|
| Gemini API 429 (할당량 초과) | 30~60초 랜덤 대기 후 자동 재시도 |
| 기타 API 호출 실패 | `_fallback()` 반환 (score=0, verdict="uncertain") |
| JSON 파싱 실패 | `_fallback()` 반환 |
| 개별 기준 score 누락/오류 | `_parse_criterion()`에서 score=0, reason="" 반환 |

---

## 8. 향후 개선 방향

- **가중치 조정**: 루브릭 기반 채점 결과를 실제 데이터로 검증 후 WEIGHTS 재조정
- **few-shot 예시 추가**: 프롬프트에 실제 기사 채점 예시(1~2개) 삽입으로 일관성 추가 향상
- **Decision System 도입**: TD1 계획대로 sklearn RandomForestClassifier 연동 (현재는 Weighted Sum Rule만 사용)
