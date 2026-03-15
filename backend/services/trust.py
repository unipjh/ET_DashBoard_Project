# backend/services/trust.py
# TD1 — TELLER 기반 신뢰도 분석 모듈 (Gemini CoT + Weighted Sum Rule)

import json
import random
import time
import google.generativeai as genai
from backend.services.config import get_gemini_api_key

genai.configure(api_key=get_gemini_api_key())

# ============================================================
# 가중치 상수 (수동 조정 가능)
# clickbait_risk는 역방향: 높을수록 신뢰도 감점
# ============================================================
TRUST_MODEL = "gemini-2.5-flash"

WEIGHTS = {
    "source_credibility":  0.25,
    "evidence_support":    0.25,
    "style_neutrality":    0.20,
    "logical_consistency": 0.20,
    "clickbait_risk":     -0.10,
}

CRITERIA_LABELS = {
    "source_credibility":  "출처 신뢰성",
    "evidence_support":    "근거 지지도",
    "style_neutrality":    "문체 중립성",
    "logical_consistency": "논리 일관성",
    "clickbait_risk":      "어뷰징 위험도",
}

_PROMPT_TEMPLATE = """\
당신은 뉴스 신뢰도 분석 전문가입니다. 아래 뉴스 기사를 읽고 5가지 기준으로 평가하세요.

[CoT 평가 절차 — 반드시 이 순서를 따를 것]
각 기준을 평가할 때 다음 순서를 엄수하세요:
1. 기사에서 해당 기준과 관련된 근거(텍스트, 표현, 구조)를 먼저 찾는다.
2. 찾은 근거를 바탕으로 reason을 1~2문장으로 작성한다.
3. reason을 확인한 뒤 score를 결정한다.
절대로 score를 먼저 정하고 reason을 끼워 맞추지 마세요.
JSON 응답에서도 반드시 "reason" 키가 "score" 키보다 먼저 위치해야 합니다.

[평가 기준 및 채점 루브릭]

1. source_credibility — 출처 신뢰성
   10점: 메이저 공식 언론사(조선·중앙·한겨레 등), 정부/공공기관, 검증된 전문 매체
    5점: 중소 언론사 또는 출처가 부분적으로 확인됨
    0점: 출처 불분명한 블로그, 유언비어, 찌라시

2. evidence_support — 근거 지지도
   10점: 구체적 수치·통계, 실명 전문가 인용, 공식 자료 다수 포함
    5점: 일부 인용이나 근거가 있으나 불충분
    0점: 근거 없는 주관적 주장만 있음

3. style_neutrality — 문체 중립성  (점수 높을수록 중립적·객관적)
   10점: 건조하고 사실 중심의 객관적 문체, 감정 표현 없음
    5점: 일부 감정적 표현이 있으나 전반적으로 중립
    0점: 매우 감정적·선동적·편향된 표현이 지배적

4. logical_consistency — 논리 일관성
   10점: 전후 문맥이 완벽히 이어지고 주장이 일관됨
    5점: 일부 논리적 비약이 있으나 전체 흐름은 파악 가능
    0점: 스스로 모순되거나 논리적 비약이 심해 신뢰 불가

5. clickbait_risk — 어뷰징 위험도
   ⚠️ [역방향 척도] 이 항목은 점수가 높을수록 어뷰징 위험이 높고 나쁜 것입니다.
   "어뷰징 위험이 낮다" → 낮은 점수(0~3점)를 부여해야 합니다.
   "어뷰징 위험이 높다" → 높은 점수(7~10점)를 부여해야 합니다.
   절대 방향을 혼동하지 마세요.
   10점: 제목과 내용이 완전 불일치하거나 극도로 자극적인 어뷰징 기사
    5점: 제목이 다소 과장되었으나 내용과 연결은 됨
    0점: 제목이 본문 내용을 정직하고 정확하게 담고 있음

[출처]
{source}

[기사 본문]
{text}

반드시 아래 JSON 형식만 반환하세요. 설명 텍스트 없이 JSON만.
"reason"이 반드시 "score"보다 먼저 위치해야 합니다:
{{
  "source_credibility":  {{"reason": "<string>", "score": <0-10>}},
  "evidence_support":    {{"reason": "<string>", "score": <0-10>}},
  "style_neutrality":    {{"reason": "<string>", "score": <0-10>}},
  "logical_consistency": {{"reason": "<string>", "score": <0-10>}},
  "clickbait_risk":      {{"reason": "<string>", "score": <0-10>}},
  "overall_reason":      "<string>"
}}"""


def _parse_criterion(item: object) -> dict:
    """
    단일 기준 항목을 안전하게 파싱.
    - score: float string("7.5") 포함 처리, 0~10 범위 강제
    - reason: None/누락 시 빈 문자열 반환
    """
    if not isinstance(item, dict):
        return {"score": 0, "reason": ""}
    try:
        score = int(float(item.get("score", 0)))
        score = max(0, min(10, score))
    except (TypeError, ValueError):
        score = 0
    return {
        "score": score,
        "reason": str(item.get("reason", "") or ""),
    }


def _weighted_sum_score(criteria: dict) -> int:
    """가중 합산 규칙으로 0~100 점수 산출."""
    raw = sum(criteria[k]["score"] * w for k, w in WEIGHTS.items())
    # 최대값: (0.25+0.25+0.20+0.20) * 10 = 9.0 (clickbait=0일 때)
    return int(max(0, min(100, raw / 9.0 * 100)))


def _verdict(score: int) -> str:
    if score >= 70:
        return "likely_true"
    elif score >= 40:
        return "uncertain"
    return "likely_false"


def _fallback(error_msg: str) -> dict:
    return {
        "score": 0,
        "verdict": "uncertain",
        "reason": f"신뢰도 분석 실패: {error_msg}",
        "per_criteria": {k: {"score": 0, "reason": "분석 실패"} for k in WEIGHTS},
    }


def score_trust(text: str, source: str | None = None) -> dict:
    """
    TELLER 기반 신뢰도 분석 (CoT + Rubric).

    Parameters
    ----------
    text   : 기사 본문 전체
    source : 언론사명 (없으면 "미상")

    Returns
    -------
    {
        "score"       : int,   # 0~100
        "verdict"     : str,   # "likely_true" | "uncertain" | "likely_false"
        "reason"      : str,   # 종합 판단 근거
        "per_criteria": {
            "source_credibility":  {"score": int, "reason": str},
            "evidence_support":    {"score": int, "reason": str},
            "style_neutrality":    {"score": int, "reason": str},
            "logical_consistency": {"score": int, "reason": str},
            "clickbait_risk":      {"score": int, "reason": str},
        }
    }
    """
    source_str = source or "미상"
    prompt = _PROMPT_TEMPLATE.format(source=source_str, text=text[:3000])

    model = genai.GenerativeModel(
        TRUST_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )

    while True:
        try:
            response = model.generate_content(prompt)
            raw_json = json.loads(response.text)
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "Quota exceeded" in err:
                wait = random.randint(30, 60)
                print(f"⚠️ [trust] 할당량 초과! {wait}초 후 재시도...")
                time.sleep(wait)
                continue
            print(f"❌ [trust] Gemini 호출 실패: {e}")
            return _fallback(err)

    # per_criteria 추출 — _parse_criterion으로 타입·범위 안전 처리
    per_criteria = {
        key: _parse_criterion(raw_json.get(key))
        for key in WEIGHTS
    }

    score = _weighted_sum_score(per_criteria)

    return {
        "score":        score,
        "verdict":      _verdict(score),
        "reason":       str(raw_json.get("overall_reason", "") or ""),
        "per_criteria": per_criteria,
    }
