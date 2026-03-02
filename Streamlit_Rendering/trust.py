# Streamlit_Rendering/trust.py
# TD1 — TELLER 기반 신뢰도 분석 모듈 (Gemini Cognitive + Weighted Sum Rule)

import os
import json
import random
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
각 기준을 0~10점으로 채점하고, 판단 근거를 한국어로 간략히(1~2문장) 작성하세요.
마지막으로 전체적인 종합 판단(overall_reason)도 2~3문장으로 작성하세요.

[평가 기준]
- source_credibility : 출처 신뢰성 — 언론사 규모, 공신력, 전문성
- evidence_support   : 근거 지지도 — 수치, 인용, 전문가 등장 여부
- style_neutrality   : 문체 중립성 — 감정적·선동적 표현 여부 (낮을수록 선동적)
- logical_consistency: 논리 일관성 — 전후 문맥, 주장 일관성
- clickbait_risk     : 어뷰징 위험도 — 자극적 제목·내용 불일치 여부 (높을수록 어뷰징)

[출처]
{source}

[기사 본문]
{text}

반드시 아래 JSON 형식만 반환하세요. 설명 텍스트 없이 JSON만:
{{
  "source_credibility":  {{"score": <0-10>, "reason": "<string>"}},
  "evidence_support":    {{"score": <0-10>, "reason": "<string>"}},
  "style_neutrality":    {{"score": <0-10>, "reason": "<string>"}},
  "logical_consistency": {{"score": <0-10>, "reason": "<string>"}},
  "clickbait_risk":      {{"score": <0-10>, "reason": "<string>"}},
  "overall_reason":      "<string>"
}}"""


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
    TELLER 기반 신뢰도 분석.

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

    # per_criteria 추출 (5개 기준)
    per_criteria = {}
    for key in WEIGHTS:
        item = raw_json.get(key, {})
        per_criteria[key] = {
            "score": int(item.get("score", 0)),
            "reason": str(item.get("reason", "")),
        }

    score = _weighted_sum_score(per_criteria)

    return {
        "score":        score,
        "verdict":      _verdict(score),
        "reason":       str(raw_json.get("overall_reason", "")),
        "per_criteria": per_criteria,
    }
