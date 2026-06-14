# backend/services/trust.py
# TD1 — TELLER 기반 신뢰도 분석 모듈 (Phase 1 + Phase 2 적용)

import json
import logging
import random
import re
import time

import google.generativeai as genai

try:
    from backend.services.config import get_gemini_api_key
except ModuleNotFoundError:
    from config import get_gemini_api_key

genai.configure(api_key=get_gemini_api_key())
logger = logging.getLogger(__name__)

# ============================================================
# 상수
# ============================================================
TRUST_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3

WEIGHTS = {
    "source_credibility":  0.20,  # W3: 패턴 경고 목적, style 강화 대비 소폭 하향
    "evidence_support":    0.25,
    "style_neutrality":    0.25,  # W3: 감정·선동 표현이 가장 명확한 패턴 신호
    "logical_consistency": 0.20,
    "clickbait_risk":     -0.10,
}

# Phase 1-5: MAX_RAW 동적 계산 (양수 가중치만 합산하여 최대 가능 원점수 산출)
_MAX_RAW = sum(w for w in WEIGHTS.values() if w > 0) * 10

CRITERIA_LABELS = {
    "source_credibility":  "출처 신뢰성",
    "evidence_support":    "근거 지지도",
    "style_neutrality":    "문체 중립성",
    "logical_consistency": "논리 일관성",
    "clickbait_risk":      "어뷰징 위험도",
}

# ============================================================
# Phase 1-1: SOURCE_TIER 룰 기반 사전 (LLM 평가 제거)
# ============================================================
SOURCE_TIER: dict[int, set[str]] = {
    10: {"연합뉴스", "KBS", "MBC", "SBS", "YTN"},
    8:  {"조선일보", "중앙일보", "동아일보", "한겨레", "경향신문",
         "한국일보", "국민일보", "서울신문", "세계일보", "문화일보"},
    6:  {"머니투데이", "뉴시스", "아시아경제", "뉴스1", "헤럴드경제",
         "이데일리", "파이낸셜뉴스", "아주경제", "데일리안", "매일경제"},
    3:  {"온라인 커뮤니티", "카카오스토리", "네이버 블로그", "티스토리",
         "인스타그램", "페이스북", "유튜브"},
}
SOURCE_DEFAULT_SCORE = 5  # 미상 또는 미등록 → 중립 처리


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


# ============================================================
# Phase 2-1: PREDICATES — Yes/No 질문 분해
# ============================================================
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
        "기사 앞부분의 주장과 뒷부분 결론이 서로 일치하는가? (앞에서 A라 했는데 결론에서 B라 하면 no로 답하라)",
        "기사 내에 서로 모순되는 문장이 없는가? (한 문단에서 긍정했다가 다른 문단에서 부정하면 no로 답하라)",
        "제목에서 제시한 주장이 본문 내용으로 실질적으로 뒷받침되는가?",
    ],
    "clickbait_risk": [
        "제목이 본문 내용을 과장하거나 왜곡하고 있는가?",       # 역방향: Yes → 위험
        "독자의 클릭을 유도하기 위한 자극적 표현이 제목에 있는가?",  # 역방향
        "기사 내용이 제목이 암시하는 것과 실질적으로 다른가?",    # 역방향
    ],
}

NEGATIVE_PREDICATES: set[str] = set()
# clickbait_risk는 정방향(yes=위험=고점)으로 처리.
# WEIGHTS["clickbait_risk"]=-0.10(음수)이 감산을 담당하므로
# is_negative=True로 점수를 뒤집으면 이중 반전이 됨.

# Phase 2-1: Yes/No 프롬프트 템플릿
_PREDICATE_QUESTIONS = "\n".join(
    f"\n{key}:\n" + "\n".join(f"  Q{i+1}: {q}" for i, q in enumerate(qs))
    for key, qs in PREDICATES.items()
)

_PROMPT_TEMPLATE = f"""\
당신은 뉴스 신뢰도 분석 전문가입니다.

[평가 방식]
각 질문에 반드시 "yes", "no", "uncertain" 중 하나만 답하세요.
이유(reason)를 먼저 한국어 3문장 이내로 간결하게 쓴 뒤 answer를 결정하세요. reason은 반드시 3문장을 넘지 않습니다.

[질문 목록]
{_PREDICATE_QUESTIONS}

[overall_reason 작성 규칙]
- 위 평가 기준 중 "no" 답변이 가장 많은 기준을 중심으로 서술
- 신뢰도를 낮추는 핵심 요인 1~2개를 구체적으로 지목
- 한국어 2문장 이내
- 예시: "근거 지지도가 낮습니다. 전문가 인용이나 통계 수치 없이 주관적 주장으로만 구성되어 있습니다."

[출처]
{{source}}

[기사 제목]
{{title}}

[기사 본문]
{{text}}

[반환 형식 — JSON만, 설명 텍스트 없이]
{{{{
  "evidence_support": [
    {{{{"reason": "...", "answer": "yes"}}}},
    {{{{"reason": "...", "answer": "no"}}}},
    {{{{"reason": "...", "answer": "uncertain"}}}}
  ],
  "style_neutrality": [
    {{{{"reason": "...", "answer": "yes"}}}},
    {{{{"reason": "...", "answer": "no"}}}},
    {{{{"reason": "...", "answer": "uncertain"}}}}
  ],
  "logical_consistency": [
    {{{{"reason": "...", "answer": "yes"}}}},
    {{{{"reason": "...", "answer": "no"}}}},
    {{{{"reason": "...", "answer": "uncertain"}}}}
  ],
  "clickbait_risk": [
    {{{{"reason": "...", "answer": "yes"}}}},
    {{{{"reason": "...", "answer": "no"}}}},
    {{{{"reason": "...", "answer": "uncertain"}}}}
  ],
  "overall_reason": "신뢰도를 낮추는 핵심 요인 1~2개를 한국어 2문장으로 서술"
}}}}"""


# ============================================================
# Phase 1-2: 텍스트 슬라이싱 (앞 2000 + 뒤 1000)
# ============================================================
def _trim_text(text: str, front: int = 2500, tail: int = 1500) -> str:
    """앞 + 뒤 슬라이싱으로 결론부 보존. (B-4: 4000자로 확대, 중간 누락 범위 축소)"""
    if len(text) <= front + tail:
        return text
    return text[:front] + "\n\n...(중략)...\n\n" + text[-tail:]


# ============================================================
# Phase 1-3: 3단계 폴백 파싱
# ============================================================
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


def _parse_criterion(item: object) -> dict:
    """
    단일 기준 항목을 안전하게 파싱.
    파싱 실패 시 _parse_failed 플래그 추가.
    """
    if not isinstance(item, dict):
        return {"score": 0, "reason": "", "_parse_failed": True}
    try:
        score = int(float(item.get("score", 0)))
        score = max(0, min(10, score))
    except (TypeError, ValueError):
        score = 0
    return {
        "score": score,
        "reason": str(item.get("reason", "") or ""),
    }


# ============================================================
# Phase 2-1: Yes/No → 점수 변환
# ============================================================
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
        else:  # uncertain — 명확한 판단을 피하는 경향 보정, 0.5보다 약간 부정적으로 처리
            values.append(0.3)

    ratio = sum(values) / len(values)

    if is_negative:
        score = int((1 - ratio) * 10)  # 역방향: Yes 많을수록 낮은 점수
    else:
        score = int(ratio * 10)

    reasons = [a.get("reason", "") for a in answers if a.get("reason")]
    return {"score": max(0, min(10, score)), "reason": " / ".join(reasons[:2])}


# ============================================================
# Phase 1-5 & 2-2: 동적 정규화 및 교차 패널티
# ============================================================
def _weighted_sum_score(criteria: dict) -> int:
    """가중 합산 규칙으로 0~100 점수 산출 (교차 패널티 포함)."""
    raw = sum(criteria[k]["score"] * w for k, w in WEIGHTS.items())
    score_100 = (raw / _MAX_RAW) * 100
    
    # Phase 2-2: 교차 패널티 (감정 과잉 + 근거 부족)
    neutrality = criteria["style_neutrality"]["score"]
    evidence   = criteria["evidence_support"]["score"]
    if neutrality < 4 and evidence < 4:
        logger.debug("[trust] 교차 패널티 적용: neutrality=%d, evidence=%d", neutrality, evidence)
        score_100 -= 10.0

    return int(max(0, min(100, score_100)))


def _verdict(score: int) -> str:
    if score >= 70:
        return "likely_true"
    elif score >= 40:
        return "uncertain"
    return "likely_false"


def _fallback(error_msg: str) -> dict:
    return {
        "score": None,
        "verdict": "unanalyzed",
        "reason": f"신뢰도 분석 실패: {error_msg}",
        "per_criteria": {k: {"score": None, "reason": "분석 실패"} for k in WEIGHTS},
        "flags": {
            "short_article": False,
        },
    }


# ============================================================
# 메인 함수
# ============================================================
def score_trust(text: str, source: str | None = None, title: str | None = None) -> dict:
    """
    TELLER 기반 신뢰도 분석 (Phase 1 + Phase 2).

    Parameters
    ----------
    text   : 기사 본문 전체
    source : 언론사명 (없으면 "미상")
    title  : 기사 제목 (없으면 "제목 없음")

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
    title_str  = title  or "제목 없음"

    # Phase 1-2: 앞 2000 + 뒤 1000 슬라이싱
    trimmed = _trim_text(text)

    # 텍스트 내 중괄호 때문에 .format()에서 KeyError가 발생하지 않도록 이스케이프 처리
    safe_source = source_str.replace("{", "{{").replace("}", "}}")
    safe_title  = title_str.replace("{",  "{{").replace("}", "}}")
    safe_text   = trimmed.replace("{",   "{{").replace("}", "}}")

    prompt = _PROMPT_TEMPLATE.format(source=safe_source, title=safe_title, text=safe_text)

    model = genai.GenerativeModel(
        TRUST_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )

    raw_json = None

    # Phase 1-4: MAX_RETRIES 상한 + logging
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
                logger.warning("[trust] 할당량 초과, %d초 후 재시도 (%d/%d)", wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
            else:
                logger.error("[trust] Gemini 호출 실패: %s", e)
                return _fallback(err)

    if raw_json is None:
        return _fallback("재시도 소진 후 파싱 실패")

    # Phase 1-1: source_credibility는 룰 기반으로 직접 채움
    per_criteria: dict = {
        "source_credibility": _rule_source_score(source_str),
    }

    # Phase 2-1: 나머지 4개 기준은 Yes/No predicate로 변환
    for key in ("evidence_support", "style_neutrality", "logical_consistency", "clickbait_risk"):
        answers = raw_json.get(key, [])
        if isinstance(answers, list):
            per_criteria[key] = _predicate_to_score(
                answers, is_negative=(key in NEGATIVE_PREDICATES)
            )
        else:
            # 예상치 못한 형식 → 안전 파싱 폴백
            per_criteria[key] = _parse_criterion(answers)

    score = _weighted_sum_score(per_criteria)

    return {
        "score":        score,
        "verdict":      _verdict(score),
        "reason":       str(raw_json.get("overall_reason", "") or ""),
        "per_criteria": per_criteria,
        "flags": {
            "short_article": len(text) < 300,  # 단신 속보 — 근거 지지도 구조적 불리
        },
    }
