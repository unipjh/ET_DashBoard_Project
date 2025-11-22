# ⚙️ functions.py (v0.2)
# 이 파일은 데이터, 상태 변경(콜백), UI 렌더링을 모두 담당합니다.

import streamlit as st
import numpy as np  # (더미 그래프용)
import json
from .data import MOCK_DB  # (같은 폴더의 data.py에서 MOCK_DB 가져오기)

#========================#
# --- 기본 데이터 로직 ---#
#========================#

def get_all_articles() -> list:
    """[To. DE] DB에 있는 모든 기사 리스트를 반환합니다."""
    return list(MOCK_DB.values())

def get_article_by_id(article_id: str) -> dict:
    """[To. DE] 특정 ID의 기사 상세 정보를 반환합니다."""
    return MOCK_DB.get(article_id)

# Gemini 클라이언트 / 모델 이름은 app.py에서 할당
client = None
MODEL_NAME = None


#========================#
# --- Gemini 요약 함수 ---#
#========================#

def summarize_article_with_gemini(article_text: str) -> str:
    """
    Gemini API를 사용하여 뉴스 기사 텍스트를 요약하는 함수.

    :param article_text: 요약할 뉴스 기사 본문 (str)
    :return: 요약된 텍스트 (str) 또는 오류 메시지
    """
    if not client or not MODEL_NAME:
        print("❌ 'summarize_article_with_gemini' 호출 실패: Gemini 클라이언트가 초기화되지 않았습니다.")
        return "요약 오류: Gemini 클라이언트가 초기화되지 않았습니다. API 키를 확인하세요."

    if not article_text or len(article_text.strip()) < 10:
        return "본문 내용이 너무 짧거나 비어 있습니다."

    # 시스템 역할 + 요구사항을 프롬프트 상단에 붙이는 방식으로 처리
    system_instruction = (
        "당신은 뉴스 요약 어시스턴트입니다. "
        "주어진 텍스트의 핵심 내용만 파악하여 2~3개의 객관적이고 완결된 문장으로 요약하세요. "
        "요약문에는 '필자는', '기자는', '저자는'과 같은 표현이나, "
        "텍스트 내용과 직접 관련 없는 서론/결론은 포함하지 마세요. "
        "모든 문장은 자연스럽게 끝나고 논리적으로 연결되도록 작성하십시오."
    )

    user_prompt = f"다음 뉴스 기사를 한국어로 요약해 주세요:\n\n{article_text}"

    # 최종 프롬프트 문자열
    prompt = system_instruction + "\n\n[기사]\n" + user_prompt

    try:
        config = {
            "max_output_tokens": 2048,
            "temperature": 0.3,
        }

        # google-genai 구버전: config= 만 지원
        response = client.models.generate_content(
            model=f"models/{MODEL_NAME}",
            contents=prompt,
            config=config,
        )

        if response.text is not None:
            return response.text.strip()

        # response.text가 None인 경우
        reason = "알 수 없는 이유로 모델이 텍스트를 생성하지 못했습니다."
        try:
            if response.candidates and response.candidates[0].finish_reason:
                reason = f"모델 종료 이유: {response.candidates[0].finish_reason.name}"
        except Exception:
            pass

        snippet = article_text[:50].replace("\n", " ")
        print(f"⚠️ 요약 실패 (기사 일부: {snippet}...): {reason}")
        return f"요약 실패: {reason}"

    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        # 구체적인 오류 메시지 분리
        msg = str(e)
        if "API_KEY" in msg.upper():
            return "요약 오류: API 키가 유효하지 않거나 설정되지 않았습니다."
        if "DeadlineExceeded" in msg:
            return "요약 오류: 요청 시간이 초과되었습니다. 나중에 다시 시도해 주세요."
        return "요약 오류: 예상치 못한 오류가 발생했습니다. 다시 시도해 주세요."


def get_summary(article_text: str) -> str:
    """[To. M1 (요약)] 기사 본문을 받아 요약문을 반환합니다."""
    snippet = article_text[:30].replace("\n", " ")
    print(f"요약 요청 시작 (본문 일부: {snippet}...)")
    with st.spinner("요약 중... 잠시만 기다려 주세요."):
        summary = summarize_article_with_gemini(article_text)
    snippet_sum = summary[:30].replace("\n", " ")
    print(f"요약 완료 (요약문 일부: {snippet_sum}...)")
    return summary


#==========================#
# --- 신뢰도 판별 파트 ---#
#==========================#

def get_trust_score(article_text: str, source: str) -> dict:
    """
    [To. PM (신뢰)] 기사 본문과 출처를 받아 신뢰도 점수와 근거를 반환합니다.
    v0.2: TELLER 프레임워크 아이디어를 간단히 차용한
          LLM + 규칙 기반 최소 구현 (ML 미사용)
    """
    # 0. 기본 방어 로직: 클라이언트 없음 or 본문 너무 짧음

    def _simple_source_heuristic(src: str) -> dict:
        base = 60
        if not src:
            base -= 10
        else:
            high_trust_keywords = ["연합뉴스", "KBS", "MBC", "SBS", "YTN", "BBC", "Reuters", "로이터", "AP통신"]
            low_trust_keywords = ["블로그", "카페", "커뮤니티", "유튜브", "SNS", "카더라"]
            if any(k in src for k in high_trust_keywords):
                base += 20
            if any(k in src for k in low_trust_keywords):
                base -= 20
        base = max(0, min(100, base))
        return {
            "score": base,
            "verdict": "uncertain",
            "reason": "Gemini 없이 간단한 출처 휴리스틱으로 추정한 신뢰도입니다.",
            "framework_version": "TELLER-v0.2-heuristic"
        }

    def _parse_json_from_text(text: str) -> dict:
        """```json 코드블럭이나 앞뒤 설명이 섞여 있어도 JSON 부분만 뽑아 파싱"""
        if not text:
            raise ValueError("빈 응답")

        stripped = text.strip()
        # ```json ... ``` 제거 시도
        if stripped.startswith("```"):
            # 코드블럭 헤더/푸터 제거 (아주 러프하게)
            stripped = stripped.strip("`")
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"JSON 구간을 찾지 못함: {stripped[:100]}")
        json_str = stripped[start:end+1]
        return json.loads(json_str)

    if client is None or MODEL_NAME is None:
        print("❌ 'get_trust_score' 호출 실패: Gemini 클라이언트가 초기화되지 않았습니다.")
        return _simple_source_heuristic(source)

    if not article_text or len(article_text.strip()) < 30:
        return {
            "score": 50,
            "verdict": "uncertain",
            "reason": "본문이 너무 짧아 신뢰도 평가를 수행하기 어렵습니다.",
            "framework_version": "TELLER-v0.2-minimal"
        }

    # 1. TELLER-lite 1차 프롬프트
    system_instruction = (
        "당신은 TELLER 프레임워크를 응용한 뉴스 신뢰도 평가자입니다. "
        "기사의 출처, 증거 유무, 표현 방식, 논리 일관성, 클릭베이트 위험도를 기준으로 "
        "각 항목을 0~2점(0=매우 의심, 1=애매, 2=신뢰 가능)으로 채점하고, 각 기준에 대해 한 문장 근거를 작성하십시오. "
        "또한 전체적인 진실성 판단(verdict)과 그 근거(overall_reason)를 제공합니다. "
        "반드시 유효한 JSON만 출력해야 하며, JSON 이외의 텍스트는 절대 포함하지 마십시오."
    )

    user_prompt = f"""
다음은 하나의 뉴스 기사와 그 출처입니다.

[출처]
{source or "미상"}

[기사 본문]
{article_text}

아래 5개 기준에 대해 각각 0~2점의 정수 점수와 한 문장 근거를 평가해 주세요.
점수: 0=매우 의심스럽다, 1=애매하다, 2=신뢰 가능하다.

1) source_credibility: 출처의 일반적인 신뢰도는 어느 정도인가?
2) evidence_support: 기사 내용이 구체적인 사실, 수치, 인용 등 검증 가능한 증거에 얼마나 기반하는가?
3) style_neutrality: 기사 표현이 과도하게 선정적이거나 감정적이지 않고, 중립적인가?
4) logical_consistency: 기사 내부에 논리적 모순이나 자기모순이 없이 일관적인가?
5) clickbait_risk: 제목/내용이 클릭을 유도하는 과장·선정적 표현에 크게 의존하지 않는가?

반드시 아래 JSON 형식으로만, 공백과 줄바꿈은 자유롭게 사용하되 유효한 JSON이 되도록 응답하세요.

{{
  "per_criteria": {{
    "source_credibility": {{"score": 0, "reason": "..." }},
    "evidence_support":   {{"score": 0, "reason": "..." }},
    "style_neutrality":   {{"score": 0, "reason": "..." }},
    "logical_consistency":{{"score": 0, "reason": "..." }},
    "clickbait_risk":     {{"score": 0, "reason": "..." }}
  }},
  "verdict": "likely_true 또는 likely_false 또는 uncertain 중 하나",
  "overall_reason": "전체적인 판단 근거를 한국어 한 문단으로 서술"
}}
"""

    full_prompt = system_instruction + "\n\n[평가 대상 기사]\n" + user_prompt

    def _call_llm(prompt: str) -> str:
        """LLM 호출: JSON 모드 강제 + 토큰 증량 + 안전 설정 해제"""
        try:
            # 1. 안전 설정 (유지)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            # 2. 설정 강화 (JSON 모드 + 토큰 8배 증량)
            config = {
                "max_output_tokens": 8192,   # 1024 -> 8192 (짤림 방지)
                "temperature": 0.1,
                "response_mime_type": "application/json", # <--- 핵심: 무조건 JSON으로만 뱉음
                "safety_settings": safety_settings,
            }

            # 3. 호출
            response = client.models.generate_content(
                model=f"models/{MODEL_NAME}",
                contents=prompt,
                config=config,
            )

            # 4. 텍스트 추출
            # JSON 모드를 쓰면 response.text가 깔끔하게 떨어질 확률이 99.9%입니다.
            if hasattr(response, "text"):
                return response.text
            
            # 혹시라도 text 속성이 안 잡힐 경우를 대비한 백업
            if hasattr(response, "candidates") and response.candidates:
                return response.candidates[0].content.parts[0].text

            return ""

        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            # 에러 발생 시 빈 JSON 문자열 리턴하여 파싱 에러 방지
            return "{}"
              
    try:
        with st.spinner("신뢰도 평가 중... 잠시만 기다려 주세요."):
            raw_text = _call_llm(full_prompt)

        print("===== RAW TRUST RESPONSE BEGIN =====")
        print(raw_text)
        print("===== RAW TRUST RESPONSE END =====")

        # 1차 호출에서 빈 응답이면 → 단순 백업 프롬프트로 한 번 더 시도
        if not raw_text:
            print("⚠️ 1차 신뢰도 프롬프트에서 빈 응답. 단순 프롬프트로 재시도합니다.")
            simple_prompt = f"""
당신은 뉴스의 진실성을 평가하는 한국어 평가자입니다.

다음 뉴스 기사와 출처를 보고, 전체적인 신뢰도를 0~100 사이 정수(score)로 평가하고,
진실성 판단(verdict)과 한 문장 이유(reason)를 JSON으로만 출력하세요.

형식:
{{
  "score": 0,
  "verdict": "likely_true 또는 likely_false 또는 uncertain 중 하나",
  "reason": "..."
}}

[출처]
{source or "미상"}

[기사 본문]
{article_text}
"""
            raw_text = _call_llm(simple_prompt)
            print("===== RAW SIMPLE TRUST RESPONSE BEGIN =====")
            print(raw_text)
            print("===== RAW SIMPLE TRUST RESPONSE END =====")

            if not raw_text:
                raise ValueError("2차 단순 프롬프트에서도 빈 응답")

            # simple 모드 파싱
            data = _parse_json_from_text(raw_text)
            score = int(data.get("score", 50))
            verdict = data.get("verdict", "uncertain")
            reason = data.get("reason", "LLM의 단순 평가 결과입니다.")
            score = max(0, min(100, score))
            if not isinstance(verdict, str):
                verdict = "uncertain"
            if not isinstance(reason, str) or len(reason.strip()) < 3:
                reason = "LLM의 단순 평가 결과입니다."
            return {
                "score": score,
                "verdict": verdict,
                "reason": reason,
                "per_criteria": {},
                "framework_version": "TELLER-v0.2-simple"
            }

        # 1차(TELLER-lite) 응답 파싱
        try:
            data = json.loads(raw_text)
        except Exception:
            data = _parse_json_from_text(raw_text)

        per_criteria = data.get("per_criteria", {}) or {}

        weights = {
            "source_credibility": 1.5,
            "evidence_support": 1.5,
            "style_neutrality": 1.0,
            "logical_consistency": 1.0,
            "clickbait_risk": 1.0,
        }
        total_weight = sum(weights.values())
        weighted_sum = 0.0

        for key, w in weights.items():
            score_val = per_criteria.get(key, {}).get("score", 1)
            try:
                score_float = float(score_val)
            except Exception:
                score_float = 1.0
            score_float = max(0.0, min(2.0, score_float))
            weighted_sum += (score_float / 2.0) * 100.0 * w

        overall_score = int(round(weighted_sum / total_weight))

        verdict = data.get("verdict", "uncertain")
        if not isinstance(verdict, str):
            verdict = "uncertain"

        overall_reason = data.get("overall_reason")
        if not isinstance(overall_reason, str) or len(overall_reason.strip()) < 5:
            label_map = {
                "source_credibility": "출처 신뢰도",
                "evidence_support": "증거 기반성",
                "style_neutrality": "표현 중립성",
                "logical_consistency": "논리 일관성",
                "clickbait_risk": "클릭베이트 위험도",
            }
            reason_list = []
            for k, label in label_map.items():
                r = per_criteria.get(k, {}).get("reason")
                if isinstance(r, str) and r.strip():
                    reason_list.append(f"{label}: {r.strip()}")
            overall_reason = " / ".join(reason_list) if reason_list else "LLM의 기준별 평가를 종합하여 산출한 신뢰도 점수입니다."

        return {
            "score": overall_score,
            "verdict": verdict,
            "reason": overall_reason,
            "per_criteria": per_criteria,
            "framework_version": "TELLER-v0.2-rule"
        }

    except Exception as e:
        print(f"신뢰도 평가 중 오류 발생: {e}")
        fallback = _simple_source_heuristic(source)
        fallback["framework_version"] = "TELLER-v0.2-heuristic-fallback"
        return fallback


#==========================#
# --- 추천 (더미 구현) ---#
#==========================#

def get_recommendations(user_id: str = "default_user") -> list:
    """[To. M2 (추천)] 사용자 ID 기반으로 추천 기사 리스트를 반환합니다."""
    return [MOCK_DB['news_002'], MOCK_DB['news_004'], MOCK_DB['news_001']]


#===========================#
# --- 상태 변경 콜백 함수 ---#
#===========================#

def select_article(article_id: str):
    """(콜백) 상세 페이지로 이동"""
    st.session_state['selected_article_id'] = article_id
    st.session_state['admin_mode'] = False 

def show_main_page():
    """(콜백) 메인 페이지로 이동 (모든 상태 초기화)"""
    st.session_state['selected_article_id'] = None
    st.session_state['search_executed'] = False
    st.session_state['search_query'] = ""
    st.session_state['admin_mode'] = False

def execute_search():
    """(콜백) 검색 결과 페이지로 이동"""
    st.session_state['search_executed'] = True
    st.session_state['selected_article_id'] = None
    st.session_state['admin_mode'] = False

def go_to_admin(): 
    """(콜백) 관리자 모드로 전환"""
    st.session_state['admin_mode'] = True
    st.session_state['selected_article_id'] = None 
    st.session_state['search_executed'] = False


#=========================#
# --- UI 렌더링 함수들 ---#
#=========================#

def render_admin_page():
    """관리자 페이지 UI를 렌더링합니다."""
    st.title("🛠️ 관리자 대시보드 (Admin Mode)")
    st.button("‹ ‹ 사용자 모드로 돌아가기", on_click=show_main_page)
    st.markdown("---")

    col1, col2 = st.columns([2, 1]) 
    with col1:
        st.subheader("일간 활성 사용자(DAU) 로그 (Dummy Graph)")
        chart_data = np.random.randn(30, 3)
        st.line_chart(chart_data)
        st.info("이곳에 ClickHouse에서 집계한 사용자 로그 그래프가 표시됩니다.")

    with col2:
        st.subheader("관리자 기능 (Dummy)")
        if st.button("⚙️ 일일 기사 자동 크롤링 실행 (Dummy)"):
            with st.spinner("크롤링을 시작합니다... (가짜)"):
                import time
                time.sleep(2)
                st.success("오늘의 기사 100개가 크롤링되었습니다! (가짜)")
        
        st.divider() 
        st.subheader("주요 지표 (Dummy)")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("총 기사 수 (DB)", "5", "+5 (오늘)")
        with m_col2:
            st.metric("총 사용자 수", "1", "+0 (오늘)")


def render_detail_page(article_id: str):
    """상세 기사 페이지 UI를 렌더링합니다. (UI/UX 개선판)"""
    article = get_article_by_id(article_id)
    
    if article:
        # --- 메인 컨텐츠 영역 ---
        st.button("‹ 메인으로 돌아가기", on_click=show_main_page)
        st.header(article['title'])
        
        # 메타 정보 배지 형태로 표시
        c1, c2 = st.columns([0.8, 0.2])
        with c1:
            st.caption(f"📰 출처: **{article['source']}** | 🔗 원문: {article['url']}")
        
        st.markdown("---")
        st.subheader("기사 본문")
        # 가독성을 위해 줄바꿈 처리 강화
        st.markdown(article['full_text'].replace("\n", "  \n\n")) 
        st.markdown("---")

        # --- 사이드바 AI 분석 영역 (대폭 개선) ---
        trust_info = get_trust_score(article['full_text'], article['source'])
        summary = get_summary(article['full_text'])
        
        st.sidebar.title("🤖 AI Insight")
        
        # 1. 신뢰도 메인 스코어 (게이지 스타일)
        st.sidebar.subheader("🛡️ 신뢰도 분석 (TELLER)")
        score = trust_info.get('score', 0)
        
        # 점수에 따른 색상/상태 결정
        if score >= 70:
            score_color = "green"
            score_delta = "안전 (Safe)"
        elif score >= 40:
            score_color = "off" # 회색/노란색 계열
            score_delta = "주의 (Caution)"
        else:
            score_color = "inverse" # 빨간색 계열
            score_delta = "위험 (Danger)"

        st.sidebar.metric(
            label="종합 신뢰도 점수",
            value=f"{score}점",
            delta=score_delta,
            delta_color=score_color
        )
        
        # 점수 바 (시각적 체감)
        st.sidebar.progress(score / 100)

        st.sidebar.markdown("---")

        # 2. 5대 평가 기준 상세 (아코디언 스타일)
        st.sidebar.subheader("📊 상세 평가 리포트")
        
        per_criteria = trust_info.get('per_criteria', {})
        
        # 매핑: (키 -> 한글 명칭)
        criteria_map = {
            "source_credibility": "📰 출처 신뢰도",
            "evidence_support": "🔍 증거 기반성",
            "style_neutrality": "⚖️ 표현 중립성",
            "logical_consistency": "🧠 논리 일관성",
            "clickbait_risk": "🎣 낚시성(Clickbait) 없음" 
            # 주의: 점수가 높을수록 낚시성이 '없다(좋다)'는 뜻으로 해석
        }

        if not per_criteria:
            st.sidebar.warning("상세 평가 데이터가 없습니다.")
        else:
            for key, label_kr in criteria_map.items():
                item = per_criteria.get(key, {})
                # 방어 로직: item이 dict가 아닐 경우 대비
                if not isinstance(item, dict):
                    s_score = 0
                    s_reason = "데이터 오류"
                else:
                    s_score = item.get('score', 0)
                    s_reason = item.get('reason', '근거 없음')

                # 0~2점을 아이콘으로 변환
                icon_map = {2: "🟢 (우수)", 1: "🟡 (보통)", 0: "🔴 (미흡/의심)"}
                status_text = icon_map.get(s_score, "⚪ (미상)")
                
                # Expander로 깔끔하게 접기/펼치기
                with st.sidebar.expander(f"{label_kr}: {status_text}"):
                    st.markdown(f"**평가 근거:**\n\n{s_reason}")

        st.sidebar.markdown("---")

        # 3. 종합 코멘트 (가독성 개선)
        st.sidebar.subheader("📝 종합 코멘트")
        overall_reason = trust_info.get('reason', '')
        # 문장 단위로 줄바꿈하여 가독성 확보
        formatted_reason = overall_reason.replace(". ", ".\n\n")
        
        if score >= 70:
            st.sidebar.success(formatted_reason)
        elif score >= 40:
            st.sidebar.warning(formatted_reason)
        else:
            st.sidebar.error(formatted_reason)

        st.sidebar.divider() 
        st.sidebar.subheader("📑 3줄 요약")
        st.sidebar.info(summary)

    else:
        st.error("오류: 기사를 찾을 수 없습니다.")
        st.button("‹ 메인으로 돌아가기", on_click=show_main_page)


def render_search_results_page(query: str):
    """검색 결과 페이지 UI를 렌더링합니다."""
    st.subheader(f"'{query}' 검색 결과 (v0.1 더미)")
    st.markdown("v0.1에서는 검색어와 상관없이 **모든 기사(DB 전체)**를 표시합니다.")
    
    all_articles = get_all_articles() 
    st.markdown(f"총 {len(all_articles)}개의 (가짜) 검색 결과가 있습니다.")
    
    for article in all_articles:
        st.button(
            f"[{article['source']}] {article['title']}", 
            on_click=select_article, 
            args=(article['id'],), 
            key=article['id'],
            use_container_width=True
        )
    
    st.markdown("---")
    st.button("‹ 메인으로 돌아가기", on_click=show_main_page)


def render_main_page():
    """메인(랜딩) 페이지 UI를 렌더링합니다."""
    
    title_col, button_col = st.columns([0.8, 0.2])
    with title_col:
        st.title("🤖 AI 기반 뉴스 검색 플랫폼")
    with button_col:
        st.button(
            "🛠️ Manage Mode", 
            on_click=go_to_admin,
            use_container_width=True
        )
    st.markdown("---") 

    with st.form(key="search_form"):
        st.session_state.search_query = st.text_input(
            "무엇이 궁금하신가요?", 
            value=st.session_state.search_query,
            placeholder="v0.1에서는 어떤 검색어를 입력해도 모든 기사가 나옵니다."
        )
        submitted = st.form_submit_button(
            label="🔍 검색", 
            on_click=execute_search,
            use_container_width=True
        )
    st.markdown("---")
    
    st.subheader("당신을 위한 최신 추천 뉴스 (Dummy 추천)")
    recommended_articles = get_recommendations(user_id="dummy_user")
    
    for article in recommended_articles:
        st.button(
            f"[{article['source']}] {article['title']}", 
            on_click=select_article,
            args=(article['id'],), 
            key=article['id'],
            use_container_width=True
        )
