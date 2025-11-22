# 📜 app.py (v0.4)
# 이 파일은 오직 '페이지 라우터' 역할만 수행합니다.
# 모든 로직과 UI 렌더링은 functions.py에 위임합니다.

import streamlit as st
import os
# functions.py에서 렌더링 함수 및 콜백 함수를 가져옵니다.
# (실제 콜백은 functions.py 내부의 버튼들이 직접 호출)
import Streamlit_Rendering.functions as fn 

try:
    from google import genai
except ImportError:
    print("❌ 'google-genai' 라이브러리를 찾을 수 없습니다. (pip install google-genai)")
    genai = None
    
# ...existing code...
if genai:
    try:
        # (참고) Streamlit 배포 시에는 st.secrets["API_KEY"] 사용을 권장합니다.
        
        # 우선순위: st.secrets -> 환경변수
        if "API_KEY" in st.secrets:
            api_key = st.secrets["API_KEY"]
        elif "API_KEY" in st.secrets:  # 기존 secrets.toml에 API_KEY만 있던 경우 호환
            api_key = st.secrets["API_KEY"]
        else:
            api_key = os.getenv("API_KEY", None)
        
        if not api_key:
            print("⚠️ 경고: Gemini API 키가 설정되지 않았습니다. 'get_summary' 함수가 작동하지 않을 수 있습니다.")
            print("         로컬 테스트 시 Streamlit secrets 또는 API_KEY 환경 변수를 설정하세요.")
        else:
            client = genai.Client(api_key=api_key)
            print("✅ Gemini 클라이언트가 성공적으로 초기화되었습니다.")

    except Exception as e:
        print(f"❌ Gemini 클라이언트 초기화 오류: {e}")
else:
    print("❌ 'google-genai' 라이브러리가 임포트되지 않아 클라이언트를 초기화할 수 없습니다.")

MODEL_NAME = "gemini-2.5-flash" # 또는 "gemini-1.5-flash-latest"

# --- app.py에서 functions 모듈로 Gemini 클라이언트/타입/모델명을 노출 ---
# fn 모듈에 값 할당 (genai가 없거나 client가 생성되지 않으면 None이 들어감)
try:
    fn.client = client  # client 변수는 genai 블록에서 정의되었을 수 있습니다.
except NameError:
    fn.client = None
fn.MODEL_NAME = MODEL_NAME

# --- 페이지 상태 관리 (4-state logic) ---
# (앱 실행 시 가장 먼저 상태 변수들을 초기화합니다)

if 'admin_mode' not in st.session_state:
    st.session_state['admin_mode'] = False
if 'selected_article_id' not in st.session_state:
    st.session_state['selected_article_id'] = None
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
if 'search_executed' not in st.session_state:
    st.session_state['search_executed'] = False

# --- Streamlit UI 렌더링 (라우터) ---
st.set_page_config(layout="wide")

# (콜백 함수 정의는 모두 functions.py로 이동했습니다)

# --- UI 렌더링 로직 (v0.4 '라우터') ---
# st.session_state의 '깃발'을 확인하고
# functions.py에 정의된 4개의 '렌더링 함수' 중 하나를 호출합니다.

if st.session_state['admin_mode']:
    # 1. 관리자 페이지 렌더링
    fn.render_admin_page() 

elif st.session_state['selected_article_id'] is not None:
    # 2. 상세 페이지 렌더링
    fn.render_detail_page(st.session_state['selected_article_id'])

elif st.session_state['search_executed']:
    # 3. 검색 결과 페이지 렌더링
    fn.render_search_results_page(st.session_state['search_query'])

else:
    # 4. 메인 페이지 렌더링
    fn.render_main_page()