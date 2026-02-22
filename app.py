
import streamlit as st
from Streamlit_Rendering import repo, functions as fn

st.set_page_config(layout="wide")
repo.init_db()

# 세션 초기화
for k, v in {"selected_article_id": None, "search_executed": False, 
             "search_query": "", "admin_mode": False}.items():
    if k not in st.session_state: st.session_state[k] = v

# 라우팅
if st.session_state["admin_mode"]:
    fn.render_admin_page()
elif st.session_state["selected_article_id"]:
    fn.render_detail_page(st.session_state["selected_article_id"])
elif st.session_state["search_executed"]:
    fn.render_search_results_page(st.session_state["search_query"])
else:
    fn.render_main_page()
