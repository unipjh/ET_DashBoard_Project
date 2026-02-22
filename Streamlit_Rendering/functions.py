import streamlit as st
import pandas as pd
from Streamlit_Rendering import repo
from Streamlit_Rendering import admin_pipeline as ap
from Streamlit_Rendering.crawl import fetch_articles_from_naver
import re

def select_article(aid):
    st.session_state["selected_article_id"] = aid
    st.session_state["search_executed"] = False
    st.session_state["admin_mode"] = False

def show_main_page():
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False
    st.session_state["admin_mode"] = False

def go_to_admin():
    st.session_state["admin_mode"] = True
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False

def highlight_important_sentences(text: str, summary: str) -> str:
    """요약에 포함된 문장들을 본문에서 형광펜으로 하이라이트"""
    highlighted_text = text
    
    summary_sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 10]
    
    for sent in summary_sentences:
        key_phrase = sent[:min(20, len(sent))]
        pattern = re.compile(re.escape(key_phrase), re.IGNORECASE)
        highlighted_text = pattern.sub(f"🔍 **{key_phrase}**", highlighted_text)
    
    return highlighted_text

def render_main_page():
    title_col, button_col = st.columns([0.8, 0.2])
    with title_col:
        st.title("🚀 AI 기반 뉴스 검색 엔진")
    with button_col:
        st.button("⚙️ Admin", on_click=go_to_admin, use_container_width=True)

    st.markdown("---")
    query = st.text_input("궁금한 키워드를 입력하세요", placeholder="예: 날씨, 삼성 반도체, AI")
    if st.button("🔍 AI 분석 검색", use_container_width=True, type="primary"):
        if query.strip():
            st.session_state["search_query"] = query
            st.session_state["search_executed"] = True
    
    st.subheader("📬 최신 뉴스 리스트")
    df = repo.load_articles()
    if df.empty:
        st.info("DB가 비어있습니다. Admin에서 데이터를 적재하세요.")
    else:
        st.caption(f"총 {len(df)}개의 기사")
        
        # 페이지네이션 설정
        articles_per_page = 10
        total_pages = (len(df) + articles_per_page - 1) // articles_per_page
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        
        # 현재 페이지 기사 표시
        start_idx = (st.session_state.current_page - 1) * articles_per_page
        end_idx = start_idx + articles_per_page
        page_articles = df.iloc[start_idx:end_idx]
        
        for _, r in page_articles.iterrows():
            st.button(f"[{r['source']}] {r['title']}", on_click=select_article, 
                      args=(r['article_id'],), key=f"m_{r['article_id']}", use_container_width=True)
        
        st.divider()
        
        # 페이지 네비게이션 (하단)
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col1:
            if st.button("⬅️ 이전", use_container_width=True):
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
        with col2:
            st.markdown(f"<div style='text-align: center; font-weight: bold; padding: 10px;'>{st.session_state.current_page} / {total_pages}</div>", unsafe_allow_html=True)
        with col3:
            if st.button("다음 ➡️", use_container_width=True):
                if st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1
                    st.rerun()

def render_search_results_page(query):
    st.button("⬅️ 메인으로", on_click=show_main_page, use_container_width=True)
    st.title(f"🔍 '{query}' 검색 결과")

    with st.spinner("관련 뉴스를 찾는 중..."):
        q_vec = ap.run_gemini_embedding(query)
        df_hits = repo.search_similar_chunks(q_vec, limit=10, min_score=0.55)

    if df_hits.empty:
        st.warning("❌ 관련 뉴스를 찾지 못했습니다.")
        return

    st.success(f"✅ 관련 뉴스를 찾았습니다!")

    df_articles = df_hits.drop_duplicates("article_id")
    for idx, (_, r) in enumerate(df_articles.iterrows(), 1):
        similarity_percent = int(r['score'] * 100)
        
        with st.container(border=True):
            col_t, col_s, col_b = st.columns([0.65, 0.15, 0.2])
            with col_t:
                st.markdown(f"**{idx}. {r['title']}**")
                st.caption(f"📰 출처: {r['source']}")
            with col_s:
                if similarity_percent >= 70:
                    st.markdown(f"🟢 **{similarity_percent}%**")
                elif similarity_percent >= 55:
                    st.markdown(f"🟡 **{similarity_percent}%**")
                else:
                    st.markdown(f"🔴 **{similarity_percent}%**")
            with col_b:
                st.button("보기", key=f"s_{r['article_id']}", 
                          on_click=select_article, args=(r['article_id'],), use_container_width=True)

def render_detail_page(aid):
    st.button("⬅️ 메인으로", on_click=show_main_page, use_container_width=True)
    df = repo.load_articles()
    row = df[df["article_id"] == aid].iloc[0]
    
    # 레이아웃: 왼쪽(요약 + 관련 기사), 오른쪽(본문)
    left_col, right_col = st.columns([0.35, 0.65])
    
    # ===== 왼쪽 컬럼: 요약 및 관련 기사 =====
    with left_col:
        st.subheader("📋 AI 요약")
        with st.container(border=True):
            st.markdown(row['summary_text'])
        
        st.divider()
        
        # 관련 기사
        st.subheader("🔗 관련 기사")
        article_vec = ap.run_gemini_embedding(row['full_text'])
        related_df = repo.search_similar_chunks(article_vec, limit=100, min_score=0.65)
        
        if not related_df.empty:
            related_articles = related_df[related_df['article_id'] != aid].drop_duplicates('article_id')
            for _, rel in related_articles.head(3).iterrows():
                score_pct = int(rel['score'] * 100)
                st.markdown(f"""
                <div style="padding: 8px; border-left: 3px solid #FF6B35; background-color: #f9f9f9; margin-bottom: 8px;">
                    <small><b>{rel['title'][:50]}...</b></small><br>
                    <small>유사도: {score_pct}%</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button("확인", key=f"rel_{rel['article_id']}", use_container_width=True):
                    select_article(rel['article_id'])
        else:
            st.info("관련 기사가 없습니다.")
    
    # ===== 오른쪽 컬럼: 기사 본문 =====
    with right_col:
        # 메타정보
        st.markdown(f"**📰 {row['source']}** | 📅 {row.get('published_at', '날짜미상')}")
        st.markdown(f"✍️ {row.get('title', '')}")
        
        # 원본 링크
        if row.get('url'):
            st.markdown(f"[🔗 원문 보기]({row['url']})")
        
        st.divider()
        
        st.markdown(row['full_text'])

def render_admin_page():
    st.title("⚙️ 관리자 설정")
    st.button("⬅️ 돌아가기", on_click=show_main_page, use_container_width=True)
    
    st.divider()
    st.subheader("🔄 네이버 뉴스 크롤링")
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.caption("카테고리당 기사 수를 지정할 수 있습니다")
    with col2:
        articles_per_cat = st.number_input("카테고리당 기사 수", min_value=5, max_value=50, value=10)
    
    if st.button("🚀 크롤링 시작", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 크롤링
            status_text.info("📰 네이버 뉴스 크롤링 중...")
            df_raw = fetch_articles_from_naver(max_articles_per_category=articles_per_cat)
            
            if df_raw.empty:
                st.error("❌ 크롤링된 기사가 없습니다.")
                return
            
            st.success(f"✅ {len(df_raw)}개 기사 크롤링 완료!")
            progress_bar.progress(30)
            
            # 요약 및 임베딩
            status_text.info("📝 요약 및 임베딩 생성 중...")
            ap.build_ready_rows_from_naver(df_raw)
            
            progress_bar.progress(100)
            status_text.success("✅ 모든 작업 완료!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
    
    st.divider()
    
    # 현재 DB 상태
    st.subheader("📊 데이터베이스 현황")
    df_articles = repo.load_articles()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 기사 수", len(df_articles))
    with col2:
        st.metric("소스 종류", df_articles['source'].nunique() if not df_articles.empty else 0)
    with col3:
        st.metric("DB 크기", f"{len(df_articles)} 건")
    
    if not df_articles.empty:
        st.subheader("최신 기사 목록")
        display_df = df_articles[['title', 'source', 'published_at', 'status']].head(10)
        st.dataframe(display_df, use_container_width=True)
