import streamlit as st
import pandas as pd
import json
from Streamlit_Rendering import repo
from Streamlit_Rendering import admin_pipeline as ap
from Streamlit_Rendering.crawl import fetch_articles_from_naver
from Streamlit_Rendering.trust import score_trust
import re
import sys
import os


def go_back():
    if st.session_state["selected_article_id"]:
        st.session_state["selected_article_id"] = None
    else:
        st.session_state["admin_mode"] = False
        st.session_state["search_executed"] = False
        st.session_state["search_query"] = ""


def select_article(aid):
    st.session_state["selected_article_id"] = aid
    st.session_state["admin_mode"] = False


def show_main_page():
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False
    st.session_state["admin_mode"] = False


def go_to_admin():
    st.session_state["admin_mode"] = True
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False


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
        articles_per_page = 10
        total_pages = (len(df) + articles_per_page - 1) // articles_per_page

        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        start_idx = (st.session_state.current_page - 1) * articles_per_page
        end_idx = start_idx + articles_per_page
        page_articles = df.iloc[start_idx:end_idx]

        for _, r in page_articles.iterrows():
            st.button(
                f"[{r['source']}] {r['title']}",
                on_click=select_article,
                args=(r['article_id'],),
                key=f"m_{r['article_id']}",
                use_container_width=True
            )

        st.divider()
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col1:
            if st.button("⬅️ 이전", use_container_width=True):
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
        with col2:
            st.markdown(
                f"<div style='text-align:center;font-weight:bold;padding:10px;'>"
                f"{st.session_state.current_page} / {total_pages}</div>",
                unsafe_allow_html=True
            )
        with col3:
            if st.button("다음 ➡️", use_container_width=True):
                if st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1
                    st.rerun()


def render_search_results_page(query: str):
    st.button("⬅️ 뒤로가기", on_click=go_back, use_container_width=True)
    st.title(f"🔍 '{query}' 분석 결과")

    with st.spinner("관련 뉴스를 분석 중..."):
        q_vec = ap.run_gemini_embedding(query, task_type="retrieval_query")
        # 65% 이상인 기사만, 개수 제한 없이 반환
        df_hits = repo.search_similar_chunks(q_vec, limit=9999, min_score=0.65)

    if df_hits.empty:
        st.warning("❌ 관련 뉴스를 찾지 못했습니다.")
        return

    st.success(f"✅ {len(df_hits)}개의 관련 뉴스를 찾았습니다!")

    for idx, (_, r) in enumerate(df_hits.iterrows(), 1):
        similarity_percent = int(r['score'] * 100)
        with st.container(border=True):
            col_t, col_s, col_b = st.columns([0.65, 0.15, 0.2])
            with col_t:
                st.markdown(f"**{idx}. {r['title']}**")
                st.caption(f"📰 출처: {r['source']}")
                # 🔧 매칭된 청크 미리보기 표시 → 왜 이 기사가 떴는지 투명하게 보여줌
                if r.get('chunk_text') and not str(r['chunk_text']).startswith("[제목]"):
                    preview = str(r['chunk_text'])[:80].strip()
                    st.caption(f"💬 관련 내용: ...{preview}...")
            with col_s:
                if similarity_percent >= 65:
                    st.markdown(f"🟢 **{similarity_percent}%**")
                elif similarity_percent >= 50:
                    st.markdown(f"🟡 **{similarity_percent}%**")
                else:
                    st.markdown(f"🔴 **{similarity_percent}%**")
            with col_b:
                st.button(
                    "보기",
                    key=f"s_{r['article_id']}",
                    on_click=select_article,
                    args=(r['article_id'],),
                    use_container_width=True
                )


def render_detail_page(aid: str):
    st.button("⬅️ 뒤로가기", on_click=go_back, use_container_width=True)
    df = repo.load_articles()
    row = df[df["article_id"] == aid].iloc[0]

    left_col, right_col = st.columns([0.35, 0.65])

    with left_col:
        st.subheader("📋 AI 요약")
        with st.container(border=True):
            st.markdown(row['summary_text'])

        st.divider()
        st.subheader("🔗 관련 기사")

        # 🔧 튜닝 포인트 D: 관련 기사 검색 전략 개선
        #    기존: 전체 본문(full_text) 임베딩 → 너무 길어 노이즈 많음
        #    변경: 제목 + 요약문 임베딩 → 핵심 의미만 압축해서 더 정확한 관련 기사 탐색
        title_summary = f"{row['title']}\n{row.get('summary_text', '')}"
        article_vec = ap.run_gemini_embedding(title_summary, task_type="retrieval_query")

        # 🔧 튜닝 포인트 E: 현재 기사를 SQL에서 미리 제외 (전용 함수 활용)
        related_df = repo.search_similar_chunks_excluding(
            article_vec,
            exclude_article_id=aid,
            limit=5,
            min_score=0.65   # 관련 기사는 조금 더 관대하게
        )

        if not related_df.empty:
            for _, rel in related_df.iterrows():
                score_pct = int(rel['score'] * 100)
                # 색상: 점수에 따라 구분
                border_color = "#4CAF50" if score_pct >= 65 else "#FF9800" if score_pct >= 50 else "#9E9E9E"
                st.markdown(f"""
                <div style="padding:8px;border-left:3px solid {border_color};
                            background-color:#f9f9f9;margin-bottom:8px;">
                    <small><b>{rel['title'][:50]}...</b></small><br>
                    <small>📰 {rel['source']} &nbsp;|&nbsp; 유사도: {score_pct}%</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button("확인", key=f"rel_{rel['article_id']}", use_container_width=True):
                    select_article(rel['article_id'])
                    st.rerun()
        else:
            st.info("관련 기사가 없습니다.")

    with right_col:
        st.markdown(f"**📰 {row['source']}** | 📅 {row.get('published_at', '날짜미상')}")
        st.markdown(f"### ✍️ {row.get('title', '')}")
        if row.get('url'):
            st.markdown(f"[🔗 원문 보기]({row['url']})")

        # ── 신뢰도 패널 ──────────────────────────────────────────
        trust_score = int(row.get("trust_score") or 0)
        trust_verdict = str(row.get("trust_verdict") or "")
        trust_reason = str(row.get("trust_reason") or "")
        trust_raw = row.get("trust_per_criteria") or "{}"

        st.divider()
        st.subheader("🔐 신뢰도 분석")

        if trust_score == 0 and trust_verdict in ("", "None", "uncertain"):
            st.info("신뢰도 분석 데이터가 없습니다. Admin에서 크롤링 후 자동 분석됩니다.")
        else:
            verdict_badge = {
                "likely_true":  "🟢 likely_true",
                "uncertain":    "🟡 uncertain",
                "likely_false": "🔴 likely_false",
            }.get(trust_verdict, trust_verdict)

            col_score, col_verdict = st.columns([0.4, 0.6])
            with col_score:
                st.metric("종합 점수", f"{trust_score}점 / 100")
            with col_verdict:
                st.markdown(f"**판정**: {verdict_badge}")

            st.progress(trust_score / 100)

            # 기준별 점수
            try:
                per_criteria = json.loads(trust_raw) if isinstance(trust_raw, str) else trust_raw
            except (json.JSONDecodeError, TypeError):
                per_criteria = {}

            if per_criteria:
                criteria_labels = {
                    "source_credibility":  "출처 신뢰성",
                    "evidence_support":    "근거 지지도",
                    "style_neutrality":    "문체 중립성",
                    "logical_consistency": "논리 일관성",
                    "clickbait_risk":      "어뷰징 위험도",
                }
                with st.expander("기준별 세부 점수 보기", expanded=True):
                    for key, label in criteria_labels.items():
                        item = per_criteria.get(key, {})
                        c_score = int(item.get("score", 0))
                        c_reason = str(item.get("reason", ""))
                        col_l, col_r = st.columns([0.55, 0.45])
                        with col_l:
                            st.markdown(f"**{label}**")
                            st.progress(c_score / 10)
                        with col_r:
                            st.markdown(f"{c_score} / 10")
                            if c_reason:
                                st.caption(c_reason)

            if trust_reason:
                st.markdown(f"**📝 종합 판단**: {trust_reason}")
        # ── 신뢰도 패널 끝 ────────────────────────────────────────

        st.divider()
        st.markdown(row['full_text'])


def render_admin_page():
    st.title("⚙️ 관리자 설정")
    st.button("⬅️ 뒤로가기", on_click=go_back, use_container_width=True)
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
            status_text.info("📰 네이버 뉴스 크롤링 중...")
            df_raw = fetch_articles_from_naver(max_articles_per_category=articles_per_cat)
            if df_raw.empty:
                st.error("❌ 크롤링된 기사가 없습니다.")
                return
            st.success(f"✅ {len(df_raw)}개 기사 크롤링 완료!")
            progress_bar.progress(30)
            status_text.info("📝 요약 및 임베딩 생성 중...")
            ap.build_ready_rows_from_naver(df_raw)
            progress_bar.progress(100)
            status_text.success("✅ 모든 작업 완료!")
            st.balloons()
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

    st.divider()
    st.subheader("🔍 신뢰도 일괄 분석")
    df_no_trust = repo.load_articles_without_trust()
    unanalyzed_count = len(df_no_trust)
    st.caption(f"trust_score=0인 미분석 기사: **{unanalyzed_count}건**")

    if unanalyzed_count > 0:
        if st.button("🔍 신뢰도 일괄 분석 시작", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            success_count = 0
            for i, (_, row) in enumerate(df_no_trust.iterrows()):
                status_text.info(f"[{i+1}/{unanalyzed_count}] {row['title'][:40]}...")
                try:
                    trust = score_trust(str(row["full_text"]), str(row["source"]))
                    repo.update_article_trust(
                        article_id=row["article_id"],
                        score=trust["score"],
                        verdict=trust["verdict"],
                        reason=trust["reason"],
                        per_criteria=json.dumps(trust["per_criteria"], ensure_ascii=False),
                    )
                    success_count += 1
                except Exception as e:
                    st.warning(f"⚠️ {row['title'][:30]}... 분석 실패: {e}")
                progress_bar.progress((i + 1) / unanalyzed_count)
            status_text.success(f"✅ {success_count}/{unanalyzed_count}건 분석 완료!")
            st.rerun()
    else:
        st.success("✅ 모든 기사에 신뢰도 분석이 완료되어 있습니다.")

    st.divider()
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