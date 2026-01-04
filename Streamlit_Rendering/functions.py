# Streamlit_Rendering/function.py
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from Streamlit_Rendering import repo
from Streamlit_Rendering import admin_pipeline as ap
# 변경
from Streamlit_Rendering.data import MOCK_DB_NORMALIZED as MOCK_DB


# =========================
# 공통 유틸 (LLM/API 호출 없음)
# =========================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _traffic_light(score: int) -> str:
    if score >= 70:
        return "🟢"
    if score >= 40:
        return "🟡"
    return "🔴"

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _safe_json_loads(s: str, default):
    try:
        if not s:
            return default
        return json.loads(s)
    except Exception:
        return default

def _append_event(event: str, article_id: str):
    user_id = st.session_state.get("user_id", "default_user")
    repo.append_event(user_id=user_id, ts=_now_iso(), event=event, article_id=article_id)


# =========================
# DB seed (데모용)
# - 실제 운영에서는 관리자 파이프라인으로만 데이터가 들어와야 하지만,
#   초기 UI 검증을 위해 DB가 비어 있으면 MOCK를 넣는 버튼/헬퍼를 둡니다.
# =========================

def _seed_db_from_mock(force: bool = False) -> int:
    """
    DB가 비어 있으면 MOCK_DB를 articles 테이블에 적재합니다.
    force=True면 기존 데이터가 있어도 upsert로 덮어씁니다(주의).
    반환: 적재된 row 수(추정)
    """
    df_exist = repo.load_articles()
    if (not force) and (len(df_exist) > 0):
        return 0

    rows = []
    for a in MOCK_DB.values():
        article_id = a.get("article_id") or a.get("id")
        rows.append({
            "article_id": str(article_id),
            "title": a.get("title", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "published_at": a.get("published_at", ""),
            "full_text": a.get("full_text", ""),

            # 아래는 '사전 계산 결과' 컬럼.
            # MOCK에서는 비어있어도 되며, 사용자 화면은 "없음"을 표시합니다.
            "summary_text": a.get("summary_text", ""),
            "keywords": json.dumps(a.get("keywords", []), ensure_ascii=False),
            "embed_full": json.dumps(a.get("embed_full", [])),
            "embed_summary": json.dumps(a.get("embed_summary", [])),
            "trust_score": _safe_int(a.get("trust_score", 50), 50),
            "trust_verdict": a.get("trust_verdict", "uncertain"),
            "trust_reason": a.get("trust_reason", "MOCK 데이터입니다."),
            "trust_per_criteria": json.dumps(a.get("trust_per_criteria", {}), ensure_ascii=False),
            "status": a.get("status", "ready"),
        })

    df_seed = pd.DataFrame(rows)
    repo.upsert_articles(df_seed)
    return len(df_seed)

def _load_articles_df() -> pd.DataFrame:
    df = repo.load_articles()
    if len(df) == 0:
        # 자동 seed는 MVP 편의용. 원치 않으면 주석 처리하십시오.
        _seed_db_from_mock(force=False)
        df = repo.load_articles()

    # 정렬 안정화(문자열 기준)
    if "published_at" in df.columns:
        df = df.sort_values(by="published_at", ascending=False, na_position="last")

    return df

def _get_article_row(article_id: str) -> dict | None:
    df = _load_articles_df()
    sub = df[df["article_id"] == article_id]
    if len(sub) == 0:
        return None
    return sub.iloc[0].to_dict()


# =========================
# 상태 변경 콜백
# =========================

def select_article(article_id: str):
    st.session_state["selected_article_id"] = article_id
    st.session_state["admin_mode"] = False

def show_main_page():
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False
    st.session_state["search_query"] = ""
    st.session_state["admin_mode"] = False

def execute_search():
    st.session_state["search_executed"] = True
    st.session_state["selected_article_id"] = None
    st.session_state["admin_mode"] = False

def go_to_admin():
    st.session_state["admin_mode"] = True
    st.session_state["selected_article_id"] = None
    st.session_state["search_executed"] = False


# =========================
# 관리자 페이지
# - 여기서만 admin_pipeline을 호출하여 DB에 적재
# - function.py는 summary.py/trust.py를 직접 import하지 않음
# =========================

def render_admin_page():
    st.title("관리자 페이지")
    st.button("사용자 모드로 돌아가기", on_click=show_main_page, use_container_width=True)

    st.divider()

    left, right = st.columns([0.6, 0.4])

    with left:
        st.subheader("데이터 적재/갱신")

        if st.button("파이프라인 실행: 크롤링→요약→임베딩→키워드→신뢰도→DB", use_container_width=True):
            try:
                with st.spinner("파이프라인 실행 중..."):
                    # admin_pipeline 내부에서 summary.py, trust.py 등을 사용하도록 구성하십시오.
                    df_raw = ap.crawl_latest_articles()
                    df_ready = ap.build_ready_rows(df_raw)
                    repo.upsert_articles(df_ready)
                st.success(f"적재 완료: {len(df_ready)}건")
            except NotImplementedError:
                st.error("admin_pipeline.py가 아직 미완입니다. (crawl/build_ready_rows 등 구현 필요)")
            except Exception as e:
                st.error(f"파이프라인 실패: {e}")

        st.caption("주의: 사용자 페이지에서는 요약/신뢰도 계산을 절대 수행하지 않습니다. 계산은 관리자 파이프라인에서만 하십시오.")

        st.divider()
        st.subheader("데모용 데이터")
        if st.button("MOCK_DB → DB 적재(초기 UI 확인용)", use_container_width=True):
            n = _seed_db_from_mock(force=False)
            if n == 0:
                st.info("이미 DB에 데이터가 존재합니다(추가 적재 없음).")
            else:
                st.success(f"MOCK_DB 적재 완료: {n}건")

    with right:
        st.subheader("대시보드(더미)")
        chart_data = np.random.randn(30, 2)
        st.line_chart(chart_data)
        st.info("이 영역은 user_events를 집계한 실데이터 지표로 교체하십시오.")

    st.divider()

    df = _load_articles_df()
    st.subheader("articles 테이블 현황")
    st.caption(f"총 {len(df)}건")
    cols = [c for c in ["article_id", "title", "source", "published_at", "status", "trust_score"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)


# =========================
# 사용자 페이지 - 메인
# =========================

def render_main_page():
    title_col, button_col = st.columns([0.8, 0.2])
    with title_col:
        st.title("AI 기반 뉴스 검색 플랫폼")
    with button_col:
        st.button("Manage Mode", on_click=go_to_admin, use_container_width=True)

    st.markdown("---")

    with st.form(key="search_form"):
        st.session_state["search_query"] = st.text_input(
            "검색어를 입력하세요",
            value=st.session_state["search_query"],
            placeholder="MVP: 제목/본문/키워드 문자열 검색(조회만 수행)"
        )
        st.form_submit_button("검색", on_click=execute_search, use_container_width=True)

    st.markdown("---")

    st.subheader("추천 뉴스 (MVP: 최신 순)")
    df = _load_articles_df()
    top = df.head(8)

    for _, r in top.iterrows():
        aid = r["article_id"]
        label = f"[{r.get('source','')}] {r.get('title','')}"
        st.button(
            label,
            on_click=select_article,
            args=(aid,),
            key=f"main_rec_{aid}",
            use_container_width=True
        )


# =========================
# 사용자 페이지 - 검색 결과
# =========================

def render_search_results_page(query: str):
    st.subheader(f"검색 결과: '{query}'")
    st.caption("MVP: 제목/본문/키워드 문자열 검색 + 최신순 정렬 (모델 호출 없음)")

    df = _load_articles_df()

    q = (query or "").strip()
    if q:
        mask = (
            df["title"].fillna("").str.contains(q, case=False) |
            df["full_text"].fillna("").str.contains(q, case=False) |
            df["keywords"].fillna("").str.contains(q, case=False)
        )
        df = df[mask]

    st.write(f"총 {len(df)}건")

    for _, r in df.iterrows():
        aid = r["article_id"]
        source = r.get("source", "")
        title = r.get("title", "")
        status = r.get("status", "ready")
        trust_score = _safe_int(r.get("trust_score", 0), 0)

        badge = "✅" if status == "ready" else "⏳"

        st.button(
            f"{badge} [{source}] {title}  ({_traffic_light(trust_score)} {trust_score})",
            on_click=select_article,
            args=(aid,),
            key=f"sr_{aid}",
            use_container_width=True
        )

    st.divider()
    st.button("메인으로 돌아가기", on_click=show_main_page, use_container_width=True)


# =========================
# 사용자 페이지 - 상세 (DB 조회만)
# =========================
def render_detail_page(article_id: str):
    row = _get_article_row(article_id)
    if row is None:
        st.error("오류: 기사를 찾을 수 없습니다.")
        st.button("메인으로 돌아가기", on_click=show_main_page, use_container_width=True)
        return

    # 조회 이벤트(중복 방지는 추후)
    _append_event("view", article_id)

    st.button("메인으로 돌아가기", on_click=show_main_page, use_container_width=True)

    st.header(row.get("title", ""))
    st.caption(
        f"출처: {row.get('source','')} | 발행: {row.get('published_at','')} | 원문: {row.get('url','')}"
    )

    st.divider()

    # ✅ 본문만 메인에 표시
    st.subheader("기사 본문")
    full_text = row.get("full_text", "")
    st.markdown(full_text.replace("\n", "  \n\n"))

    st.divider()

    # ✅ 팝업에서 사용할 데이터(사전 계산값)
    score = _safe_int(row.get("trust_score", 0), 0)
    verdict = row.get("trust_verdict", "uncertain")
    reason = row.get("trust_reason", "")
    summary_text = row.get("summary_text", "")

    @st.dialog("요약 / 신뢰도 / 기준별 평가 / 피드백")
    def open_insight_dialog():
        # ---- 신뢰도 ----
        st.subheader("신뢰도")
        st.markdown(f"**{_traffic_light(score)} {score}점**  (verdict: `{verdict}`)")
        st.progress(min(max(score, 0), 100) / 100)
        st.write(reason if reason else "-")

        st.divider()

        # ---- 요약 ----
        st.subheader("요약")
        if summary_text:
            st.info(summary_text)
        else:
            st.warning("요약문이 없습니다. (관리자 파이프라인에서 생성 후 적재해야 합니다.)")

        st.divider()

        # ---- 기준별 평가 ----
        st.subheader("기준별 평가")
        per = _safe_json_loads(row.get("trust_per_criteria", ""), default={})
        if not per:
            st.warning("기준별 평가 데이터가 없습니다.")
        else:
            for k, v in per.items():
                s = v.get("score", None)
                r = v.get("reason", "")
                st.markdown(f"- **{k}**: `{s}`")
                if r:
                    st.write(r)

        st.divider()

        # ---- 피드백 ----
        st.subheader("피드백")
        c1, c2, c3 = st.columns(3)
        if c1.button("도움이 됐어요", key=f"like_{article_id}"):
            _append_event("like", article_id)
            st.success("저장되었습니다.")
        if c2.button("별로였어요", key=f"dislike_{article_id}"):
            _append_event("dislike", article_id)
            st.success("저장되었습니다.")
        if c3.button("이 기사 숨기기", key=f"hide_{article_id}"):
            _append_event("hide", article_id)
            st.success("저장되었습니다.")

    # ✅ 메인에는 팝업 여는 버튼만
    if st.button("요약/신뢰도/피드백 보기", use_container_width=True):
        open_insight_dialog()
