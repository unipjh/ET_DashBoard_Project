import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import pandas as pd
import json
from Streamlit_Rendering import repo
from Streamlit_Rendering.crawl import fetch_articles_from_naver
from Streamlit_Rendering.search import run_gemini_embedding
from Streamlit_Rendering.trust import score_trust
from Streamlit_Rendering.config import get_gemini_api_key
import random
import time

genai.configure(api_key=get_gemini_api_key())


def run_gemini_summary(text: str) -> str:
    """Gemini로 요약"""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    attempt = 1

    while True:
        try:
            response = model.generate_content(f"다음 뉴스를 3문장 내외로 요약해줘:\n\n{text}")
            return response.text
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = random.randint(30, 60)
                print(f"⚠️ 할당량 초과! {wait_time}초 후 재시도...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                return f"요약 중 에러: {e}"


def _make_chunk_context(title: str, source: str, category: str, chunk_text: str) -> str:
    """
    Contextual Chunking
    청크에 제목/출처/카테고리를 prefix로 추가.
    → 청크가 어떤 기사의 일부인지 의미 정보가 임베딩에 반영됨
    """
    return f"[제목: {title}] [출처: {source}] [카테고리: {category}]\n{chunk_text}"


def build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int:
    """
    크롤링된 기사를 처리하여 DB에 적재
    """
    rows, chunk_rows = [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150,
        separators=["다.\n", "다. ", ".\n", ". ", "\n\n", "\n", " ", ""]
    )

    total = len(df_raw)
    for idx, r in df_raw.iterrows():
        progress = int((idx + 1) / total * 100)
        print(f"📦 [{progress}%] {idx+1}/{total} - {r.get('title', '')[:40]}")

        full_text = str(r.get("content", ""))
        title = str(r.get("title", ""))
        source = str(r.get("source", "미상"))
        url = str(r.get("link", ""))
        published_at = str(r.get("date", "날짜미상"))
        reporter = str(r.get("reporter", "미상"))
        category = str(r.get("category", "일반"))

        if len(full_text.strip()) < 10:
            print(f"⚠️ {idx+1}번 기사 본문이 비어있어 건너뜁니다.")
            continue

        aid = f"naver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"

        # 1. 요약 생성
        print(f"  📝 요약 생성 중...")
        summary = run_gemini_summary(full_text)
        time.sleep(2)

        # 2. 신뢰도 분석
        print(f"  🔍 신뢰도 분석 중...")
        trust = score_trust(full_text, source)
        time.sleep(2)

        rows.append({
            "article_id": aid,
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_at,
            "full_text": full_text,
            "summary_text": summary,
            "keywords": "[]",
            "embed_full": "[]",
            "embed_summary": "[]",
            "trust_score":        trust["score"],
            "trust_verdict":      trust["verdict"],
            "trust_reason":       trust["reason"],
            "trust_per_criteria": json.dumps(trust["per_criteria"], ensure_ascii=False),
            "status": "ready"
        })

        # 3. 청크 임베딩 (Contextual Chunking 적용)
        print(f"  🔢 청크 임베딩 생성 중...")
        chunks = splitter.split_text(full_text)

        for i, chunk_text in enumerate(chunks):
            time.sleep(0.5)
            contextualized = _make_chunk_context(title, source, category, chunk_text)
            v = run_gemini_embedding(contextualized, task_type="retrieval_document")
            chunk_rows.append({
                "chunk_id": f"{aid}_{i}",
                "article_id": aid,
                "chunk_text": chunk_text,
                "embedding": v
            })

        # 4. 제목 전용 청크 추가
        title_embedding = run_gemini_embedding(
            f"[제목] {title}", task_type="retrieval_document"
        )
        chunk_rows.append({
            "chunk_id": f"{aid}_title",
            "article_id": aid,
            "chunk_text": f"[제목] {title}",
            "embedding": title_embedding
        })

        time.sleep(3)

    # DB 저장
    if rows:
        print(f"\n💾 {len(rows)}개 기사를 DB에 저장 중...")
        repo.upsert_articles(pd.DataFrame(rows))

    if chunk_rows:
        print(f"💾 {len(chunk_rows)}개 청크를 DB에 저장 중...")
        repo.upsert_chunks(pd.DataFrame(chunk_rows))

    print(f"✅ 작업 완료! {len(rows)}개 기사 적재됨")
    return len(rows)