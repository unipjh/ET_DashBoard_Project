import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import pandas as pd
from Streamlit_Rendering import repo
from Streamlit_Rendering.crawl import fetch_articles_from_naver
import random
import time

# API 키 설정
genai.configure(api_key="") # 본인 api 키

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
                print(f"⚠️ 할당량 초과! {wait_time}초 후 다시 도전합니다...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                return f"요약 중 에러: {e}"

def run_gemini_embedding(text: str) -> list:
    """Gemini 임베딩"""
    if not text or str(text).strip() == "":
        return [0.0] * 768
    
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embedding = result['embedding']
        
        if len(embedding) > 768:
            embedding = embedding[:768]
        
        return embedding
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return [0.0] * 768

def build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int:
    """
    크롤링된 기사를 처리하여 DB에 적재
    
    Args:
        df_raw: 크롤링된 기사 DataFrame
    
    Returns:
        적재된 기사 수
    """
    rows, chunk_rows = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    total = len(df_raw)
    for idx, r in df_raw.iterrows():
        progress = int((idx + 1) / total * 100)
        print(f"📦 [{progress}%] {idx+1}/{total} 번째 기사 처리 중... - {r.get('title', '')[:40]}")
        
        # crawl.py의 반환 포맷 대응
        full_text = str(r.get("content", ""))  # crawl.py에서는 'content' 사용
        title = str(r.get("title", ""))
        source = str(r.get("source", "미상"))
        url = str(r.get("link", ""))  # crawl.py에서는 'link' 사용
        published_at = str(r.get("date", "날짜미상"))
        reporter = str(r.get("reporter", "미상"))
        
        if len(full_text.strip()) < 10:
            print(f"⚠️ {idx+1}번 기사 본문이 비어있어 건너뜁니다.")
            continue

        aid = f"naver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
        
        # 1. Gemini로 요약
        print(f"  📝 요약 생성 중...")
        summary = run_gemini_summary(full_text)
        time.sleep(2)

        # 2. 기사 데이터 구성
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
            "trust_score": 0,
            "trust_verdict": "None",
            "trust_reason": "",
            "trust_per_criteria": "{}",
            "status": "ready"
        })

        # 3. 청크 임베딩
        print(f"  🔢 임베딩 생성 중...")
        chunks = splitter.split_text(full_text)
        for i, txt in enumerate(chunks):
            time.sleep(0.5)
            v = run_gemini_embedding(txt)
            chunk_rows.append({
                "chunk_id": f"{aid}_{i}", 
                "article_id": aid,
                "chunk_text": txt, 
                "embedding": v
            })

        time.sleep(3)  # API 속도 제한 회피

    # DB 저장
    if rows: 
        print(f"\n💾 {len(rows)}개 기사를 DB에 저장 중...")
        repo.upsert_articles(pd.DataFrame(rows))
    
    if chunk_rows: 
        print(f"💾 {len(chunk_rows)}개 청크를 DB에 저장 중...")
        repo.upsert_chunks(pd.DataFrame(chunk_rows))
    
    print(f"✅ 작업 완료! {len(rows)}개 기사 적재됨")
    return len(rows)


