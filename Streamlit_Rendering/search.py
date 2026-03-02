import os
import json
import duckdb
import pandas as pd
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 초기 설정 (API 키를 입력하세요)
GEMINI_API_KEY = "AIzaSyCS-p0a8ZgbRgVE01d5oGyafWJy5yh48xw" # 본인 api 키
genai.configure(api_key=GEMINI_API_KEY)
DB_PATH = "app_db.duckdb"

# 2. Gemini 유틸리티 함수
def get_gemini_summary(text):
    """Gemini 2.5 Flash를 이용한 요약"""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(f"다음 뉴스를 3문장 이내로 핵심만 요약해줘:\n\n{text}")
    return response.text

# search.py

def get_gemini_embedding(text):
    """검색어를 벡터로 변환 (001 모델로 통일)"""
    import google.generativeai as genai
    
    # 모델명을 001로 변경
    model_name = "models/embedding-001" 
    
    try:
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_query" 
        )
        embedding = result['embedding']
        
        # DB의 FLOAT[768] 규격과 맞추는 방어 로직
        if len(embedding) > 768:
            embedding = embedding[:768]
        elif len(embedding) < 768:
            embedding = embedding + [0.0] * (768 - len(embedding))
            
        return embedding
    except Exception as e:
        print(f"❌ 검색 임베딩 생성 실패: {e}")
        return [0.0] * 768

# 3. DB 전용 테이블 생성 및 초기화
def init_experiment_db():
    con = duckdb.connect(DB_PATH)
    # 청크 및 벡터 저장용 테이블 (이미 있으면 무시)
    con.execute("""
    CREATE TABLE IF NOT EXISTS article_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        article_id VARCHAR,
        chunk_text VARCHAR,
        embedding FLOAT[768]
    );
    """)
    con.close()

# 4. RAG 실험 프로세스 (기사 5개 처리)
def run_experiment():
    print("🚀 Gemini RAG 실험을 시작합니다...")
    init_experiment_db()
    
    # 기존 MOCK_DB에서 기사 5개 가져오기 (파일 import 대신 직접 정의된 데이터 사용 가정)
    from Streamlit_Rendering.data import MOCK_DB_NORMALIZED
    sample_articles = list(MOCK_DB_NORMALIZED.values())[:5]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    con = duckdb.connect(DB_PATH)

    for a in sample_articles:
        aid = a['article_id']
        print(f"📦 처리 중: {a['title']}")
        
        # [요약] Gemini 활용
        summary = get_gemini_summary(a['full_text'])
        
        # [청크 분할]
        chunks = splitter.split_text(a['full_text'])
        
        for i, text in enumerate(chunks):
            # [임베딩] Gemini 활용
            vec = get_gemini_embedding(text)
            
            # [DB 저장]
            con.execute("""
                INSERT OR REPLACE INTO article_chunks VALUES (?, ?, ?, ?)
            """, [f"{aid}_{i}", aid, text, vec])
            
    con.close()
    print("✅ 데이터 적재 완료!")

# 5. 검색 및 Gemini 브리핑 (RAG 핵심)
def search_and_analyze(query):
    print(f"\n🔍 검색어: '{query}'에 대한 분석 결과")
    
    # 1. 질의 임베딩
    query_vec = get_gemini_embedding(query)
    
    # 2. 시맨틱 검색 (유사도 계산)
    con = duckdb.connect(DB_PATH)
    df_hits = con.execute("""
        SELECT c.chunk_text, a.title, 
               list_cosine_similarity(c.embedding, ?::FLOAT[768]) as score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        ORDER BY score DESC LIMIT 3
    """, [query_vector]).fetchdf()
    con.close()

    # 3. Gemini 최종 분석 (Explainable AI)
    context = "\n\n".join([f"[{r['title']}] {r['chunk_text']}" for _, r in df_hits.iterrows()])
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    prompt = f"""
    당신은 뉴스 분석 전문가입니다. 아래 뉴스 조각(Context)들을 읽고 사용자의 질문 '{query}'에 대해 브리핑하세요.
    
    [진행 지침]
    1. 관련 있는 기사들을 리스트업하고 핵심 내용을 요약할 것.
    2. 왜 이 기사들이 사용자의 검색어와 관련이 있는지 '설명'할 것.
    3. 근거가 부족하면 아는 척 하지 말고 제공된 내용 안에서만 답변할 것.

    [Context]
    {context}
    """
    
    response = model.generate_content(prompt)
    print("-" * 50)
    print(response.text)
    print("-" * 50)

if __name__ == "__main__":
    # 실험 실행 (데이터 적재)
    run_experiment()
    
    # 테스트 검색
    search_and_analyze("강서구 화재 사건의 원인이 뭐야?")