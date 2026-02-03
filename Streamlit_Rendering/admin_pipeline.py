# Streamlit_Rendering/admin_pipeline.py
import re
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
from Streamlit_Rendering.summary import get_summarizer
from sklearn.metrics.pairwise import cosine_similarity
from Streamlit_Rendering.trust import score_trust_dummy

ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def run_crawling_pipeline(max_articles_per_category: int = 30) -> tuple:
    """
    네이버 실시간 크롤링 후 DB 적재 (crawl.py 포맷 대응)
    """
    from Streamlit_Rendering.crawl import fetch_articles_from_naver
    try:
        # 1. crawl.py의 새로운 포맷으로 데이터 수집
        df_raw = fetch_articles_from_naver(max_articles_per_category=max_articles_per_category)
        if df_raw.empty: 
            return 0, ["크롤링 결과가 비어있습니다."]
        
        # 2. 새로운 포맷 데이터를 DB용 포맷으로 변환 및 AI 처리
        df_ready = build_ready_rows_from_naver(df_raw)
        
        # 3. DB 적재
        repo.upsert_articles(df_ready)
        return len(df_ready), []
    except Exception as e:
        return 0, [str(e)]



def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    """
    URL 1개 → 크롤링 → (중복 필터링) → DB 적재
    """
    try:
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다. (중복 스킵)", "url": url}

        df_raw = fetch_article_from_url(url=url, source=source)
        df_ready = build_ready_rows(df_raw)

        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}

def run_trust(full_text: str, source: str) -> dict:
    return score_trust_dummy(full_text, source=source, low=30, high=100)


## 0201 가현 수정 사항
def run_summary(full_text: str) -> str:
    """KoBERT 기반 문장 추출 요약 (상위 3개 문장)"""
    if not full_text: return ""
    model_obj = get_summarizer()
    clean_text = model_obj._preprocess_text(full_text)
    
    # 문장 분리
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 20]
    if len(sents) <= 3:
        return clean_text

    # 문장 임베딩 생성
    inputs = model_obj.tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model_obj.device)
    with torch.no_grad():
        outputs = model_obj.model(**inputs)
    
    sent_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # 유사도 기반 중요 문장 추출
    sim_matrix = cosine_similarity(sent_embs, sent_embs)
    scores = sim_matrix.sum(axis=1)
    top_indices = sorted(np.argsort(scores)[::-1][:3])
    
    return ' '.join([sents[i] for i in top_indices])

def run_keywords(full_text: str) -> list[str]:
    """KeyBERT 기반 키워드 추출 (MMR 적용)"""
    if not full_text: return []
    model_obj = get_summarizer()
    clean_text = model_obj._preprocess_text(full_text)
    
    try:
        keywords_tuples = model_obj.kw_model.extract_keywords(
            clean_text, 
            keyphrase_ngram_range=(1, 1), 
            stop_words=model_obj.stopwords_list, 
            top_n=5,
            use_mmr=True,
            diversity=0.3
        )
        return [k[0] for k in keywords_tuples]
    except Exception:
        return []

def run_embedding(text: str) -> list[float]:
    """KoBERT 임베딩 생성"""
    if not text: return [0.0] * 768
    model_obj = get_summarizer()
    
    inputs = model_obj.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_obj.device)
    with torch.no_grad():
        outputs = model_obj.model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding.tolist()

def build_ready_rows_from_naver(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    crawl.py의 반환값(date, link, content 등)을 DB 컬럼명에 맞게 변환
    """
    rows = []
    for idx, r in df_raw.iterrows():
        # crawl.py 포맷 -> DB 스키마 포맷 매핑
        full_text = str(r.get("content", ""))
        source = str(r.get("source", "Naver News"))
        url = str(r.get("link", ""))
        published_at = str(r.get("date", ""))
        title = str(r.get("title", ""))
        
        # 고유 ID 생성
        article_id = f"naver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
        
        # AI 분석 엔진 호출
        summary_text = run_summary(full_text)
        keywords = run_keywords(full_text)
        embed_full = run_embedding(full_text)
        embed_summary = run_embedding(summary_text)
        trust = score_trust_dummy(full_text, source=source)

        rows.append({
            "article_id": article_id,
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_at,
            "full_text": full_text,
            "summary_text": summary_text,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "embed_full": json.dumps(embed_full),
            "embed_summary": json.dumps(embed_summary),
            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": str(trust.get("verdict", "uncertain")),
            "trust_reason": str(trust.get("reason", "")),
            "trust_per_criteria": json.dumps(trust.get("per_criteria", {}), ensure_ascii=False),
            "status": "ready",
        })
    return pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
