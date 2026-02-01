# Streamlit_Rendering/admin_pipeline.py
import json
import re
import torch
import numpy as np
import pandas as pd

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
from sklearn.metrics.pairwise import cosine_similarity
from Streamlit_Rendering.summary import get_summarizer_instance
from Streamlit_Rendering.trust import score_trust_dummy
ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    """
    더미 크롤링 함수
    URL 1개 → 크롤링 → (중복 필터링) → DB 적재
    반환: {"status": "inserted"/"skipped"/"error", "message": "...", "url": "..."}
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


## 0201_ 가현 수정 사항 ##
def run_summary(full_text: str) -> str:
    """KoBERT 기반 문장 추출 요약 수행 """
    if not full_text: return ""
    summarizer = get_summarizer_instance()
    clean_text = summarizer.preprocess(full_text)
    
    # 문장 단위 분리 
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 20]
    if len(sents) <= 3:
        return clean_text

    # 문장 임베딩 생성 
    inputs = summarizer.tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=128).to(summarizer.device)
    with torch.no_grad():
        outputs = summarizer.model(**inputs)
    
    sent_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # 문장 간 유사도 합산 점수 기반 상위 3개 추출 
    sim_matrix = cosine_similarity(sent_embs, sent_embs)
    scores = sim_matrix.sum(axis=1)
    top_indices = sorted(np.argsort(scores)[::-1][:3])
    
    return ' '.join([sents[i] for i in top_indices])

def run_keywords(full_text: str) -> list[str]:
    """KeyBERT 기반 키워드 추출 """
    if not full_text: return []
    summarizer = get_summarizer_instance()
    clean_text = summarizer.preprocess(full_text)
    
    keywords_tuples = summarizer.kw_model.extract_keywords(
        clean_text, keyphrase_ngram_range=(1, 1), 
        stop_words=summarizer.stopwords, top_n=5
    )
    return [k[0] for k in keywords_tuples]

def run_embedding(text: str) -> list[float]:
    """KoBERT CLS 토큰 기반 임베딩 생성 """
    if not text: return []
    summarizer = get_summarizer_instance()
    
    inputs = summarizer.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(summarizer.device)
    with torch.no_grad():
        outputs = summarizer.model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding.tolist()

def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """수정된 run_ 함수들을 사용하여 데이터를 가공 """
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        
        # 모델 기반 데이터 생성 
        summary_text = run_summary(full_text)
        keywords = run_keywords(full_text)
        embed_full = run_embedding(full_text)
        embed_summary = run_embedding(summary_text)

        # 신뢰도 평가 (더미 로직 유지) 
        trust = score_trust_dummy(full_text, source=str(r["source"]))

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": str(r["source"]),
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,
            "summary_text": summary_text,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "embed_full": json.dumps(embed_full),
            "embed_summary": json.dumps(embed_summary),
            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(trust.get("per_criteria", {}), ensure_ascii=False),
            "status": "ready",
        })

    return pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)

