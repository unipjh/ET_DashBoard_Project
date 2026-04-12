import os
import re
import duckdb
import pandas as pd
import uuid
from datetime import datetime
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
_MD_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "")
DB_PATH = f"md:et_db?motherduck_token={_MD_TOKEN}" if _MD_TOKEN else "app_db.duckdb"

# ============================================================
# 한국어 토크나이저 (공백 + 2글자 이상 단어)
# ============================================================
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[가-힣a-zA-Z0-9]{2,}', text)
    return tokens if tokens else [""]


def init_db():
    con = duckdb.connect(DB_PATH)

    con.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        article_id VARCHAR PRIMARY KEY,
        title VARCHAR,
        source VARCHAR,
        url VARCHAR,
        published_at VARCHAR,
        full_text VARCHAR,
        summary_text VARCHAR,
        keywords VARCHAR,
        embed_full VARCHAR,
        embed_summary VARCHAR,
        trust_score INTEGER,
        trust_verdict VARCHAR,
        trust_reason VARCHAR,
        trust_per_criteria VARCHAR,
        status VARCHAR
    );
    """)

    # DB 마이그레이션: 기존 테이블에 category 컬럼이 없으면 자동으로 추가
    try:
        con.execute("ALTER TABLE articles ADD COLUMN category VARCHAR;")
    except Exception:
        pass
    
    # 남은 미분류 기사들을 '연예'로 일괄 강제 업데이트 (1회성 자동 보정)
    try:
        con.execute("UPDATE articles SET category = '연예' WHERE category IS NULL OR category IN ('미분류', '일반', '');")
    except Exception:
        pass

    # DB 마이그레이션: 과거에 저장된 제목 전용 청크의 '[제목]...' 텍스트 노이즈 일괄 제거
    try:
        con.execute("UPDATE article_chunks SET chunk_text = '' WHERE chunk_id LIKE '%_title';")
    except Exception:
        pass

    # 특정 에러/테스트 기사 강제 일괄 삭제 ("Britain Royals" 포함 제목)
    try:
        con.execute("DELETE FROM article_chunks WHERE article_id IN (SELECT article_id FROM articles WHERE title LIKE '%Britain Royals%');")
        con.execute("DELETE FROM articles WHERE title LIKE '%Britain Royals%';")
    except Exception:
        pass

    # DB 마이그레이션: 기존 본문/요약에서 맨 앞이나 맨 뒤에 혼자 남은 "OOO 기자" 꼬리표만 안전하게 제거
    try:
        for table_col in [("articles", "full_text"), ("articles", "summary_text"), ("article_chunks", "chunk_text")]:
            table, col = table_col
            # 맨 앞 기자명 제거 (예: "홍길동 기자 = ")
            con.execute(f"UPDATE {table} SET {col} = regexp_replace({col}, '^\\s*[가-힣]{2,5}\\s*기자\\s*(?:=\\s*)?', '');")
            # 맨 뒤 기자명 제거
            con.execute(f"UPDATE {table} SET {col} = regexp_replace({col}, '\\s*[가-힣]{2,5}\\s*기자\\s*$', '');")
    except Exception:
        pass

    con.execute("""
    CREATE TABLE IF NOT EXISTS article_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        article_id VARCHAR,
        chunk_text VARCHAR,
        embedding FLOAT[768]
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS feedback_logs (
        feedback_id VARCHAR PRIMARY KEY,
        article_id VARCHAR,
        feedback_type VARCHAR, -- 'like' or 'dislike'
        created_at VARCHAR
    );
    """)

    con.close()
    print("DB initialized.")


def upsert_articles(df: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df", df)
    try:
        # BY NAME을 사용하면 컬럼 순서나 누락에 상관없이 이름 기준으로 안전하게 매핑됩니다.
        con.execute("INSERT OR REPLACE INTO articles BY NAME SELECT * FROM df")
    except Exception:
        con.execute("INSERT OR REPLACE INTO articles SELECT * FROM df")
    con.close()


def upsert_chunks(df_chunks: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df_c", df_chunks)
    con.execute("INSERT OR REPLACE INTO article_chunks SELECT * FROM df_c")
    con.close()


# ============================================================
# 시맨틱 검색
# ============================================================
def search_similar_chunks(
    query_vector: list,
    limit: int = 10,
    min_score: float = 0.5,
    dedupe_per_article: bool = True
) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)

    internal_limit = limit * 5 if dedupe_per_article else limit

    sql = """
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
               a.summary_text, a.keywords, a.trust_score, a.trust_verdict,
               list_cosine_similarity(c.embedding, ?::FLOAT[768]) AS score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE list_cosine_similarity(c.embedding, ?::FLOAT[768]) >= ?
        ORDER BY score DESC
        LIMIT ?
    """
    df = con.execute(sql, [query_vector, query_vector, min_score, internal_limit]).fetchdf()
    con.close()

    if df.empty:
        return df

    # 결측치(NaN)로 인한 API 에러 방지
    df = df.fillna({
        "summary_text": "",
        "keywords": "[]",
        "trust_score": 0,
        "trust_verdict": ""
    })

    if dedupe_per_article:
        df = (
            df.sort_values("score", ascending=False)
              .drop_duplicates(subset="article_id", keep="first")
              .head(limit)
              .reset_index(drop=True)
        )

    return df


# ============================================================
# BM25 검색
# ============================================================
def search_bm25(
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    try:
        df_all = con.execute(
            "SELECT article_id, title, source, published_at, full_text, summary_text, keywords, trust_score, trust_verdict FROM articles"
        ).fetchdf()
    except Exception:
        df_all = pd.DataFrame()
    con.close()

    if df_all.empty:
        return pd.DataFrame()

    corpus = [
        _tokenize(f"{row['title']} {row['full_text']}")
        for _, row in df_all.iterrows()
    ]

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    df_all["bm25_score"] = scores

    max_score = df_all["bm25_score"].max()
    if max_score > 0:
        df_all["bm25_score"] = df_all["bm25_score"] / max_score

    df_result = (
        df_all[df_all["bm25_score"] >= 0.7]  # 0.65 → 0.7 상향
        .sort_values("bm25_score", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )

    df_result["chunk_text"] = ""
    
    # 결측치(NaN) 처리
    df_result = df_result.fillna({
        "summary_text": "",
        "keywords": "[]",
        "trust_score": 0,
        "trust_verdict": ""
    })

    return df_result[["article_id", "chunk_text", "title", "source", "published_at", "bm25_score", "summary_text", "keywords", "trust_score", "trust_verdict"]]


# ============================================================
# 하이브리드 검색 (시맨틱 + BM25) - RRF 방식
# ============================================================
def search_hybrid(
    query: str,
    query_vector: list,
    limit: int = 10,
    min_semantic_score: float = 0.5,
    rrf_k: int = 60,
) -> pd.DataFrame:
    df_semantic = search_similar_chunks(
        query_vector,
        limit=limit * 2,
        min_score=min_semantic_score,
        dedupe_per_article=True
    )

    df_bm25 = search_bm25(query, limit=limit * 2)

    rrf_scores = {}

    for rank, (_, row) in enumerate(df_semantic.iterrows(), 1):
        aid = row["article_id"]
        if aid not in rrf_scores:
            rrf_scores[aid] = {
                "article_id":     aid,
                "title":          row["title"],
                "source":         row["source"],
                "published_at":   row["published_at"],
                "chunk_text":     row.get("chunk_text", ""),
                "summary_text":   row.get("summary_text", ""),
                "keywords":       row.get("keywords", "[]"),
                "trust_score":    row.get("trust_score", 0),
                "trust_verdict":  row.get("trust_verdict", ""),
                "semantic_score": row["score"],
                "bm25_score":     0.0,
                "rrf_score":      0.0,
            }
        rrf_scores[aid]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, (_, row) in enumerate(df_bm25.iterrows(), 1):
        aid = row["article_id"]
        if aid not in rrf_scores:
            rrf_scores[aid] = {
                "article_id":     aid,
                "title":          row["title"],
                "source":         row["source"],
                "published_at":   row["published_at"],
                "chunk_text":     "",
                "summary_text":   row.get("summary_text", ""),
                "keywords":       row.get("keywords", "[]"),
                "trust_score":    row.get("trust_score", 0),
                "trust_verdict":  row.get("trust_verdict", ""),
                "semantic_score": 0.0,
                "bm25_score":     row["bm25_score"],
                "rrf_score":      0.0,
            }
        else:
            rrf_scores[aid]["bm25_score"] = row["bm25_score"]
        rrf_scores[aid]["rrf_score"] += 1 / (rrf_k + rank)

    if not rrf_scores:
        return pd.DataFrame()

    df_hybrid = (
        pd.DataFrame(list(rrf_scores.values()))
        .sort_values("rrf_score", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )

    max_rrf = df_hybrid["rrf_score"].max()
    df_hybrid["score"] = df_hybrid["rrf_score"] / max_rrf if max_rrf > 0 else 0.0

    return df_hybrid


# ============================================================
# 관련 기사 검색 (특정 기사 제외)
# ============================================================
def search_similar_chunks_excluding(
    query_vector: list,
    exclude_article_id: str,
    limit: int = 5,
    min_score: float = 0.5
) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    sql = """
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
               a.summary_text, a.keywords, a.trust_score, a.trust_verdict,
               list_cosine_similarity(c.embedding, ?::FLOAT[768]) AS score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE c.article_id != ?
          AND list_cosine_similarity(c.embedding, ?::FLOAT[768]) >= ?
        ORDER BY score DESC
        LIMIT ?
    """
    internal_limit = limit * 5
    df = con.execute(
        sql,
        [query_vector, exclude_article_id, query_vector, min_score, internal_limit]
    ).fetchdf()
    con.close()

    if df.empty:
        return df

    # 결측치(NaN) 처리
    df = df.fillna({
        "summary_text": "",
        "keywords": "[]",
        "trust_score": 0,
        "trust_verdict": ""
    })

    df = (
        df.sort_values("score", ascending=False)
          .drop_duplicates(subset="article_id", keep="first")
          .head(limit)
          .reset_index(drop=True)
    )
    return df


def load_articles() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute("SELECT * FROM articles ORDER BY published_at DESC").fetchdf()
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df


def get_articles_paginated(page: int, size: int) -> list:
    con = duckdb.connect(DB_PATH)
    offset = (page - 1) * size
    try:
        df = con.execute(
            "SELECT * FROM articles ORDER BY published_at DESC LIMIT ? OFFSET ?",
            [size, offset]
        ).fetchdf()
        # Handle NaN values before converting to dict
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception:
        return []
    finally:
        con.close()


def get_article_by_id(article_id: str) -> dict | None:
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute("SELECT * FROM articles WHERE article_id = ?", [article_id]).fetchdf()
        if df.empty:
            return None
        # Handle NaN values
        record = df.fillna("").iloc[0].to_dict()
        return record
    except Exception:
        return None
    finally:
        con.close()


def get_trust_by_article_id(article_id: str) -> dict | None:
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute(
            "SELECT article_id, trust_score, trust_verdict, trust_reason, trust_per_criteria FROM articles WHERE article_id = ?",
            [article_id]
        ).fetchdf()
        if df.empty:
            return None
        
        r = df.iloc[0]
        return {
            "article_id": article_id,
            "trust_score": int(r.get("trust_score", 0) or 0),
            "trust_verdict": str(r.get("trust_verdict", "") or ""),
            "trust_reason": str(r.get("trust_reason", "") or ""),
            "trust_per_criteria": str(r.get("trust_per_criteria", "") or ""),
        }
    except Exception:
        return None
    finally:
        con.close()


def load_articles_without_trust() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute(
            "SELECT * FROM articles WHERE trust_score = 0 ORDER BY published_at DESC"
        ).fetchdf()
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df


def update_article_trust(article_id: str, score: int, verdict: str, reason: str, per_criteria: str):
    con = duckdb.connect(DB_PATH)
    con.execute(
        """UPDATE articles
           SET trust_score = ?, trust_verdict = ?, trust_reason = ?, trust_per_criteria = ?
           WHERE article_id = ?""",
        [score, verdict, reason, per_criteria, article_id],
    )
    con.close()


def delete_articles(article_ids: list):
    """기사 및 관련 청크 삭제"""
    if not article_ids:
        return
    con = duckdb.connect(DB_PATH)
    placeholders = ", ".join(["?" for _ in article_ids])
    con.execute(f"DELETE FROM article_chunks WHERE article_id IN ({placeholders})", article_ids)
    con.execute(f"DELETE FROM articles WHERE article_id IN ({placeholders})", article_ids)
    con.close()


def add_feedback(article_id: str, feedback_type: str):
    con = duckdb.connect(DB_PATH)
    feedback_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    con.execute(
        "INSERT INTO feedback_logs (feedback_id, article_id, feedback_type, created_at) VALUES (?, ?, ?, ?)",
        [feedback_id, article_id, feedback_type, created_at]
    )
    con.close()
    return {"feedback_id": feedback_id, "article_id": article_id, "feedback_type": feedback_type, "created_at": created_at}


def get_all_feedback() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    try:
        # Join with articles table to get the title
        df = con.execute("""
            SELECT 
                f.feedback_id, 
                f.article_id, 
                a.title as article_title, 
                f.feedback_type, 
                f.created_at 
            FROM feedback_logs f
            JOIN articles a ON f.article_id = a.article_id
            ORDER BY f.created_at DESC
        """).fetchdf()
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df