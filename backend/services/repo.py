import os
import re
import uuid
import psycopg2
import psycopg2.extras
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

load_dotenv()


def _get_con():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _vec_str(v: list) -> str:
    return '[' + ','.join(f'{x:.6f}' for x in v) + ']'


def _fetchdf(cur) -> pd.DataFrame:
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ============================================================
# 한국어 토크나이저 (공백 + 2글자 이상 단어)
# ============================================================
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[가-힣a-zA-Z0-9]{2,}', text)
    return tokens if tokens else [""]


def init_db():
    con = _get_con()
    cur = con.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            article_id         TEXT PRIMARY KEY,
            title              TEXT,
            source             TEXT,
            url                TEXT,
            published_at       TEXT,
            full_text          TEXT,
            summary_text       TEXT,
            keywords           TEXT,
            embed_full         TEXT,
            embed_summary      TEXT,
            trust_score        INTEGER DEFAULT 0,
            trust_verdict      TEXT,
            trust_reason       TEXT,
            trust_per_criteria TEXT,
            status             TEXT,
            category           TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS article_chunks (
            chunk_id   TEXT PRIMARY KEY,
            article_id TEXT REFERENCES articles(article_id) ON DELETE CASCADE,
            chunk_text TEXT,
            embedding  vector(768)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback_logs (
            feedback_id   TEXT PRIMARY KEY,
            article_id    TEXT REFERENCES articles(article_id) ON DELETE CASCADE,
            user_id       TEXT DEFAULT 'guest',
            feedback_type TEXT,
            created_at    TEXT
        )
    """)
    # 기존 테이블에 user_id 컬럼 추가 (없을 경우에만, 가볍게 처리)
    cur.execute("ALTER TABLE feedback_logs ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'guest'")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            log_id        TEXT PRIMARY KEY,
            session_id    TEXT,
            event_type    TEXT,
            article_id    TEXT,
            event_data    TEXT,
            created_at    TEXT
        )
    """)
    cur.execute("ALTER TABLE event_logs ADD COLUMN IF NOT EXISTS user_id TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    TEXT PRIMARY KEY,
            user_pw    TEXT NOT NULL,
            created_at TEXT
        )
    """)

    con.commit()
    cur.close()
    con.close()
    print("DB initialized.")


def upsert_articles(df: pd.DataFrame):
    cols = [
        "article_id", "title", "source", "url", "published_at",
        "full_text", "summary_text", "keywords", "embed_full", "embed_summary",
        "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
        "status", "category",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0 if c == "trust_score" else ""
    df = df[cols]

    con = _get_con()
    cur = con.cursor()
    records = [tuple(r) for r in df.itertuples(index=False)]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO articles (
            article_id, title, source, url, published_at,
            full_text, summary_text, keywords, embed_full, embed_summary,
            trust_score, trust_verdict, trust_reason, trust_per_criteria,
            status, category
        ) VALUES %s
        ON CONFLICT (article_id) DO UPDATE SET
            title              = EXCLUDED.title,
            source             = EXCLUDED.source,
            url                = EXCLUDED.url,
            published_at       = EXCLUDED.published_at,
            full_text          = EXCLUDED.full_text,
            summary_text       = EXCLUDED.summary_text,
            keywords           = EXCLUDED.keywords,
            trust_score        = EXCLUDED.trust_score,
            trust_verdict      = EXCLUDED.trust_verdict,
            trust_reason       = EXCLUDED.trust_reason,
            trust_per_criteria = EXCLUDED.trust_per_criteria,
            status             = EXCLUDED.status,
            category           = EXCLUDED.category
    """, records)
    con.commit()
    cur.close()
    con.close()


def upsert_chunks(df_chunks: pd.DataFrame):
    con = _get_con()
    cur = con.cursor()
    records = [
        (
            row.chunk_id,
            row.article_id,
            row.chunk_text,
            _vec_str(row.embedding) if isinstance(row.embedding, (list, tuple)) else row.embedding,
        )
        for row in df_chunks.itertuples(index=False)
    ]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO article_chunks (chunk_id, article_id, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding
    """, records, template="(%s, %s, %s, %s::vector)")
    con.commit()
    cur.close()
    con.close()


# ============================================================
# 시맨틱 검색
# ============================================================
def search_similar_chunks(
    query_vector: list,
    limit: int = 10,
    min_score: float = 0.5,
    dedupe_per_article: bool = True,
) -> pd.DataFrame:
    internal_limit = limit * 5 if dedupe_per_article else limit
    vs = _vec_str(query_vector)

    con = _get_con()
    cur = con.cursor()
    cur.execute("""
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
               a.summary_text, a.keywords, a.trust_score, a.trust_verdict,
               1 - (c.embedding <=> %s::vector) AS score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE 1 - (c.embedding <=> %s::vector) >= %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """, [vs, vs, min_score, vs, internal_limit])
    df = _fetchdf(cur)
    cur.close()
    con.close()

    if df.empty:
        return df

    df = df.fillna({"summary_text": "", "keywords": "[]", "trust_score": 0, "trust_verdict": ""})

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
def search_bm25(query: str, limit: int = 10) -> pd.DataFrame:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT article_id, title, source, published_at, full_text, "
            "summary_text, keywords, trust_score, trust_verdict FROM articles"
        )
        df_all = _fetchdf(cur)
    except Exception:
        df_all = pd.DataFrame()
    finally:
        cur.close()
        con.close()

    if df_all.empty:
        return pd.DataFrame()

    corpus = [_tokenize(f"{row['title']} {row['full_text']}") for _, row in df_all.iterrows()]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    df_all["bm25_score"] = scores
    max_score = df_all["bm25_score"].max()
    if max_score > 0:
        df_all["bm25_score"] = df_all["bm25_score"] / max_score

    df_result = (
        df_all[df_all["bm25_score"] >= 0.7]
        .sort_values("bm25_score", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    df_result["chunk_text"] = ""
    df_result = df_result.fillna({"summary_text": "", "keywords": "[]", "trust_score": 0, "trust_verdict": ""})
    return df_result[["article_id", "chunk_text", "title", "source", "published_at",
                       "bm25_score", "summary_text", "keywords", "trust_score", "trust_verdict"]]


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
    df_semantic = search_similar_chunks(query_vector, limit=limit * 2, min_score=min_semantic_score, dedupe_per_article=True)
    df_bm25 = search_bm25(query, limit=limit * 2)

    rrf_scores: dict = {}

    for rank, (_, row) in enumerate(df_semantic.iterrows(), 1):
        aid = row["article_id"]
        if aid not in rrf_scores:
            rrf_scores[aid] = {
                "article_id": aid, "title": row["title"], "source": row["source"],
                "published_at": row["published_at"], "chunk_text": row.get("chunk_text", ""),
                "summary_text": row.get("summary_text", ""), "keywords": row.get("keywords", "[]"),
                "trust_score": row.get("trust_score", 0), "trust_verdict": row.get("trust_verdict", ""),
                "semantic_score": row["score"], "bm25_score": 0.0, "rrf_score": 0.0,
            }
        rrf_scores[aid]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, (_, row) in enumerate(df_bm25.iterrows(), 1):
        aid = row["article_id"]
        if aid not in rrf_scores:
            rrf_scores[aid] = {
                "article_id": aid, "title": row["title"], "source": row["source"],
                "published_at": row["published_at"], "chunk_text": "",
                "summary_text": row.get("summary_text", ""), "keywords": row.get("keywords", "[]"),
                "trust_score": row.get("trust_score", 0), "trust_verdict": row.get("trust_verdict", ""),
                "semantic_score": 0.0, "bm25_score": row["bm25_score"], "rrf_score": 0.0,
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
    min_score: float = 0.5,
) -> pd.DataFrame:
    internal_limit = limit * 5
    vs = _vec_str(query_vector)

    con = _get_con()
    cur = con.cursor()
    cur.execute("""
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
               a.summary_text, a.keywords, a.trust_score, a.trust_verdict,
               1 - (c.embedding <=> %s::vector) AS score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE c.article_id != %s
          AND 1 - (c.embedding <=> %s::vector) >= %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """, [vs, exclude_article_id, vs, min_score, vs, internal_limit])
    df = _fetchdf(cur)
    cur.close()
    con.close()

    if df.empty:
        return df

    df = df.fillna({"summary_text": "", "keywords": "[]", "trust_score": 0, "trust_verdict": ""})
    df = (
        df.sort_values("score", ascending=False)
          .drop_duplicates(subset="article_id", keep="first")
          .head(limit)
          .reset_index(drop=True)
    )
    return df


def load_articles() -> pd.DataFrame:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT * FROM articles ORDER BY published_at DESC")
        df = _fetchdf(cur)
    except Exception:
        df = pd.DataFrame()
    finally:
        cur.close()
        con.close()
    return df


def get_articles_total_count(category: str | None = None) -> int:
    con = _get_con()
    cur = con.cursor()
    try:
        if category:
            cur.execute("SELECT COUNT(*) FROM articles WHERE category = %s", [category])
        else:
            cur.execute("SELECT COUNT(*) FROM articles")
        return cur.fetchone()[0]
    except Exception as e:
        print(f"[DB ERROR] get_articles_total_count: {type(e).__name__}: {e}")
        return 0
    finally:
        cur.close()
        con.close()


def get_articles_paginated(page: int, size: int) -> list:
    offset = (page - 1) * size
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM articles ORDER BY published_at DESC LIMIT %s OFFSET %s",
            [size, offset],
        )
        df = _fetchdf(cur)
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        print(f"[DB ERROR] get_articles_paginated: {type(e).__name__}: {e}")
        return []
    finally:
        cur.close()
        con.close()


def get_articles_paginated_by_category(page: int, size: int, category: str) -> list:
    offset = (page - 1) * size
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM articles WHERE category = %s ORDER BY published_at DESC LIMIT %s OFFSET %s",
            [category, size, offset],
        )
        df = _fetchdf(cur)
        return df.fillna("").to_dict(orient="records")
    except Exception:
        return []
    finally:
        cur.close()
        con.close()


def delete_articles_without_keywords() -> int:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT article_id FROM articles WHERE keywords IS NULL OR keywords = '' OR keywords = '[]'"
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0
        cur.execute("DELETE FROM article_chunks WHERE article_id = ANY(%s)", [ids])
        cur.execute("DELETE FROM articles WHERE article_id = ANY(%s)", [ids])
        con.commit()
        return len(ids)
    except Exception:
        con.rollback()
        return 0
    finally:
        cur.close()
        con.close()


def get_article_by_id(article_id: str) -> dict | None:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT * FROM articles WHERE article_id = %s", [article_id])
        df = _fetchdf(cur)
        if df.empty:
            return None
        return df.fillna("").iloc[0].to_dict()
    except Exception:
        return None
    finally:
        cur.close()
        con.close()


def get_trust_by_article_id(article_id: str) -> dict | None:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT article_id, trust_score, trust_verdict, trust_reason, trust_per_criteria "
            "FROM articles WHERE article_id = %s",
            [article_id],
        )
        df = _fetchdf(cur)
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
        cur.close()
        con.close()


def load_articles_without_trust() -> pd.DataFrame:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT * FROM articles WHERE trust_score = 0 ORDER BY published_at DESC")
        df = _fetchdf(cur)
    except Exception:
        df = pd.DataFrame()
    finally:
        cur.close()
        con.close()
    return df


def update_article_trust(article_id: str, score: int, verdict: str, reason: str, per_criteria: str):
    con = _get_con()
    cur = con.cursor()
    cur.execute(
        "UPDATE articles SET trust_score=%s, trust_verdict=%s, trust_reason=%s, trust_per_criteria=%s "
        "WHERE article_id=%s",
        [score, verdict, reason, per_criteria, article_id],
    )
    con.commit()
    cur.close()
    con.close()


def delete_articles(article_ids: list):
    if not article_ids:
        return
    con = _get_con()
    cur = con.cursor()
    cur.execute("DELETE FROM article_chunks WHERE article_id = ANY(%s)", [article_ids])
    cur.execute("DELETE FROM articles WHERE article_id = ANY(%s)", [article_ids])
    con.commit()
    cur.close()
    con.close()


def add_feedback(article_id: str, user_id: str, feedback_type: str):
    feedback_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    con = _get_con()
    cur = con.cursor()
    # user_id가 users 테이블에 없으면 NULL로 저장 (FK 위반 방지)
    cur.execute("SELECT 1 FROM users WHERE user_id = %s", [user_id])
    safe_user_id = user_id if cur.fetchone() else None
    cur.execute(
        "DELETE FROM feedback_logs WHERE article_id = %s AND user_id IS NOT DISTINCT FROM %s",
        [article_id, safe_user_id],
    )
    cur.execute(
        "INSERT INTO feedback_logs (feedback_id, article_id, user_id, feedback_type, created_at) VALUES (%s, %s, %s, %s, %s)",
        [feedback_id, article_id, safe_user_id, feedback_type, created_at],
    )
    con.commit()
    cur.close()
    con.close()
    return {"feedback_id": feedback_id, "article_id": article_id, "user_id": safe_user_id, "feedback_type": feedback_type, "created_at": created_at}


def get_all_feedback() -> pd.DataFrame:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute("""
            SELECT f.feedback_id, f.article_id, a.title AS article_title,
                   f.feedback_type, f.created_at
            FROM feedback_logs f
            JOIN articles a ON f.article_id = a.article_id
            ORDER BY f.created_at DESC
        """)
        df = _fetchdf(cur)
    except Exception:
        df = pd.DataFrame()
    finally:
        cur.close()
        con.close()
    return df


def create_user(email: str, password: str) -> dict:
    hashed = pwd_context.hash(password)
    created_at = datetime.now().isoformat()
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO users (user_id, user_pw, created_at) VALUES (%s, %s, %s)",
            [email, hashed, created_at],
        )
        con.commit()
        return {"user_id": email, "created_at": created_at}
    except psycopg2.errors.UniqueViolation:
        con.rollback()
        raise ValueError("이미 사용 중인 이메일입니다.")
    finally:
        cur.close()
        con.close()


def authenticate_user(email: str, password: str) -> dict:
    con = _get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT user_id, user_pw FROM users WHERE user_id = %s", [email])
        row = cur.fetchone()
        if not row:
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다.")
        user_id, hashed_pw = row
        if not pwd_context.verify(password, hashed_pw):
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다.")
        return {"user_id": user_id}
    finally:
        cur.close()
        con.close()


def insert_log(event: dict):
    try:
        con = _get_con()
        cur = con.cursor()
        log_id = f"log_{uuid.uuid4().hex}"
        created_at = datetime.now().isoformat()
        event_data_str = json.dumps(event.get('event_data', {}), ensure_ascii=False)
        
        # PostgreSQL 문법인 %s 를 사용합니다.
        raw_user_id = event.get('user_id') or None
        if raw_user_id:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", [raw_user_id])
            safe_user_id = raw_user_id if cur.fetchone() else None
        else:
            safe_user_id = None
        cur.execute(
            "INSERT INTO event_logs (log_id, session_id, event_type, article_id, event_data, created_at, user_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [log_id, event.get('session_id'), event.get('event_type'), event.get('article_id'), event_data_str, created_at, safe_user_id]
        )
        con.commit()
        cur.close()
        con.close()
    except Exception as e:
        print(f"❌ [로그 DB 저장 실패] {e}")
