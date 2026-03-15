import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_MD_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "")
DB_PATH = f"md:et_db?motherduck_token={_MD_TOKEN}" if _MD_TOKEN else "app_db.duckdb"


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

    con.execute("""
    CREATE TABLE IF NOT EXISTS article_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        article_id VARCHAR,
        chunk_text VARCHAR,
        embedding FLOAT[768]
    );
    """)
    con.close()
    print("DB initialized.")


def upsert_articles(df: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df", df)
    con.execute("INSERT OR REPLACE INTO articles SELECT * FROM df")
    con.close()


def upsert_chunks(df_chunks: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df_c", df_chunks)
    con.execute("INSERT OR REPLACE INTO article_chunks SELECT * FROM df_c")
    con.close()


def search_similar_chunks(
    query_vector: list,
    limit: int = 10,
    min_score: float = 0.5,      # 🔧 기본값 0.7 → 0.5로 낮춤 (더 많은 후보 탐색)
    dedupe_per_article: bool = True
) -> pd.DataFrame:
    """
    🔧 튜닝된 유사도 검색

    변경 사항:
    - min_score 기본값 0.7 → 0.5: 너무 엄격하면 관련 기사를 놓침
    - 내부 limit을 넉넉하게 잡고 dedupe 후 상위 N개 반환
    - article당 best chunk score만 남기는 dedupe 옵션 추가
      → 같은 기사의 여러 청크가 결과를 차지하는 문제 해결
    """
    con = duckdb.connect(DB_PATH)

    # 🔧 내부 limit을 크게 잡아서 dedupe 후에도 충분한 결과 보장
    internal_limit = limit * 5 if dedupe_per_article else limit

    sql = """
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
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

    if dedupe_per_article:
        # 🔧 article_id당 최고 점수 청크만 남김 → 다양한 기사가 결과에 노출됨
        df = (
            df.sort_values("score", ascending=False)
              .drop_duplicates(subset="article_id", keep="first")
              .head(limit)
              .reset_index(drop=True)
        )

    return df


def search_similar_chunks_excluding(
    query_vector: list,
    exclude_article_id: str,
    limit: int = 5,
    min_score: float = 0.5
) -> pd.DataFrame:
    """
    🔧 신규: 특정 기사를 제외한 관련 기사 검색 (상세 페이지 전용)

    기존에는 Python단에서 필터링했으나,
    SQL에서 제외하면 더 정확하게 limit개를 채울 수 있음
    """
    con = duckdb.connect(DB_PATH)
    sql = """
        SELECT c.article_id, c.chunk_text, a.title, a.source, a.published_at,
               list_cosine_similarity(c.embedding, ?::FLOAT[768]) AS score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE c.article_id != ?
          AND list_cosine_similarity(c.embedding, ?::FLOAT[768]) >= ?
        ORDER BY score DESC
        LIMIT ?
    """
    # 내부 limit 넉넉히
    internal_limit = limit * 5
    df = con.execute(
        sql,
        [query_vector, exclude_article_id, query_vector, min_score, internal_limit]
    ).fetchdf()
    con.close()

    if df.empty:
        return df

    # article_id당 best chunk만 남기고 limit개 반환
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


def load_articles_without_trust() -> pd.DataFrame:
    """trust_score=0인 기사만 반환 (일괄 재분석용)."""
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
    """단일 기사의 trust 컬럼만 UPDATE."""
    con = duckdb.connect(DB_PATH)
    con.execute(
        """UPDATE articles
           SET trust_score = ?, trust_verdict = ?, trust_reason = ?, trust_per_criteria = ?
           WHERE article_id = ?""",
        [score, verdict, reason, per_criteria, article_id],
    )
    con.close()