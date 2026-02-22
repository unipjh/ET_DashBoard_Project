import duckdb
import pandas as pd

DB_PATH = "app_db.duckdb"

def init_db():
    con = duckdb.connect(DB_PATH)
    
    # 1. 기사 테이블 생성
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
    
    # 2. 청크 테이블 생성
    con.execute("""
    CREATE TABLE IF NOT EXISTS article_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        article_id VARCHAR,
        chunk_text VARCHAR,
        embedding FLOAT[768]
    );
    """)
    con.close()
    print("✅ DB가 15개 컬럼 구조로 초기화되었습니다.")

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

def search_similar_chunks(query_vector: list, limit: int = 10, min_score: float = 0.3):
    """
    ✅ 유사도 기반 검색 (점수로 필터링)
    
    Args:
        query_vector: 쿼리 임베딩 벡터
        limit: 반환할 최대 개수
        min_score: 최소 유사도 임계값 (0.0~1.0, 기본값 0.3)
    """
    con = duckdb.connect(DB_PATH)
    sql = """
        SELECT c.article_id, c.chunk_text, a.title, a.source,
               list_cosine_similarity(c.embedding, ?::FLOAT[768]) as score
        FROM article_chunks c
        JOIN articles a ON c.article_id = a.article_id
        WHERE list_cosine_similarity(c.embedding, ?::FLOAT[768]) >= ?
        ORDER BY score DESC LIMIT ?
    """
    df = con.execute(sql, [query_vector, query_vector, min_score, limit]).fetchdf()
    con.close()
    return df

def load_articles():
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute("SELECT * FROM articles ORDER BY published_at DESC").fetchdf()
    except:
        df = pd.DataFrame()
    con.close()
    return df
