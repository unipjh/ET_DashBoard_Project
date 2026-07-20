"""
CSV/NPY → Supabase 가져오기 스크립트

실행 전 준비:
    1. Supabase SQL Editor에서 스키마 생성 완료 (pgvector + 테이블 3개)
    2. .env에 DATABASE_URL 설정
    3. export_from_motherduck.py 실행 완료

실행 방법:
    python scripts/import_to_supabase.py

입력 파일 (프로젝트 루트):
    export_articles.csv
    export_chunks_meta.csv
    export_embeddings.npy
    export_feedback.csv  (있을 경우)
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("❌ DATABASE_URL이 설정되지 않았습니다.")
    sys.exit(1)

print("Supabase 연결 중...")
con = psycopg2.connect(DATABASE_URL)
cur = con.cursor()

# ── articles ──────────────────────────────────────────────
ARTICLES_FILE = "export_articles.csv"
if not os.path.exists(ARTICLES_FILE):
    print(f"❌ {ARTICLES_FILE} 파일이 없습니다. export_from_motherduck.py를 먼저 실행하세요.")
    sys.exit(1)

df = pd.read_csv(ARTICLES_FILE, dtype=str).fillna("")
# trust_score는 INTEGER
if "trust_score" in df.columns:
    df["trust_score"] = pd.to_numeric(df["trust_score"], errors="coerce").fillna(0).astype(int)

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

records = [tuple(r) for r in df.itertuples(index=False)]
psycopg2.extras.execute_values(cur, """
    INSERT INTO articles (
        article_id, title, source, url, published_at,
        full_text, summary_text, keywords, embed_full, embed_summary,
        trust_score, trust_verdict, trust_reason, trust_per_criteria,
        status, category
    ) VALUES %s
    ON CONFLICT (article_id) DO NOTHING
""", records)
con.commit()
print(f"✅ articles: {len(records)}건 완료")

# ── article_chunks + embeddings ───────────────────────────
CHUNKS_FILE = "export_chunks_meta.csv"
EMB_FILE = "export_embeddings.npy"

if not os.path.exists(CHUNKS_FILE) or not os.path.exists(EMB_FILE):
    print(f"❌ {CHUNKS_FILE} 또는 {EMB_FILE} 파일이 없습니다.")
    sys.exit(1)

df_meta = pd.read_csv(CHUNKS_FILE, dtype=str).fillna("")
embeddings = np.load(EMB_FILE)

if len(df_meta) != len(embeddings):
    print(f"❌ chunks({len(df_meta)})와 embeddings({len(embeddings)}) 건수가 다릅니다.")
    sys.exit(1)

print(f"chunks {len(df_meta)}건 삽입 중 (batch=200)...")
BATCH = 200
for i in range(0, len(df_meta), BATCH):
    batch_meta = df_meta.iloc[i:i + BATCH]
    batch_emb = embeddings[i:i + BATCH]
    records = [
        (
            batch_meta.iloc[j]["chunk_id"],
            batch_meta.iloc[j]["article_id"],
            batch_meta.iloc[j]["chunk_text"],
            '[' + ','.join(f'{x:.6f}' for x in batch_emb[j]) + ']',
        )
        for j in range(len(batch_meta))
    ]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO article_chunks (chunk_id, article_id, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO NOTHING
    """, records, template="(%s, %s, %s, %s::vector)")
    con.commit()
    done = min(i + BATCH, len(df_meta))
    print(f"  {done}/{len(df_meta)}")

print(f"✅ chunks 완료")

# ── feedback_logs ─────────────────────────────────────────
FEEDBACK_FILE = "export_feedback.csv"
if os.path.exists(FEEDBACK_FILE):
    df_fb = pd.read_csv(FEEDBACK_FILE, dtype=str).fillna("")
    if not df_fb.empty:
        fb_records = [tuple(r) for r in df_fb.itertuples(index=False)]
        psycopg2.extras.execute_values(cur, """
            INSERT INTO feedback_logs (feedback_id, article_id, feedback_type, created_at)
            VALUES %s
            ON CONFLICT (feedback_id) DO NOTHING
        """, fb_records)
        con.commit()
        print(f"✅ feedback: {len(fb_records)}건 완료")
else:
    print("ℹ️  export_feedback.csv 없음 (건너뜀)")

cur.close()
con.close()
print("\n가져오기 완료.")
