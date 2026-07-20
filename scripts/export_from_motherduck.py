"""
MotherDuck → CSV/NPY 내보내기 스크립트

실행 방법:
    MOTHERDUCK_TOKEN=<token> python scripts/export_from_motherduck.py
    또는 .env에 MOTHERDUCK_TOKEN 설정 후 실행

출력 파일 (프로젝트 루트):
    export_articles.csv
    export_chunks_meta.csv
    export_embeddings.npy
    export_feedback.csv  (feedback_logs가 있을 경우)
"""

import os
import sys
import numpy as np
import pandas as pd
import duckdb
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MOTHERDUCK_TOKEN", "")
if not token:
    print("❌ MOTHERDUCK_TOKEN이 설정되지 않았습니다.")
    sys.exit(1)

DB_PATH = f"md:et_db?motherduck_token={token}"
print(f"MotherDuck 연결 중...")

con = duckdb.connect(DB_PATH)

# ── articles ──────────────────────────────────────────────
df_articles = con.execute("SELECT * FROM articles").fetchdf()
df_articles.to_csv("export_articles.csv", index=False)
print(f"✅ articles: {len(df_articles)}건 → export_articles.csv")

# ── article_chunks (임베딩 분리 저장) ────────────────────
df_chunks = con.execute(
    "SELECT chunk_id, article_id, chunk_text, embedding FROM article_chunks"
).fetchdf()

embeddings = np.array(df_chunks["embedding"].tolist(), dtype=np.float32)
df_chunks.drop(columns=["embedding"]).to_csv("export_chunks_meta.csv", index=False)
np.save("export_embeddings.npy", embeddings)
print(f"✅ chunks: {len(df_chunks)}건 → export_chunks_meta.csv + export_embeddings.npy")
print(f"   embedding shape: {embeddings.shape}")

# ── feedback_logs ─────────────────────────────────────────
try:
    df_feedback = con.execute("SELECT * FROM feedback_logs").fetchdf()
    df_feedback.to_csv("export_feedback.csv", index=False)
    print(f"✅ feedback: {len(df_feedback)}건 → export_feedback.csv")
except Exception as e:
    print(f"ℹ️  feedback_logs 없음 (무시): {e}")

con.close()
print("\n내보내기 완료.")
