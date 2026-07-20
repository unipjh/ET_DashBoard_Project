import duckdb

def reset_existing_trust_scores():
    # DuckDB 데이터베이스 연결
    con = duckdb.connect("app_db.duckdb")
    
    # 모든 기사의 신뢰도 관련 컬럼을 미분석 상태(0점)로 초기화
    con.execute("""
        UPDATE articles 
        SET trust_score = 0, 
            trust_verdict = 'unanalyzed'
    """)
    con.close()
    print("✅ 기존 DB의 모든 기사 신뢰도 점수가 초기화되었습니다.")

if __name__ == "__main__":
    reset_existing_trust_scores()