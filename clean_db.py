import duckdb

DB_PATH = "app_db.duckdb"

def clean_existing_db():
    print("접속 중... (서버가 켜져 있으면 에러가 날 수 있으니 서버를 끄고 실행하세요)")
    con = duckdb.connect(DB_PATH)
    
    print("1. 검색 결과의 '제목 청크' 꼬리표 비우기...")
    con.execute("UPDATE article_chunks SET chunk_text = '' WHERE chunk_id LIKE '%_title'")
    
    print("2. 기존 기사 본문/요약의 맨 앞에 붙은 [언론사명] 및 기사 서두 노이즈 제거...")
    # 대괄호가 여러 번 겹쳐있거나 언론사+기자 이름이 붙어있는 경우를 대비해 2번씩 정제
    for _ in range(2):
        for table_col in [("articles", "full_text"), ("articles", "summary_text"), ("article_chunks", "chunk_text")]:
            table, col = table_col
            # 1. 맨 앞의 [제목] 명시적 제거 (공백 무시)
            con.execute(f"UPDATE {table} SET {col} = regexp_replace({col}, '^\\s*\\[제목\\]\\s*', '');")
            # 2. 맨 앞의 [블라블라] 제거
            con.execute(f"UPDATE {table} SET {col} = regexp_replace({col}, '^\\s*\\[.*?\\]\\s*', '');")
            # 3. 맨 앞의 OO일보, OO뉴스 = 형태 제거
            con.execute(f"UPDATE {table} SET {col} = regexp_replace({col}, '^\\s*[가-힣a-zA-Z\\s]+(?:일보|신문|뉴스|방송|미디어)\\s*(?:=\\s*)?', '');")
        
    print("✅ 기존 DB 데이터 정리가 완료되었습니다!")
    con.close()

if __name__ == "__main__":
    clean_existing_db()
