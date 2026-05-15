import time
import pandas as pd
from backend.services import repo
from backend.services.admin_pipeline import run_gemini_summary

def resummarize_all():
    # 1. DB에서 모든 기사 불러오기
    df = repo.load_articles()
    if df.empty:
        print("DB에 기사가 없습니다.")
        return

    total = len(df)
    print(f"총 {total}개의 기사 요약을 새로운 조건으로 다시 진행합니다...\n")
    
    updated_rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row.get("full_text", "") or "")
        title = str(row.get("title", "") or "")
        
        # 본문이 너무 짧은 기사는 패스
        if len(text.strip()) < 10:
            print(f"[{i}/{total}] ⚠️ 본문이 너무 짧아 건너뜁니다: {title[:30]}")
            continue
            
        print(f"[{i}/{total}] 🔄 재요약 중: {title[:30]}...")
        
        # 2. 바뀐 로직이 적용된 요약 함수 실행
        new_summary = run_gemini_summary(text)
        
        updated_row = row.copy()
        updated_row["summary_text"] = new_summary
        updated_rows.append(updated_row.to_dict())
        
        time.sleep(2)  # Gemini API 호출 제한(Rate Limit) 방지
        
        # 3. 10개 단위로 DB에 안전하게 덮어쓰기 저장
        if len(updated_rows) >= 10:
            repo.upsert_articles(pd.DataFrame(updated_rows))
            updated_rows = []

    # 남은 데이터 최종 저장
    if updated_rows:
        repo.upsert_articles(pd.DataFrame(updated_rows))
        
    print("\n✅ 기존 기사들의 재요약 업데이트가 모두 완료되었습니다!")

if __name__ == "__main__":
    resummarize_all()