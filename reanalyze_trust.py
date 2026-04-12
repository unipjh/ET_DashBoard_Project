import time
import json
import pandas as pd
from backend.services import repo
from backend.services.trust import score_trust

def reanalyze_all_trust():
    # 1. DB에서 모든 기사 불러오기
    df = repo.load_articles()
    if df.empty:
        print("DB에 기사가 없습니다.")
        return

    total = len(df)
    print(f"총 {total}개의 기사 신뢰도 평가를 새로운 조건(150자 이하 제한)으로 다시 진행합니다...\n")
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        aid = str(row["article_id"])
        text = str(row.get("full_text", "") or "")
        source = str(row.get("source", "미상") or "미상")
        title = str(row.get("title", "") or "")
        
        # 본문이 너무 짧은 기사는 패스
        if len(text.strip()) < 10:
            print(f"[{i}/{total}] ⚠️ 본문이 너무 짧아 건너뜁니다: {title[:30]}")
            continue
            
        print(f"[{i}/{total}] 🔄 신뢰도 재분석 중: {title[:30]}...")
        
        # 2. 바뀐 로직(150자 제한 프롬프트)이 적용된 신뢰도 함수 실행
        result = score_trust(text, source)
        
        # 3. DB에 기존 항목 덮어쓰기
        repo.update_article_trust(
            article_id=aid,
            score=result["score"],
            verdict=result["verdict"],
            reason=result["reason"],
            per_criteria=json.dumps(result["per_criteria"], ensure_ascii=False),
        )
        
        time.sleep(2)  # Gemini API 호출 제한(Rate Limit) 방지

    print("\n✅ 기존 기사들의 신뢰도 재분석 업데이트가 모두 완료되었습니다!")

if __name__ == "__main__":
    reanalyze_all_trust()