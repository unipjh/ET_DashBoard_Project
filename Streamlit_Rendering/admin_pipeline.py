import json
import time
import random
import pandas as pd
import google.generativeai as genai
import concurrent.futures
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from Streamlit_Rendering import repo
from Streamlit_Rendering.crawl import fetch_articles_from_naver
from Streamlit_Rendering.search import run_gemini_embedding
from Streamlit_Rendering.trust import score_trust
from Streamlit_Rendering.config import get_gemini_api_key

genai.configure(api_key=get_gemini_api_key())

def run_gemini_summary(text: str) -> str:
    """Gemini로 본문 길이 맞춤 요약 및 키워드 추출 (JSON 반환)"""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    attempt = 1

    # 1. 본문 길이에 따른 프롬프트 분기
    text_len = len(text)
    if text_len < 500:
        summary_guide = "핵심 위주로 아주 짧게 2문장 내외"
    elif text_len < 1200:
        summary_guide = "중요 내용을 포함하여 3~4문장 내외"
    else:
        summary_guide = "전체 맥락을 파악할 수 있도록 5~6문장 내외"


    prompt = f"""
    당신은 복잡한 뉴스를 핵심만 짚어주는 전문 뉴스 분석가입니다.
    다음 뉴스 본문을 읽고, 내용을 쉽게 파악할 수 있도록 요약하여 JSON 형식으로 응답하세요.
    
    [요약 지침]
    1. 본문 길이에 맞춰 {summary_guide}로 요약할 것.
    2. 수치(날짜, 금액, 퍼센트 등)가 있다면 반드시 포함하여 신뢰도를 높일 것.
    3. 문장의 끝은 '~함', '~임' 형태가 아닌 정중한 대화체(~입니다)를 사용할 것.
    
    [키워드 지침]
    1. 핵심 단어 5개를 반드시 **문자열(String)**로 이루어진 단일 리스트로 추출할 것.
    2. 각 키워드는 "대분류 > 중분류 > 인물/단체 등 소분류" 형태의 텍스트로 작성할 것 (예: ["정치 > 국회 > 홍길동", "경제 > 기업 > 삼성전자"])
    
    [뉴스 본문]
    {text}
"""

    while True:
        try:
            response = model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
                ) # JSON 문자열을 Python 딕셔너리로 변환하여 반환
            return json.loads(response.text)
        
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = random.randint(30, 60)
                print(f"⚠️ 할당량 초과! {wait_time}초 후 재시도...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                print(f"요약 중 에러: {e}")
                # 에러 발생 시 기본값 반환
                return {"summary": f"요약 중 에러 발생: {e}", "keywords": []}


def _make_chunk_context(title: str, source: str, category: str, chunk_text: str) -> str:
    """
    Contextual Chunking
    청크에 제목/출처/카테고리를 prefix로 추가.
    → 청크가 어떤 기사의 일부인지 의미 정보가 임베딩에 반영됨
    """
    return f"[제목: {title}] [출처: {source}] [카테고리: {category}]\n{chunk_text}"


def build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int:
    """
    [최적화 파이프라인]
    Phase 1: 원문 기반 청킹 및 기본 임베딩 즉시 DB 적재 (빠른 저장 및 유실 방지)
    Phase 2: 요약, 신뢰도 분석(병렬 처리) 및 종속 임베딩 생성 후 DB 업데이트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=150,
        separators=["다.\n", "다. ", ".\n", ". ", "\n\n", "\n", " ", ""]
    )

    total = len(df_raw)
    pending_aids = []

    # ==========================================
    # 🚀 Phase 1: 빠른 DB 선적재 (원문 & 청크 임베딩)
    # ==========================================
    print("\n=== [Phase 1] 원문/청크 임베딩 추출 및 DB 선적재 ===")
    for idx, r in df_raw.iterrows():
        progress = int((idx + 1) / total * 100)
        print(f"📦 [Phase 1: {progress}%] {idx+1}/{total} - {r.get('title', '')[:40]}")

        full_text = str(r.get("content", ""))
        if len(full_text.strip()) < 10:
            print(f"⚠️ {idx+1}번 기사 본문이 비어있어 건너뜁니다.")
            continue

        aid = f"naver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
        title = str(r.get("title", ""))
        source = str(r.get("source", "미상"))
        url = str(r.get("link", ""))
        published_at = str(r.get("date", "날짜미상"))
        category = str(r.get("category", "일반"))

        # 1. 원문 임베딩 생성
        embed_full = run_gemini_embedding(full_text, task_type="retrieval_document")
        time.sleep(0.3)

        # 1차 DB Row 구성 (요약/신뢰도는 임시 값으로 대기)
        initial_row = {
            "article_id": aid,
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_at,
            "full_text": full_text,
            "summary_text": "AI 요약 생성 중...",
            "keywords": "[]",
            "embed_full": json.dumps(embed_full),
            "embed_summary": "[]",
            "embed_keywords": "[]",
            "trust_score": 0,
            "trust_verdict": "uncertain",
            "trust_reason": "분석 대기 중",
            "trust_per_criteria": "{}",
            "status": "pending"
        }

        # 2. 청크 임베딩 생성
        chunks = splitter.split_text(full_text)
        chunk_rows = []
        for i, chunk_text in enumerate(chunks):
            contextualized = _make_chunk_context(title, source, category, chunk_text)
            v = run_gemini_embedding(contextualized, task_type="retrieval_document")
            chunk_rows.append({"chunk_id": f"{aid}_{i}", "article_id": aid, "chunk_text": chunk_text, "embedding": v})
            time.sleep(0.3)

        # 제목 전용 청크 추가
        title_embedding = run_gemini_embedding(f"[제목] {title}", task_type="retrieval_document")
        chunk_rows.append({"chunk_id": f"{aid}_title", "article_id": aid, "chunk_text": f"[제목] {title}", "embedding": title_embedding})
        
        # ✨ 즉시 DB 적재 (리스트에 모아두지 않아 중간 에러가 나도 데이터 유실 없음)
        repo.upsert_articles(pd.DataFrame([initial_row]))
        repo.upsert_chunks(pd.DataFrame(chunk_rows))
        pending_aids.append(aid)

    # ==========================================
    # 🧠 Phase 2: 무거운 작업 후처리 및 DB 업데이트 (요약 & 신뢰도)
    # ==========================================
    print(f"\n=== [Phase 2] 요약 및 신뢰도 분석 병렬 처리 ({len(pending_aids)}건) ===")
    
    # 방금 저장한 pending 상태의 기사들 불러오기
    df_pending = repo.load_articles()
    df_pending = df_pending[df_pending['article_id'].isin(pending_aids)]

    for idx, row in df_pending.iterrows():
        aid = row["article_id"]
        full_text = row["full_text"]
        source = row["source"]
        print(f"  ⏳ [Phase 2] 처리 중... {row['title'][:30]}")

        # ✨ 핵심 최적화: ThreadPoolExecutor를 이용해 요약과 신뢰도 평가를 '동시에' 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_summary = executor.submit(run_gemini_summary, full_text)
            future_trust = executor.submit(score_trust, full_text, source)

            gemini_result = future_summary.result()
            trust = future_trust.result()

        summary_text = gemini_result.get("summary", "요약 없음")
        keywords_list = gemini_result.get("keywords", [])

        # 키워드 안전하게 정제
        safe_keywords = []
        if isinstance(keywords_list, list):
            for k in keywords_list:
                if isinstance(k, dict):
                    safe_keywords.append(" > ".join(str(v) for v in k.values()))
                else:
                    safe_keywords.append(str(k))
        else:
            safe_keywords = ["키워드 없음"]

        keywords_str_for_embed = ", ".join(safe_keywords)

        # 3. 요약 및 키워드 임베딩 생성 (결과물이 나온 후 실행)
        embed_summary = run_gemini_embedding(summary_text, task_type="retrieval_document")
        time.sleep(0.5)
        embed_keywords = run_gemini_embedding(keywords_str_for_embed, task_type="retrieval_document")
        time.sleep(0.5)

        # 4. DB 정보 Update를 위한 딕셔너리 재구성
        updated_row = row.to_dict()
        updated_row.update({
            "summary_text": summary_text,
            "keywords": json.dumps(safe_keywords, ensure_ascii=False),
            "embed_summary": json.dumps(embed_summary),
            "embed_keywords": json.dumps(embed_keywords),
            "trust_score": trust["score"],
            "trust_verdict": trust["verdict"],
            "trust_reason": trust["reason"],
            "trust_per_criteria": json.dumps(trust["per_criteria"], ensure_ascii=False),
            "status": "ready"
        })

        # ✨ 수정된 내용으로 단건 DB 덮어쓰기 (INSERT OR REPLACE)
        repo.upsert_articles(pd.DataFrame([updated_row]))
        time.sleep(2) # 다음 기사로 가기 전 API Rate Limit 조절

    print(f"✅ 모든 파이프라인 완료! 총 {len(pending_aids)}개 기사 처리됨")
    return len(pending_aids)
