from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import pandas as pd
import json
from backend.services import repo
from backend.services import encoder_inference
from backend.services.crawl import fetch_articles_from_naver
from backend.services.trust import score_trust
from backend.services.process_status import update_status
import random
import time
from backend.services.config import get_gemini_api_key

GEMINI_API_KEY = get_gemini_api_key()
client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# 🔧 튜닝 포인트 1: 임베딩 모델을 gemini-embedding-001로 통일
#    (search.py와 동일한 모델 사용 → 벡터 공간 일치)
# ============================================================
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768


def run_gemini_summary_and_keywords(text: str) -> tuple[str, str]:
    """Gemini로 요약 및 키워드 추출을 1번의 호출로 동시에 수행"""
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
    
    [응답 JSON 형식]
    반드시 아래 JSON 포맷으로만 출력하세요.
    {{
        "summary": "요약된 본문 텍스트 작성",
        "keywords": ["대분류 > 중분류 > 키워드1", "대분류 > 중분류 > 키워드2", "대분류 > 중분류 > 키워드3", "대분류 > 중분류 > 키워드4", "대분류 > 중분류 > 키워드5"]
    }}
    
    [뉴스 본문]
    {text}
    """

    while True:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.startswith("```"): raw_text = raw_text[3:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            
            # 1. 요약 타입 방어 (문자열이 아니면 기본값 처리)
            summary = data.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                summary = "요약 추출 실패"
                
            # 2. 키워드 타입 방어 (null 방지 및 무조건 리스트 형태 보장)
            raw_keywords = data.get("keywords")
            if not isinstance(raw_keywords, list):
                raw_keywords = [k.strip() for k in str(raw_keywords).split(",")] if raw_keywords else []
                
            keywords = json.dumps(raw_keywords, ensure_ascii=False)
            return summary, keywords
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = random.randint(30, 60)
                update_status("admin_pipeline.py", f"⚠️ Gemini 할당량 초과! {wait_time}초 후 재시도...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                update_status("admin_pipeline.py", f"❌ 요약/키워드 추출 에러: {e}")
                return f"요약 중 에러: {e}", "[]"


def run_gemini_embedding(text: str, task_type: str = "retrieval_document") -> list:
    """
    🔧 튜닝 포인트 2: task_type 파라미터 추가
       - 저장 시: "retrieval_document"
       - 검색 시: "retrieval_query" (함수 호출 측에서 지정)
       → 같은 의미공간에서 문서/쿼리가 최적화됨
    """
    if not text or str(text).strip() == "":
        return [0.0] * EMBEDDING_DIM

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embedding = list(result.embeddings[0].values)

        # 768 차원으로 맞추기
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = embedding + [0.0] * (EMBEDDING_DIM - len(embedding))

        return embedding
    except Exception as e:
        update_status("admin_pipeline.py", f"❌ 임베딩 실패: {e}")
        return [0.0] * EMBEDDING_DIM


def _make_chunk_context(title: str, source: str, category: str, chunk_text: str) -> str:
    """
    🔧 튜닝 포인트 3: Contextual Chunking
       청크에 제목/출처/카테고리를 prefix로 추가.
       → 청크가 어떤 기사의 일부인지 의미 정보가 임베딩에 반영됨
       → 검색 정확도 대폭 향상
    """
    return f"[제목: {title}] [출처: {source}] [카테고리: {category}]\n{chunk_text}"


def build_ready_rows_from_naver(df_raw: pd.DataFrame) -> int:
    """
    크롤링된 기사를 처리하여 DB에 적재 (RAG 튜닝 버전)
    """
    rows, chunk_rows = [], []

    # 🔧 튜닝 포인트 4: 청크 전략 최적화
    #    - chunk_size 400 (너무 크면 의미가 희석됨)
    #    - chunk_overlap 150 (문맥 연속성 강화)
    #    - separators: 한국어 문장 단위로 우선 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150,
        separators=["다.\n", "다. ", ".\n", ". ", "\n\n", "\n", " ", ""]
    )

    total = len(df_raw)
    for idx, r in df_raw.iterrows():
        progress = int((idx + 1) / total * 100)
        update_status("admin_pipeline.py", f"📦 [{progress}%] {idx+1}/{total} - {r.get('title', '')[:40]}")

        full_text = str(r.get("content", ""))
        title = str(r.get("title", ""))
        source = str(r.get("source", "미상"))
        url = str(r.get("link", ""))
        published_at = str(r.get("date", "날짜미상"))
        reporter = str(r.get("reporter", "미상"))
        category = str(r.get("category", "일반"))

        if len(full_text.strip()) < 10:
            update_status("admin_pipeline.py", f"⚠️ {idx+1}번 기사 본문이 비어있어 건너뜁니다.")
            continue

        aid = f"naver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"

        # 1. 요약 및 키워드 추출 (Gemini 호출 1회로 통합)
        update_status("admin_pipeline.py", "  📝 요약 및 키워드 추출 중...")
        summary, keywords = run_gemini_summary_and_keywords(full_text)
        time.sleep(2)

        # 2. 신뢰도 분석 (Gemini 호출 1회)
        update_status("admin_pipeline.py", "  🔍 신뢰도 분석 중...")
        trust = score_trust(full_text, source, title)
        time.sleep(2)

        # 🔧 튜닝 포인트 5: 제목+요약 복합 임베딩을 embed_summary로 저장
        #    → 상세 페이지에서 관련 기사 검색 시 이 벡터를 활용하면 정확도 향상
        title_summary_text = f"{title}\n{summary}"
        summary_embedding = run_gemini_embedding(title_summary_text, task_type="retrieval_document")
        title_embedding = run_gemini_embedding(
            f"[?쒕ぉ] {title}", task_type="retrieval_document"
        )
        learned_embedding = None
        if encoder_inference.is_model_ready():
            learned_embedding = encoder_inference.encode_news({
                "title_embedding": repo._vec_str(title_embedding),
                "embed_summary": repo._vec_str(summary_embedding),
                "category": category,
            })

        rows.append({
            "article_id": aid,
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_at,
            "category": category,
            "full_text": full_text,
            "summary_text": summary,
            "keywords": keywords,
            "embed_full": "[]",
            "embed_summary": repo._vec_str(summary_embedding),
            "learned_embedding": learned_embedding,
            "trust_score":        trust["score"],
            "trust_verdict":      trust["verdict"],
            "trust_reason":       trust["reason"],
            "trust_per_criteria": json.dumps(trust["per_criteria"], ensure_ascii=False),
            "status": "ready"
        })

        # 2. 청크 임베딩 (Contextual Chunking 적용)
        update_status("admin_pipeline.py", "  🔢 청크 임베딩 생성 중...")
        chunks = splitter.split_text(full_text)

        for i, chunk_text in enumerate(chunks):
            time.sleep(0.5)
            # 컨텍스트 prefix를 붙인 텍스트로 임베딩
            contextualized = _make_chunk_context(title, source, category, chunk_text)
            v = run_gemini_embedding(contextualized, task_type="retrieval_document")
            chunk_rows.append({
                "chunk_id": f"{aid}_{i}",
                "article_id": aid,
                "chunk_text": chunk_text,   # 원문 저장 (표시용)
                "embedding": v              # 컨텍스트 포함 임베딩 (검색용)
            })

        # 🔧 튜닝 포인트 6: 제목 전용 청크 추가
        #    제목만으로도 검색이 가능하게 (짧은 키워드 검색에 강함)
        title_embedding = run_gemini_embedding(
            f"[제목] {title}", task_type="retrieval_document"
        )
        chunk_rows.append({
            "chunk_id": f"{aid}_title",
            "article_id": aid,
            "chunk_text": "",  # 화면에 '[제목] ...'이 노출되지 않도록 텍스트를 비움
            "embedding": title_embedding
        })

        time.sleep(3)

    # DB 저장
    if rows:
        update_status("repo.py", f"💾 {len(rows)}개 기사를 DB에 저장 중...")
        repo.upsert_articles(pd.DataFrame(rows))
        for row in rows:
            if row.get("learned_embedding"):
                repo.update_article_learned_embedding(row["article_id"], row["learned_embedding"])

    if chunk_rows:
        update_status("repo.py", f"💾 {len(chunk_rows)}개 청크를 DB에 저장 중...")
        repo.upsert_chunks(pd.DataFrame(chunk_rows))

    update_status("완료", f"✅ 작업 완료! {len(rows)}개 기사 적재됨")
    return len(rows)
