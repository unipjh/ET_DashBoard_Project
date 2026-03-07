import google.generativeai as genai
from Streamlit_Rendering.config import get_gemini_api_key

genai.configure(api_key=get_gemini_api_key())

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768


def run_gemini_embedding(text: str, task_type: str = "retrieval_document") -> list:
    """
    검색 및 관련 기사 벡터 변환 함수
    - 저장 시 (청크 임베딩): task_type="retrieval_document"
    - 검색 시 (검색어/관련기사): task_type="retrieval_query"
    """
    if not text or str(text).strip() == "":
        return [0.0] * EMBEDDING_DIM

    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        embedding = result['embedding']

        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = embedding + [0.0] * (EMBEDDING_DIM - len(embedding))

        return embedding
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return [0.0] * EMBEDDING_DIM