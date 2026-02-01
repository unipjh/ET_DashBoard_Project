# Streamlit_Rendering/summary.py
import re
import torch
import streamlit as st
from kobert_transformers import get_tokenizer, get_kobert_model
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

class FastKoBertSummarizer:
    def __init__(self):
        # GPU 가용 여부에 따라 장치 설정 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. KoBERT 로드 
        self.tokenizer = get_tokenizer()
        self.model = get_kobert_model()
        self.model.to(self.device)
        self.model.eval()

        # 2. KeyBERT 로드 
        st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self.kw_model = KeyBERT(model=st_model)

        # 불용어 설정 
        self.stopwords = [
            '기자', '뉴스', '연합뉴스', '무단전재', '재배포', '금지', '지난', '이번', '이날', '것', '수', '등'
        ]

    def preprocess(self, text):
        """텍스트 정제 """
        if not text: return ""
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>|◀.*?▶', '', text)
        text = re.sub(r'[^ \.\,\?\!\w가-힣]', '', text)
        return text.strip()

@st.cache_resource
def get_summarizer_instance():
    """앱 실행 시 모델을 한 번만 로드하여 캐싱 """
    return FastKoBertSummarizer()

def summarize_text_dummy(text: str, max_chars: int = 50) -> str:
    """기존 코드와의 호환성을 위한 래퍼 함수 """
    from Streamlit_Rendering.admin_pipeline import run_summary
    return run_summary(text)
