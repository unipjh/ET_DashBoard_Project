# Streamlit_Rendering/summary.py

def summarize_text_dummy(text: str, max_chars: int = 50) -> str:
    """
    더미 요약 함수:
    - 본문을 앞에서 max_chars만 잘라 반환합니다.
    - 실제 요약 모델(BERTSum)로 교체 시, 동일한 인터페이스로 교체하면 됩니다.
    """
    if not text:
        return ""
    s = str(text).strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."

'''
# ================================
# KoBert 기반 추출 요약 모델 구현 예시
# ================================

import re
from collections import Counter

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from kobert_transformers import get_tokenizer, get_kobert_model
from konlpy.tag import Okt


class KoBertExtractiveSummarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = get_tokenizer()
        self.model = get_kobert_model()
        self.model.to(self.device)
        self.model.eval()

        self.tagger = Okt()
        self.stopwords = {
            '기자', '앵커', '연합뉴스', '뉴스', '보도', '사진', '영상', '캡처', '제공', '속보',
            '제보', '전화', '이메일', '홈페이지', '카카오톡', '페이스북', '트위터', '검색', '채널',
            '지난', '이번', '다음', '오후', '오전', '이날', '어제', '오늘', '내일', '현재',
            '시작', '종료', '결과', '이후', '직후', '당시', '최근', '통해', '위해', '관련',
            '대해', '대한', '만큼', '정도', '경우', '때문', '사실', '가장', '내용', '부분',
            '문제', '자신', '생각', '사람', '자체', '주요', '각각', '또한', '다만', '따라',
            '달라', '역시', '모두', '다시', '바로', '더욱', '있다', '없다', '말했다',
            '밝혔다', '전했다', '알렸다', '보인다', '예정', '가능성', '것으로', '하는',
            '있는', '됐다', '했다', '된다'
        }

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)

        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)
        text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>|◀.*?▶', '', text)
        text = re.sub(r'@[a-zA-Z0-9가-힣_]+', '', text)
        text = re.sub(r'\w{2,4} 기자', '', text)
        text = re.sub(r'[^ \.\,\?\!\w가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(768, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return cls_embedding[0].astype(np.float32)

    def extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        nouns = self.tagger.nouns(text)
        nouns = [n for n in nouns if len(n) > 1 and n not in self.stopwords and not n.isdigit()]
        return [w for w, _ in Counter(nouns).most_common(top_n)]

    def summarize(self, clean_text: str, max_sent: int = 3) -> str:
        # 문장 분리(간단 버전)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 20]

        if len(sents) == 0:
            return clean_text
        if len(sents) <= max_sent:
            return " ".join(sents)

        # 문장별 임베딩 → 유사도 합 기반 상위 문장 선택
        sent_embs = np.vstack([self.get_embedding(s) for s in sents])
        sim_matrix = cosine_similarity(sent_embs, sent_embs)
        scores = sim_matrix.sum(axis=1)

        selected_idx = sorted(np.argsort(scores)[::-1][:max_sent])
        return " ".join([sents[i] for i in selected_idx])

    def analyze(self, text: str, max_sent: int = 3, keyword_num: int = 5) -> dict:
        clean_text = self.preprocess(text)

        summary_text = self.summarize(clean_text, max_sent=max_sent)
        keywords = self.extract_keywords(clean_text, top_n=keyword_num)

        embed_full = self.get_embedding(clean_text)
        embed_summary = self.get_embedding(summary_text)

        # JSON 직렬화 가능한 형태로 반환
        return {
            "summary_text": summary_text,
            "keywords": keywords,
            "embed_full": embed_full.tolist(),
            "embed_summary": embed_summary.tolist(),
        }
'''
