# 파일명: Streamlit_Rendering/nlp_engine.py

import torch
import numpy as np
import pandas as pd
import re
import warnings
import json
from sklearn.metrics.pairwise import cosine_similarity
from kobert_transformers import get_tokenizer, get_kobert_model
from collections import Counter
from tqdm import tqdm
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# 경고 메시지 차단
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

class FastKoBertSummarizer:
    def __init__(self):
        """모델 로드 및 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. KoBERT (요약 및 임베딩용)
        try:
            self.tokenizer = get_tokenizer()
            self.model = get_kobert_model()
            self.model.to(self.device)
            self.model.eval()
            print("✅ KoBERT model loaded successfully.")
        except Exception as e:
            print(f"❌ KoBERT 모델 로드 실패: {e}")
            self.model = None

        # 2. KeyBERT (키워드 추출용, GPU 할당)
        print("⏳ Loading KeyBERT model...")
        try:
            st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
            self.kw_model = KeyBERT(model=st_model)
            print(f"✅ KeyBERT model loaded on {self.device}.")
        except Exception as e:
            print(f"❌ KeyBERT 로드 실패: {e}")
            self.kw_model = None

        # 불용어 세트 (Set)
        self.stopwords_set = {
            '기자', '특파원', '앵커', '뉴스', '연합뉴스', '통신', '신문', '보도', '속보', '단독',
            '종합', '취재', '사진', '영상', '캡처', '제공', '자료', '출처', '기사', '편집', '발행',
            '저작권', '무단전재', '재배포', '금지', '구독', '좋아요', '알림', '제보', '문의', '홈페이지',
            '사이트', '링크', '카카오톡', '페이스북', '트위터', '인스타그램', '유튜브', '채널', '검색', '카톡', '라인',
            '앱', '어플', '다운로드', '클릭', '로그인', '회원가입', '전화', '이메일', '뉴스1', 'VJ', '영상기자',
            '지난', '이번', '다음', '이날', '전날', '오늘', '내일', '어제', '현재', '최근', '당시',
            '직후', '이후', '이전', '앞서', '오전', '오후', '새벽', '밤', '낮', '주말', '평일', '연휴',
            '시작', '종료', '예정', '계획', '진행', '과정', '단계', '시점', '시기', '기간', '동안',
            '내년', '올해', '지난해', '작년', '분기', '상반기', '하반기', '결과',
            '말했다', '밝혔다', '전했다', '알렸다', '보인다', '설명했다', '강조했다', '덧붙였다',
            '주장했다', '비판했다', '지적했다', '언급했다', '발표했다', '공개했다', '확인했다',
            '파악됐다', '알려졌다', '나타났다', '기록했다', '풀이된다', '해석된다', '분석된다',
            '전망된다', '예상된다', '관측된다', '보도했다', '인용했다', '제안했다', '요청했다',
            '촉구했다', '지시했다', '합의했다', '결정했다', '확정했다', '추진했다', '검토했다',
            '논의했다', '협의했다', '개최했다', '참석했다', '불참했다', '됐다', '했다', '된다', '있다', '없다',
            '것', '수', '등', '때', '곳', '중', '만', '뿐', '데', '바', '측', '분', '개', '명', '원', '건',
            '위', '점', '면', '채', '식', '편', '만큼', '대로', '관련', '대해', '대한', '위해', '통해',
            '따라', '의해', '인해', '대비', '기준', '정도', '수준', '규모', '비중', '가능성', '필요성',
            '중요성', '문제', '내용', '부분', '분야', '영역', '범위', '대상', '관계', '사이', '상황',
            '여건', '조건', '분위기', '흐름', '추세', '현상', '실태', '현황', '모습', '양상', '형태',
            '구조', '체계', '시스템', '방식', '방법', '수단', '결과', '원인', '이유', '배경', '목적',
            '목표', '의도', '취지', '의미', '역할', '기능', '효과', '영향', '가치', '자신', '생각', '사람',
            '및', '또', '또는', '혹은', '그리고', '그러나', '하지만', '반면', '한편', '게다가',
            '아울러', '더불어', '따라서', '그러므로', '그래서', '결국', '즉', '곧', '다시',
            '특히', '무엇보다', '물론', '실제로', '사실', '대체로', '일반적으로', '주로',
            '가끔', '자주', '항상', '이미', '벌써', '아직', '이제', '지금', '당장', '점차',
            '점점', '갈수록', '더욱', '훨씬', '매우', '아주', '너무', '상당히', '다소', '영상편집', '영상취재',
            '약간', '전혀', '반드시', '오직', '다만', '단지', '오로지', '마치', '결국은',
            '경우', '때문', '가장', '자체', '주요', '각각', '또한', '달라', '역시', '모두', '바로', '것으로', '하는', '있는'
        }
        self.stopwords_list = list({w.lower() for w in self.stopwords_set})

    def _preprocess_text(self, text):
        if not text: return ""
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)
        text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>|◀.*?▶', '', text)
        text = re.sub(r'@[a-zA-Z0-9가-힣_]+', '', text)
        text = re.sub(r'\w{2,4} 기자', '', text)

        remove_words = ['뉴스1', '연합뉴스', '뉴시스', '오마이뉴스', 'KBS', 'SBS', 'MBC', 'VJ', '영상기자', '카톡', '라인', '홈페이지', '영상편집', '영상취재',
                        '영상', '캡처', '제공', '자료', '출처', '기사', '편집', '발행', '구독', '좋아요', '알림', '사이트', '링크', ' .',
                        '기자', '특파원', '앵커', '다운로드', '클릭', '로그인', '회원가입', '카카오톡', '페이스북', '트위터', '인스타그램', '유튜브', '채널']
        for word in remove_words:
            text = text.replace(word, '')
        text = re.sub(r'[^ \.\,\?\!\w가-힣]', '', text)
        return text.strip()

    def get_embedding_batch(self, texts, max_length=128, batch_size=32):
        if not texts or self.model is None:
            return np.zeros((len(texts), 768))

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        if not all_embeddings:
            return np.zeros((0, 768))
        return np.vstack(all_embeddings)

    def _extract_keywords(self, text, top_n=5):
        if not text or self.kw_model is None:
            return []
        try:
            keywords_tuples = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words=self.stopwords_list,
                top_n=top_n,
                use_mmr=True,
                diversity=0.3
            )
            return [kw[0] for kw in keywords_tuples]
        except Exception:
            return []

    def get_trust(self, text):
        # 더미 점수: 실제로는 모델이나 로직으로 계산
        return 85.0 

    def analyze_single(self, text, max_sent=3, keyword_num=5, max_input_sents=30):
        """
        단일 기사 텍스트를 받아 요약, 키워드, 임베딩을 모두 반환
        """
        clean_text = self._preprocess_text(text)
        if not clean_text:
            return "", [], np.zeros(768), np.zeros(768), np.zeros(768), 0.0

        # 1. 키워드 추출
        keywords = self._extract_keywords(clean_text, top_n=keyword_num)
        
        # 2. 본문 임베딩
        content_emb = self.get_embedding_batch([clean_text], max_length=512)[0]
        
        # 3. 키워드 임베딩
        keyword_text = " ".join(keywords) if keywords else ""
        keyword_emb = self.get_embedding_batch([keyword_text], max_length=64)[0] if keyword_text else np.zeros(768)
        
        # 4. 신뢰도
        trust_score = self.get_trust(clean_text)

        # 5. 요약 (중요 문장 추출 방식)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 20]
        sents_to_process = sents[:max_input_sents] if max_input_sents else sents

        if len(sents_to_process) <= max_sent:
            summary = clean_text
        else:
            sent_embs = self.get_embedding_batch(sents_to_process, max_length=128, batch_size=16)
            sim_matrix = cosine_similarity(sent_embs, sent_embs)
            scores = sim_matrix.sum(axis=1)
            top_indices = np.argsort(scores)[::-1][:max_sent]
            selected_idx = sorted(top_indices)
            summary = ' '.join([sents_to_process[i] for i in selected_idx])

        # 6. 요약문 임베딩
        summary_emb = self.get_embedding_batch([summary], max_length=256)[0] if summary else np.zeros(768)

        # 반환 (순서: 요약, 키워드, 본문임베딩, 키워드임베딩, 요약임베딩, 점수)
        return summary, keywords, content_emb, keyword_emb, summary_emb, trust_score

# --- 테스트 실행 코드 (이 파일만 직접 실행했을 때 동작) ---
if __name__ == "__main__":
    print("🚀 NLP Engine Test Mode")
    analyzer = FastKoBertSummarizer()
    test_text = "이것은 테스트 뉴스 기사 본문입니다. AI가 잘 동작하는지 확인합니다."
    summ, kws, c_emb, k_emb, s_emb, score = analyzer.analyze_single(test_text)
    
    print(f"Summary: {summ}")
    print(f"Keywords: {kws}")
    print(f"Trust Score: {score}")
    print(f"Content Emb Shape: {c_emb.shape}")
