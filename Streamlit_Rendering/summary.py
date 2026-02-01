# Streamlit_Rendering/summary.py
import re
import torch
import streamlit as st
from kobert_transformers import get_tokenizer, get_kobert_model
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

class FastKoBertSummarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer()
        self.model = get_kobert_model()
        self.model.to(self.device)
        self.model.eval()

        st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self.kw_model = KeyBERT(model=st_model)

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

        # 1. 취소선 및 마크다운 충돌 방지 (모든 마크다운 특수기호 제거)
        text = re.sub(r'[~*_#`\[\]\(\)]', '', text)

        # 2. 웹 보일러플레이트 제거 (Your browser..., 0:00 등)
        text = re.sub(r'Your browser does not support the audio element\.', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{1,2}:\d{2}', '', text) 

        # 3. 이메일, 전화번호, 괄호 텍스트 제거
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)
        text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>|◀.*?▶', '', text)
        
        # 4. 기자명 및 불필요한 단어 제거 (단어 경계 기반으로 더 정확하게)
        remove_words = ['뉴스1', '연합뉴스', '뉴시스', '영상기자', '취재기자', '독자 광고', '광고']
        for word in remove_words:
            text = text.replace(word, '')

        # 5. 한글/숫자/필수 문장부호만 유지
        text = re.sub(r'[^ \.\,\?\!\w가-힣]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

@st.cache_resource
def get_summarizer():
    return FastKoBertSummarizer()

def summarize_text_dummy(text: str, max_chars: int = 50) -> str:
    from Streamlit_Rendering.admin_pipeline import run_summary
    return run_summary(text)
