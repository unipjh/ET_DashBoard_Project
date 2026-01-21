# Streamlit_Rendering/crawl.py
import pandas as pd
from datetime import datetime, timedelta
from Streamlit_Rendering.data import MOCK_DB
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import json

def crawl_latest_articles(limit: int = 100) -> pd.DataFrame:
    """
    네이버 뉴스 최신 기사 크롤러
    - limit: 최종적으로 반환할 기사의 최대 개수
    - 반환 DataFrame 컬럼: 'date', 'category', 'title', 'reporter', 'comment_cnt', 'like_cnt', 'link', 'content'
    """
    
    # --- 설정 영역 ---
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://news.naver.com/"
    }
    NEWS_CATEGORY = {
        "정치": 100, "경제": 101, "사회": 102,
        "생활/문화": 103, "세계": 104, "IT/과학": 105,
        "연예": 106, "스포츠": 107
    }
    MAX_WORKERS = 10
    START_DATE = datetime.now() - timedelta(days=7)
    END_DATE = datetime.now() + timedelta(days=1)
    MEDIA_KEYWORDS = ['신문', '뉴스', '방송', '일보', '연합', '뉴시스', '뉴스1', '경제', 'TV', '데일리', '미디어', '포토', '기자단', '헤럴드', '타임즈']

    # --- 내부 유틸리티 함수 ---
    def _clean_news_content(text):
        if not text: return ""
        text = re.sub(r'^[가-힣]+(?:일보|신문)\s*', '', text)
        text = re.sub(r'\s*[가-힣]+(?:일보|신문)$', '', text)
        text = re.sub(r'(?:이?메일|email|카톡|카카오톡|아이디|ID)\s*[:]?\s*[a-zA-Z0-9._%+-]+(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9_]+)', '', text, flags=re.I)
        text = re.sub(r'[☎※].*?(?=\s{2,}|$)', '', text)
        text = re.sub(r'부고\s*게재\s*문의.*?([0-9\-\s]+|팩스|전화|이메일|카톡|okjebo)+', '', text)
        text = re.sub(r'(?:전화|연락처|문의|팩스|HP|TEL)\s*[:]?\s*\d{2,3}[-\s]\d{3,4}[-\s]\d{4}', '', text)
        text = re.sub(r'△[가-힣]{2,4}\s*=\s*.*?(?=\s*△|\s*$)', '', text)
        text = re.sub(r'(?:트위터|페이스북)\s+[@\w\./\-_]+', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{4}년 \d{1,2}월 \d{1,2}일', '', text)
        text = re.sub(r'<.*?>|【.*?】|\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'[●|]\s*[가-힣]{2,4}\s*기자', '', text)
        text = re.sub(r'[가-힣]{2,4}\s*기자\s*[=|-]\s*', '', text)
        text = re.sub(r'[●▲▶ⓒⒸ■➔◆▶◀▨▣◈*·]', '', text)
        text = re.sub(r'\s*#.*$', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _parse_date(text):
        if not text: return None
        try:
            if "-" in text: return datetime.strptime(text[:16], "%Y-%m-%d %H:%M")
            clean = re.sub(r'[^0-9.: ]', '', text)
            is_pm = "오후" in text
            dt = datetime.strptime(clean.strip(), "%Y.%m.%d. %H:%M")
            if is_pm and dt.hour != 12: dt += timedelta(hours=12)
            return dt
        except: return None

    def _get_naver_stats(url):
        stats = {"comment_cnt": 0, "like_cnt": 0}
        try:
            m = re.search(r'article/(\d+)/(\d+)', url)
            if not m: return stats
            oid, aid = m.group(1), m.group(2)
            # 공감수
            like_url = f"https://news.like.naver.com/v1/search/contents?q=NEWS[ne_{oid}_{aid}]"
            like_res = requests.get(like_url, headers=HEADERS, timeout=5)
            if like_res.status_code == 200:
                stats["like_cnt"] = sum(r.get("count", 0) for r in like_res.json()["contents"][0].get("reactions", []))
            # 댓글수
            comment_url = f"https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&pool=cbox5&lang=ko&objectId=news{oid},{aid}"
            c_res = requests.get(comment_url, headers={"Referer": url}, timeout=5)
            jm = re.search(r'\((.*)\)', c_res.text)
            if jm:
                data = json.loads(jm.group(1))
                if data.get("success"): stats["comment_cnt"] = data["result"]["count"]["comment"]
        except: pass
        return stats

    def _clean_reporter_name(name):
        if not name: return "미상"
        name = re.sub(r'\(.*?\)', '', name).replace("기자", "").strip()
        return name.split()[0] if name.split() else "미상"

    def _compute_jaccard_similarity(t1, t2):
        s1, s2 = set(re.findall(r'\w+', t1)), set(re.findall(r'\w+', t2))
        return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

    # --- 기사 상세 크롤링 함수 ---
    def _crawl_article_detail(url_cat):
        url, category = url_cat
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            
            title_tag = soup.select_one("h2#title_area, .media_end_head_headline")
            title = re.sub(r'\[.*?\]', '', title_tag.get_text(strip=True)) if title_tag else None
            if not title or len(title) < 6: return None

            date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
            dt = _parse_date(date_tag.get("data-date-time")) if date_tag else None
            if dt and not (START_DATE <= dt <= END_DATE): return None

            content_div = soup.select_one("div._article_content, #dic_area")
            if not content_div: return None
            for j in content_div.select("script, style, .ad_area, .img_desc"): j.decompose()
            
            raw_content = content_div.get_text(" ", strip=True)
            cleaned_content = _clean_news_content(raw_content)
            if len(cleaned_content) < 200: return None

            rep_tag = soup.select_one("em.media_end_head_journalist_name, .byline_s")
            reporter = _clean_reporter_name(rep_tag.get_text(strip=True) if rep_tag else "미상")
            stats = _get_naver_stats(url)

            return {
                "date": dt.strftime("%Y-%m-%d %H:%M") if dt else "날짜미상",
                "category": category,
                "title": title,
                "reporter": reporter,
                "comment_cnt": stats["comment_cnt"],
                "like_cnt": stats["like_cnt"],
                "link": url,
                "content": cleaned_content
            }
        except: return None

    # --- 실행 로직 ---
    all_links = []
    # 카테고리별로 기사 링크 수집 (limit의 2배 정도 넉넉하게 수집 후 필터링)
    links_per_cat = max(limit // len(NEWS_CATEGORY) * 2, 20)

    for cat, sid in NEWS_CATEGORY.items():
        page = 1
        cat_links = []
        while len(cat_links) < links_per_cat:
            date_str = datetime.now().strftime("%Y%m%d")
            url = f"https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid}&date={date_str}&page={page}"
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                found = soup.select(".type06_headline a, .type06 a, a[href*='article']")
                if not found: break
                for a in found:
                    href = a.get("href", "")
                    m = re.search(r'article/(\d+)/(\d+)', href) or re.search(r'oid=(\d+)&aid=(\d+)', href)
                    if m:
                        s_url = f"https://n.news.naver.com/article/{m.group(1)}/{m.group(2)}"
                        if (s_url, cat) not in cat_links: cat_links.append((s_url, cat))
                page += 1
                if page > 3: break # 너무 많은 페이지 방지
            except: break
        all_links.extend(cat_links)

    rows = []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(_crawl_article_detail, l) for l in all_links]
        for f in tqdm(as_completed(futures), total=len(futures), desc="뉴스 크롤링 중"):
            r = f.result()
            if r: rows.append(r)

    if not rows: return pd.DataFrame()

    # --- 데이터 후처리 ---
    df = pd.DataFrame(rows)
    # 제목 정제 및 중복 제거
    df['title'] = df['title'].str.replace(r'\(풀영상\)', '', regex=True).str.replace(r'\[.*?\]', '', regex=True).str.strip()
    df = df.drop_duplicates(subset=['content'], keep='first')
    
    # 기자 필터링
    df = df[df["reporter"] != "미상"]
    df = df[~df["reporter"].str.contains(r'\?|[a-zA-Z]|' + "|".join(MEDIA_KEYWORDS), na=False)]
    df = df[(df["reporter"].str.len() >= 2) & (df["reporter"].str.len() < 5)]
    df["reporter"] = df["reporter"].apply(lambda x: x if "기자" in x else f"{x} 기자")

    # 유사 기사 제거 (자카드 유사도 0.7 기준)
    sorted_recs = df.to_dict("records")
    final_recs, dropped = [], set()
    for i in range(len(sorted_recs)):
        if i in dropped: continue
        final_recs.append(sorted_recs[i])
        for j in range(i + 1, len(sorted_recs)):
            if j not in dropped and _compute_jaccard_similarity(sorted_recs[i]["title"], sorted_recs[j]["title"]) >= 0.7:
                dropped.add(j)
        if len(final_recs) >= limit: break

    return pd.DataFrame(final_recs)
