# Streamlit_Rendering/crawl.py
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import json

# ============================================================
# 설정
# ============================================================
DEFAULT_HEADERS = {
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

MEDIA_KEYWORDS = [
    '신문', '뉴스', '방송', '일보', '연합', '뉴시스', '뉴스1',
    '경제', 'TV', '데일리', '미디어', '포토', '기자단',
    '헤럴드', '타임즈'
]


# ============================================================
# 유틸 함수
# ============================================================
def _clean_text(s: str) -> str:
    """
    텍스트 간단 정제 (공백 통합)
    """
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s


def clean_news_content(text):
    """본문 내용 정제 (상세 버전)"""
    if not text:
        return ""
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
    text = re.sub(r'(?:\s*#[^\s#]+)+$', '', text)

    # 추가된 코드: 문장이 "다." 또는 "."으로 끝나지 않으면 완성된 마지막 문장까지만 추출
    text = text.strip()
    if text and not (text.endswith("다.") or text.endswith(".")):
        # 가장 마지막에 등장하는 "다." 또는 "."의 위치를 찾음
        last_dot = max(text.rfind("다."), text.rfind("."))
        if last_dot != -1:
            # "다."인 경우 2글자, "."인 경우 1글자를 포함하여 자름
            end_idx = last_dot + 2 if text[last_dot:last_dot+2] == "다." else last_dot + 1
            text = text[:end_idx]

    return _clean_text(text)  # 최종 공백 정제


def parse_date(text):
    """날짜 파싱"""
    if not text:
        return None
    try:
        if "-" in text:
            return datetime.strptime(text[:16], "%Y-%m-%d %H:%M")
        clean = re.sub(r'[^0-9.: ]', '', text)
        is_pm = "오후" in text
        dt = datetime.strptime(clean.strip(), "%Y.%m.%d. %H:%M")
        if is_pm and dt.hour != 12:
            dt += timedelta(hours=12)
        return dt
    except:
        return None


def get_naver_stats(url):
    """네이버 댓글 수, 추천 수 조회"""
    stats = {"comment_cnt": 0, "like_cnt": 0}
    try:
        m = re.search(r'article/(\d+)/(\d+)', url)
        if not m:
            return stats
        oid, aid = m.group(1), m.group(2)
        
        # 추천 수
        like_url = f"https://news.like.naver.com/v1/search/contents?q=NEWS[ne_{oid}_{aid}]"
        like_res = requests.get(like_url, headers=DEFAULT_HEADERS, timeout=5)
        if like_res.status_code == 200:
            stats["like_cnt"] = sum(
                r.get("count", 0) for r in like_res.json()["contents"][0].get("reactions", [])
            )
        
        # 댓글 수
        comment_url = f"https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&pool=cbox5&lang=ko&objectId=news{oid},{aid}"
        c_res = requests.get(comment_url, headers={"Referer": url}, timeout=5)
        jm = re.search(r'\((.*)\)', c_res.text)
        if jm:
            data = json.loads(jm.group(1))
            if data.get("success"):
                stats["comment_cnt"] = data["result"]["count"]["comment"]
    except:
        pass
    return stats


def clean_reporter_name(name):
    """기자명 정제"""
    if not name:
        return "미상"
    name = re.sub(r'\(.*?\)', '', name).replace("기자", "").strip()
    return name.split()[0] if name.split() else "미상"


def compute_jaccard_similarity(t1, t2):
    """Jaccard 유사도 계산"""
    s1 = set(re.findall(r'\w+', t1))
    s2 = set(re.findall(r'\w+', t2))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0


def remove_similar_articles(articles, threshold=0.7):
    """유사한 기사 제거"""
    # 키값 변경 반영: full_text -> content
    sorted_articles = sorted(articles, key=lambda x: len(x["content"]), reverse=True)
    final, dropped = [], set()
    for i in range(len(sorted_articles)):
        if i in dropped:
            continue
        base = sorted_articles[i]
        final.append(base)
        for j in range(i + 1, len(sorted_articles)):
            if j not in dropped and compute_jaccard_similarity(base["title"], sorted_articles[j]["title"]) >= threshold:
                dropped.add(j)
    return final


# ============================================================
# 링크 수집 로직
# ============================================================
def collect_and_convert_links(url, category, max_links=200):
    """엔터테인먼트/스포츠 링크 수집"""
    links = []
    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.select("a[href*='article']"):
            href = a.get("href", "")
            m = re.search(r'article/(\d+)/(\d+)', href) or re.search(r'oid=(\d+)&aid=(\d+)', href)
            if m:
                s_url = f"https://n.news.naver.com/article/{m.group(1)}/{m.group(2)}"
                if (s_url, category) not in links:
                    links.append((s_url, category))
            if len(links) >= max_links:
                break
    except:
        pass
    return links


def collect_article_urls(sid, category, max_links=200):
    """카테고리별 기사 URL 수집"""
    links, page = [], 1
    date = datetime.now().strftime("%Y%m%d")
    while len(links) < max_links:
        url = f"https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid}&date={date}&page={page}"
        try:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            found = soup.select(".type06_headline a, .type06 a")
            if not found:
                break
            for a in found:
                m = re.search(r'article/(\d+)/(\d+)', a.get("href", ""))
                if m:
                    s_url = f"https://n.news.naver.com/article/{m.group(1)}/{m.group(2)}"
                    if (s_url, category) not in links:
                        links.append((s_url, category))
                if len(links) >= max_links:
                    break
            page += 1
            time.sleep(0.1)
        except:
            break
    return links


# ============================================================
# 기사 크롤링 (본문 200자 필터링) - 수정됨
# ============================================================
def crawl_article(url_cat):
    """
    개별 기사 크롤링 (언론사 추출 로직 강화 및 반환 포맷 수정)
    """
    url, category = url_cat
    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 언론사(source) 추출 로직
        source_tag = soup.select_one("em.media_end_head_top_logo img")
        source = source_tag.get("alt") if source_tag else "미상"
        if source == "미상": # 백업용 추출 방식
            source_meta = soup.select_one('meta[property="og:article:author"]')
            source = source_meta.get("content").split("|")[0].strip() if source_meta else "미상"

        # 제목
        title_tag = soup.select_one("h2#title_area, .media_end_head_headline")
        title = re.sub(r'\[.*?\]', '', title_tag.get_text(strip=True)) if title_tag else None
        if not title or len(title) < 6:
            return None

        # 날짜
        date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
        dt = parse_date(date_tag.get("data-date-time")) if date_tag else None
        if dt and not (START_DATE <= dt <= END_DATE):
            return None

        # 본문
        content_div = soup.select_one("div._article_content, #dic_area")
        if not content_div:
            return None
        for j in content_div.select("script, style, .ad_area, .img_desc"):
            j.decompose()
        content = content_div.get_text(" ", strip=True)

        # 200자 필터링
        if len(content) < 200:
            return None

        # 기자명
        rep_tag = soup.select_one("em.media_end_head_journalist_name, .byline_s")
        reporter = clean_reporter_name(rep_tag.get_text(strip=True) if rep_tag else "미상")
        
        # 댓글, 추천 수
        stats = get_naver_stats(url)

        # 요청하신 반환 포맷으로 수정
        return {
            "date": dt.strftime("%Y-%m-%d %H:%M") if dt else "날짜미상",
            "category": category,
            "source": source,
            "title": title,
            "reporter": reporter,
            "comment_cnt": stats["comment_cnt"],
            "like_cnt": stats["like_cnt"],
            "link": url,
            "content": content
        }
    except:
        return None


# ============================================================
# 메인 크롤링 함수 (admin_pipeline에서 호출)
# ============================================================
def fetch_articles_from_naver(max_articles_per_category=30):
    """
    네이버 뉴스 크롤링 (실 크롤러)
    """
    
    print("🔍 네이버 뉴스 링크 수집 중...")
    all_links = []
    
    for cat, sid in NEWS_CATEGORY.items():
        if sid in [106, 107]:  # 연예, 스포츠
            url = f"https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid}"
            all_links.extend(collect_and_convert_links(url, cat, max_articles_per_category))
        else:
            all_links.extend(collect_article_urls(sid, cat, max_articles_per_category))
        time.sleep(0.5)

    if not all_links:
        print("❌ 수집된 링크가 없습니다.")
        return pd.DataFrame()

    print(f"📰 {len(all_links)}개 링크 수집 완료. 크롤링 시작...")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(crawl_article, link) for link in all_links]
        for f in tqdm(as_completed(futures), total=len(futures), desc="크롤링"):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        print("❌ 크롤링된 기사가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"✅ {len(df)}개 기사 크롤링 완료")

    # 본문 정제
    print("🧹 본문 노이즈 제거 중...")
    df["content"] = df["content"].apply(clean_news_content)

    # 중복 제거
    df['title'] = df['title'].str.replace(r'\(풀영상\)', '', regex=True)\
                              .str.replace(r'\[.*?\]', '', regex=True)\
                              .str.strip()
    df['tmp_title_len'] = df['title'].str.len()
    df = df.sort_values(by='tmp_title_len', ascending=False)\
            .drop_duplicates(subset=['content'], keep='first')\
            .drop(columns=['tmp_title_len'])

    # 기자명 필터링
    df = df[df["reporter"] != "미상"]
    df = df[~df["reporter"].str.contains(r'\?|[a-zA-Z]|' + "|".join(MEDIA_KEYWORDS), na=False)]
    df = df[(df["reporter"].str.len() >= 2) & (df["reporter"].str.len() < 5)]

    # 유사한 기사 제거
    final_data = remove_similar_articles(df.to_dict("records"), threshold=0.7)

    if final_data:
        df_final = pd.DataFrame(final_data)
        df_final["reporter"] = df_final["reporter"].apply(
            lambda x: x if "기자" in x else f"{x} 기자"
        )
        print(f"✅ 최종 {len(df_final)}개 기사 수집 완료")
        return df_final
    else:
        print("❌ 필터링 후 남은 기사가 없습니다.")
        return pd.DataFrame()


# ============================================================
# Backward compatibility (기존 fetch_article_from_url 유지)
# ============================================================
def fetch_article_from_url(url: str, source: str = "manual", timeout_sec: int = 10) -> pd.DataFrame:
    """
    단일 URL 크롤링 (호환성 유지)
    """
    now = datetime.now().isoformat()

    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_sec)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")

        title = _clean_text(
            re.sub(r'\[.*?\]', '', soup.title.get_text().strip())
            if soup.title
            else "Untitled"
        )

        raw_text = _clean_text(soup.get_text(" "))
        body_text = clean_news_content(raw_text)

        if len(body_text) > 4000:
            body_text = body_text[:4000] + "..."

        article_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        df = pd.DataFrame([{
            "article_id": article_id,
            "title": title,
            "source": source,
            "url": url,
            "published_at": now,
            "content": body_text, # 호환성을 위해 키값 content로 통일
        }])
        return df
    except Exception as e:
        print(f"❌ URL 크롤링 실패 ({url}): {e}")
        return pd.DataFrame()
