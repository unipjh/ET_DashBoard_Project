import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from backend.services.process_status import update_status
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
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s


def clean_news_content(text):
    """본문 내용 정제 (상세 버전)"""
    if not text:
        return ""
        
    text = text.strip()

    # 1. 맨 앞에 붙어있는 [언론사], [단독], OO뉴스 =, 기자명 등을 반복해서 완벽히 제거
    while True:
        prev_text = text
        text = re.sub(r'^\s*\[제목\]\s*', '', text) # 맨 앞의 [제목] 명시적 제거
        text = re.sub(r'^\s*\[.*?\]\s*', '', text) # 맨 앞의 [블라블라] 제거
        text = re.sub(r'^\s*[가-힣a-zA-Z\s]+(?:일보|신문|뉴스|방송|미디어)\s*(?:=\s*)?', '', text) # 맨 앞의 OO뉴스 = 제거
        text = re.sub(r'^\s*[가-힣]{2,5}\s*기자\s*(?:=\s*)?', '', text) # 맨 앞의 OOO 기자 = 제거 (5글자 허용)
        if text == prev_text:
            break
            
    text = re.sub(r'\s*[가-힣a-zA-Z]+(?:일보|신문|뉴스|방송|미디어)$', '', text)
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
    text = re.sub(r'[●▲▶ⓒⒸ■➔◆▶◀▨▣◈*]', '', text) # "강동·마포" 같은 단어의 가운데 점(·)이 삭제되지 않도록 수정
    text = re.sub(r'\s*[가-힣]{2,5}\s*기자\s*$', '', text) # 맨 끝에 덩그러니 남은 기자명만 안전하게 제거
    text = re.sub(r'(?:\s*#[^\s#]+)+$', '', text)

    text = text.strip()
    if text and not (text.endswith("다.") or text.endswith(".")):
        last_dot = max(text.rfind("다."), text.rfind("."))
        if last_dot != -1:
            end_idx = last_dot + 2 if text[last_dot:last_dot+2] == "다." else last_dot + 1
            text = text[:end_idx]

    return _clean_text(text)


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
        
        like_url = f"https://news.like.naver.com/v1/search/contents?q=NEWS[ne_{oid}_{aid}]"
        like_res = requests.get(like_url, headers=DEFAULT_HEADERS, timeout=5)
        if like_res.status_code == 200:
            stats["like_cnt"] = sum(
                r.get("count", 0) for r in like_res.json()["contents"][0].get("reactions", [])
            )
        
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
    """유사한 기사 제거 (배치 내)"""
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


def remove_db_duplicates(articles: list, db_titles: list, threshold=0.8) -> list:
    """
    DB에 이미 있는 기사 제목과 80% 이상 유사한 기사 제거
    """
    filtered = []
    for article in articles:
        is_dup = False
        for db_title in db_titles:
            if compute_jaccard_similarity(article["title"], db_title) >= threshold:
                update_status("crawl.py", f"⚠️ DB 중복 제거: {article['title'][:40]}")
                is_dup = True
                break
        if not is_dup:
            filtered.append(article)
    return filtered


# ============================================================
# 링크 수집 로직
# ============================================================
def collect_and_convert_links(url, category, max_links=200):
    """연예/스포츠 전용 링크 수집 (네이버 검색 최신순 우회 활용)"""
    links = []
    try:
        # 연예/스포츠 페이지의 동적 렌더링(CSR)을 우회하기 위해 네이버 뉴스 검색 활용
        query = "연예" if category == "연예" else "스포츠"
        for page in range(1, (max_links // 10) + 3):
            start = (page - 1) * 10 + 1
            target_url = f"https://search.naver.com/search.naver?where=news&query={query}&sort=1&start={start}"
            
            res = requests.get(target_url, headers=DEFAULT_HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            
            # 네이버 뉴스 자체 링크만 추출
            for a in soup.select("a.info"):
                href = a.get("href", "")
                if "n.news.naver.com" in href:
                    m = re.search(r'article/(\d+)/(\d+)', href)
                    if m:
                        s_url = f"https://n.news.naver.com/article/{m.group(1)}/{m.group(2)}"
                        if (s_url, category) not in links:
                            links.append((s_url, category))
                
                if len(links) >= max_links:
                    break
            if len(links) >= max_links:
                break
            time.sleep(0.2)
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
# 기사 크롤링
# ============================================================
def crawl_article(url_cat):
    url, category = url_cat
    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        
        source_tag = soup.select_one("em.media_end_head_top_logo img, .logo img, .press_logo img, img.NewsEndMain_press_logo_img__hOOfx")
        source = source_tag.get("alt") if source_tag else "미상"
        if source == "미상":
            source_meta = soup.select_one('meta[property="og:article:author"]')
            source = source_meta.get("content").split("|")[0].strip() if source_meta else "미상"

        title_tag = soup.select_one("h2#title_area, .media_end_head_headline, h2.end_tit, h4.title, .NewsEndMain_article_title__kqEzS, .news_tit, .title_area")
        title = re.sub(r'\[.*?\]', '', title_tag.get_text(strip=True)) if title_tag else None
        if not title or len(title) < 6:
            return None

        date_tag = soup.select_one("span.media_end_head_info_datestamp_time, span.author > em, div.info span, .NewsEndMain_date__J_2A1, .date_time, .info_date")
        dt = None
        if date_tag:
            dt_str = date_tag.get("data-date-time") or date_tag.get_text(strip=True)
            dt = parse_date(dt_str)
        if dt and not (START_DATE <= dt <= END_DATE):
            return None

        content_div = soup.select_one("div._article_content, #dic_area, #articeBody, #newsEndContents, .NewsEndMain_article_content__1We_E, #articleBody, .news_end")
        if not content_div:
            return None
        for j in content_div.select("script, style, .ad_area, .img_desc"):
            j.decompose()
        content = content_div.get_text(" ", strip=True)

        if len(content) < 100:
            return None

        rep_tag = soup.select_one("em.media_end_head_journalist_name, .byline_s, .byline, .author > em, .NewsEndMain_byline__uEmsm, .journalist_name")
        reporter = clean_reporter_name(rep_tag.get_text(strip=True) if rep_tag else "미상")
        
        stats = get_naver_stats(url)

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
# 메인 크롤링 함수
# ============================================================
def fetch_articles_from_naver(max_articles_per_category=30, categories: list = None):
    from backend.services import repo  # 순환 import 방지

    update_status("crawl.py", "🔍 네이버 뉴스 링크 수집 중...")
    all_links = []
    
    for cat, sid in NEWS_CATEGORY.items():
        if categories and cat not in categories:
            continue
            
        if sid in [106, 107]:
            url = f"https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid}"
            all_links.extend(collect_and_convert_links(url, cat, max_articles_per_category))
        else:
            all_links.extend(collect_article_urls(sid, cat, max_articles_per_category))
        time.sleep(0.5)

    if not all_links:
        update_status("crawl.py", "❌ 수집된 링크가 없습니다.")
        return pd.DataFrame()

    update_status("crawl.py", f"📰 {len(all_links)}개 링크 수집 완료. 크롤링 시작...")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(crawl_article, link) for link in all_links]
        for f in tqdm(as_completed(futures), total=len(futures), desc="크롤링"):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        update_status("crawl.py", "❌ 크롤링된 기사가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    update_status("crawl.py", f"✅ {len(df)}개 기사 크롤링 완료")

    # 본문 정제
    update_status("crawl.py", "🧹 본문 노이즈 제거 중...")
    df["content"] = df["content"].apply(clean_news_content)

    # 100자 미만 필터링 (조건 완화)
    df = df[df["content"].str.len() >= 100]

    # 제목 정제 및 중복 제거
    df['title'] = df['title'].str.replace(r'\(풀영상\)', '', regex=True)\
                              .str.replace(r'\[.*?\]', '', regex=True)\
                              .str.strip()
    df['tmp_title_len'] = df['title'].str.len()
    df = df.sort_values(by='tmp_title_len', ascending=False)\
            .drop_duplicates(subset=['content'], keep='first')\
            .drop(columns=['tmp_title_len'])

    # 기자명 필터링 대폭 완화
    df = df[~df["reporter"].str.contains(r'\?', na=False)]

    # 배치 내 유사 기사 제거
    final_data = remove_similar_articles(df.to_dict("records"), threshold=0.7)

    if not final_data:
        update_status("crawl.py", "❌ 필터링 후 남은 기사가 없습니다.")
        return pd.DataFrame()

    # ✅ DB 기존 기사와 중복 제거 (제목 Jaccard 유사도 80% 이상이면 제외)
    update_status("crawl.py", "🗄️ DB 기존 기사와 중복 확인 중...")
    try:
        df_db = repo.load_articles()
        if not df_db.empty:
            db_titles = df_db["title"].dropna().tolist()
            before = len(final_data)
            final_data = remove_db_duplicates(final_data, db_titles, threshold=0.8)
            update_status("crawl.py", f"✅ DB 중복 제거: {before - len(final_data)}개 제거, {len(final_data)}개 남음")
        else:
            update_status("crawl.py", "ℹ️ DB가 비어있어 중복 확인 생략")
    except Exception as e:
        update_status("crawl.py", f"⚠️ DB 중복 확인 실패 (계속 진행): {e}")

    if not final_data:
        update_status("crawl.py", "❌ DB 중복 제거 후 남은 기사가 없습니다.")
        return pd.DataFrame()

    df_final = pd.DataFrame(final_data)
    df_final["reporter"] = df_final["reporter"].apply(
        lambda x: x if "기자" in x else f"{x} 기자"
    )
    update_status("crawl.py", f"✅ 최종 {len(df_final)}개 기사 수집 완료")
    return df_final


# ============================================================
# Backward compatibility
# ============================================================
def fetch_article_from_url(url: str, source: str = "manual", timeout_sec: int = 10) -> pd.DataFrame:
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
            "content": body_text,
        }])
        return df
    except Exception as e:
        print(f"❌ URL 크롤링 실패 ({url}): {e}")
        return pd.DataFrame()