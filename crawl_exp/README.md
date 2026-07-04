# crawl_exp — SNU 팩트체크 기반 평가셋 구축 파이프라인

SNU 팩트체크 데이터(GitHub 스크래핑본)에서 실제 기사 본문을 붙여
`eval_trust.py` 용 평가셋을 만드는 실험 파이프라인.

## 실행 순서

```bash
# 1. CSV 다운로드
python crawl_exp/01_fetch_data.py

# 2. source 분류 (크롤링 가능 여부 판정)
python crawl_exp/02_parse_sources.py

# 3. Naver 검색 → 본문 크롤링 (--limit 으로 소량 먼저 테스트 권장)
python crawl_exp/03_search_and_crawl.py --limit 50 --types TYPE_A

# 4. 평가셋 조립
python crawl_exp/04_build_eval.py --per-class 10

# 5. 신뢰도 평가 실행
python -m backend.services.eval_trust --data data/trust_eval_snu.jsonl
```

## 파일 설명

| 파일 | 역할 |
|------|------|
| `01_fetch_data.py` | GitHub에서 `fact_checks_final.csv` 다운로드 → `data/snu_factcheck.csv` |
| `02_parse_sources.py` | source 필드 파싱 → type/outlet/search_query 분류 → `data/snu_classified.csv` |
| `03_search_and_crawl.py` | Naver 뉴스 검색 + 본문 크롤링 → `data/snu_crawled.jsonl` (중단 재시작 지원) |
| `04_build_eval.py` | 레이블별 샘플링 → `data/trust_eval_snu.jsonl` |

## source 분류 기준 (02 단계)

| type | 조건 | 검색 전략 |
|------|------|-----------|
| TYPE_A | 언론사명 + 기사 제목 힌트 모두 있음 | 기사 제목 힌트로 검색 |
| TYPE_B | 언론사명만 있음 | 팩트체크 주장으로 검색 |
| TYPE_C | 언론사 자체 문제제기 | 팩트체크 주장으로 검색 |
| TYPE_D | 국회발언·정부성명 등 | 관련 보도 검색 |
| SKIP | SNS·커뮤니티·유튜브 | 건너뜀 |

## label 변환 기준 (04 단계)

| SNU judge | label |
|-----------|-------|
| 사실 | 1.0 |
| 대체로 사실 | 0.75 |
| 절반의 사실 / 판단 유보 | 0.5 |
| 대체로 사실 아님 | 0.25 |
| 전혀 사실 아님 | 0.0 |

## 주의사항

- Naver 검색은 HTML 파싱 기반 — 구조 변경 시 `03_search_and_crawl.py`의 CSS 셀렉터 수정 필요
- 크롤링 딜레이 기본 1.0초 (`--delay` 조정 가능)
- `snu_crawled.jsonl`은 중단 후 재시작해도 기존 완료 건 자동 스킵
- 저작권: 이 파이프라인은 내부 개발/테스트 목적에 한해 사용
