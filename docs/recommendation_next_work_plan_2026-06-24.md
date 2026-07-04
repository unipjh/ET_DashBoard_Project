# 추천 시스템 다음 작업 계획

작성일: 2026-06-24

## 목표

추천 5개가 사용자의 실제 행동에 따라 바뀌는지 확인하고, 이후 attention 기반 추천을 학습/서빙까지 안정적으로 연결한다.

지금 상태의 핵심은 다음과 같다.

- 화면에는 `오늘의 추천 뉴스`와 `당신을 위한 뉴스`가 분리되어 있다.
- `당신을 위한 뉴스`는 `/api/recommendations`를 호출한다.
- 로그인하지 않아도 브라우저 `session_id` 기준으로 추천이 동작한다.
- attention encoder 코드는 들어갔지만, 학습된 가중치가 없으면 기존 임베딩/카테고리/최신순 fallback을 사용한다.
- 따라서 당장 확인할 일은 "학습"보다 "로그가 제대로 쌓이고 추천 API가 그 로그를 읽는지"다.

## 우선순위

1. DB 적재 확인
2. 프론트 행동 로그 확인
3. 추천 API 반영 확인
4. 며칠간 impression/click 데이터 축적
5. attention 학습 실행
6. 학습 결과 export 및 서빙 적용
7. 성능/품질 검증 후 배포

## 1단계: DB 적재 확인

### 확인 목적

`impression`, `view_article_list`, `click_article`, `view_detail` 로그가 `event_logs`에 실제로 쌓이는지 확인한다.

중요한 점:

- `impression`은 새 컬럼이 아니다.
- `event_logs.event_type` 값 중 하나다.
- Supabase Table Editor에서는 `event_type = impression`인 행을 필터링해서 본다.

### Supabase에서 볼 것

Table Editor:

- 테이블: `event_logs`
- 필터: `event_type = impression`
- 확인 컬럼: `session_id`, `user_id`, `event_type`, `article_id`, `event_data`, `created_at`

SQL Editor:

```sql
select event_type, count(*) as count
from event_logs
group by event_type
order by count desc;
```

```sql
select event_type, article_id, session_id, user_id, event_data, created_at
from event_logs
order by created_at desc
limit 30;
```

```sql
select event_data->'articles' as exposed_articles, created_at
from event_logs
where event_type = 'impression'
order by created_at desc
limit 20;
```

### 통과 기준

- 메인 화면 진입 후 `view_article_list`가 생긴다.
- 메인 화면에 기사 목록이 보이면 `impression`이 생긴다.
- 기사 클릭 후 `click_article` 또는 상세 조회 로그가 생긴다.
- `impression`의 `event_data.articles` 안에 여러 기사 ID와 위치가 들어 있다.

## 2단계: 프론트 행동 테스트

### 로컬 실행

백엔드:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

프론트:

```powershell
cd frontend
npm.cmd run dev
```

### 테스트 시나리오

1. 브라우저 개발자 도구를 연다.
2. Network 탭을 켠다.
3. 메인 화면에 접속한다.
4. `/api/logs` 호출이 있는지 본다.
5. `/api/recommendations?...limit=5` 호출이 있는지 본다.
6. `당신을 위한 뉴스`에서 기사 하나를 클릭한다.
7. 상세 페이지로 이동한 뒤 다시 메인으로 돌아온다.
8. `/api/recommendations`가 다시 호출되는지 본다.
9. Supabase `event_logs`에서 방금 행동의 `session_id`가 이어지는지 확인한다.

### 통과 기준

- 메인 진입 시 로그가 1회 이상 쌓인다.
- 같은 화면에서 무한 반복 로그가 생기지 않는다.
- 페이지 이동, 카테고리 이동, 검색 결과 이동 시 `context_key`가 달라진다.
- 상세 페이지를 보고 돌아오면 추천 API가 다시 호출된다.

## 3단계: 추천 API 반영 확인

### 브라우저에서 확인

Network 탭에서 다음 요청을 확인한다.

```text
GET /api/recommendations?session_id=...&user_id=...&limit=5
```

확인할 것:

- `session_id`가 비어 있지 않은지
- 로그인 상태라면 `user_id`가 같이 들어가는지
- 메인 복귀 후 같은 API가 다시 호출되는지
- 응답 5개가 전부 최신순 고정인지, 최근 본 카테고리와 연관되는지

### 직접 호출

```powershell
curl.exe "http://localhost:8000/api/recommendations?session_id=브라우저_SESSION_ID&limit=5"
```

### DB에서 추천 히스토리 확인

```sql
select event_type, article_id, session_id, user_id, created_at
from event_logs
where session_id = '여기에_SESSION_ID'
order by created_at desc
limit 50;
```

### 통과 기준

- 행동 기록이 0개면 최신 기사 중심으로 나온다.
- 행동 기록이 1개 이상이면 추천 계산 흐름에 들어간다.
- 임베딩이 부족해도 최근 본 카테고리 기반 추천이 먼저 시도된다.
- 학습 가중치가 없다는 이유만으로 API가 실패하지 않는다.

## 4단계: negative sample 확인

### 확인 목적

attention 학습에는 "보였지만 클릭하지 않은 기사"가 필요하다.

이 데이터는 다음 조합으로 찾는다.

- `impression`: 사용자에게 보인 기사
- `click_article` 또는 `view_detail`: 사용자가 실제로 누른 기사
- impression에는 있지만 click/view에는 없는 기사: negative sample

### Supabase SQL

```sql
with impression_items as (
  select
    event_logs.session_id,
    item->>'article_id' as article_id,
    event_logs.created_at
  from event_logs,
       jsonb_array_elements((event_logs.event_data::jsonb)->'articles') as item
  where event_type = 'impression'
),
clicked as (
  select distinct session_id, article_id
  from event_logs
  where article_id is not null
    and event_type in ('click_article', 'view_detail', 'view_article_detail')
)
select i.session_id, i.article_id, i.created_at
from impression_items i
left join clicked c
  on c.session_id = i.session_id
 and c.article_id = i.article_id
where c.article_id is null
order by i.created_at desc
limit 50;
```

### 통과 기준

- 결과가 0개가 아니어야 한다.
- 같은 세션에서 impression과 click이 구분되어야 한다.
- 이 데이터가 충분해야 Phase 3 학습이 의미를 가진다.

## 5단계: 학습 준비

### 실행 전 확인

- `event_logs`에 impression이 충분히 쌓였는지
- 클릭/상세조회 로그가 같이 쌓였는지
- `articles.embed_summary`가 비어 있지 않은지
- `articles.learned_embedding` 컬럼이 있는지
- 로컬 학습 환경에 `torch`가 설치되어 있는지

### 권장 데이터 기준

- 최소: 클릭/상세조회 행동 100개 이상
- 권장: impression 1,000개 이상
- 권장: 여러 카테고리에 걸친 클릭 데이터
- 권장: 최소 2~3일 이상 실제 사용 로그

## 6단계: attention 학습 실행

### 학습 명령

```powershell
python -m backend.training.train_user_encoder
```

### 확인할 로그

- epoch마다 loss가 내려가는지
- validation AUC 또는 Recall@10이 출력되는지
- 기존 가중평균 방식보다 성능이 같거나 나은지
- checkpoint가 생성되는지

### 통과 기준

- 학습이 끝까지 실행된다.
- `backend/training/checkpoints/`에 `.pt` 파일이 생긴다.
- 검증 성능이 기존 방식보다 낮지 않다.

## 7단계: 서빙 적용

### 가중치 export

```powershell
python -m backend.training.export_weights
```

### 기존 기사 learned vector 채우기

```powershell
python -m backend.training.backfill_learned_embeddings
```

### 확인할 것

- `backend/services/model_weights/`에 서빙용 가중치가 있는지
- `articles.learned_embedding`이 채워지는지
- `/api/recommendations`가 None/NaN 없이 응답하는지
- 행동 0개, 1개, 20개 초과 모두 정상 응답하는지

## 8단계: 배포 전 검증

### 필수 명령

```powershell
python -m unittest backend.tests.test_recommendation_attention
```

```powershell
cd frontend
npm.cmd run lint
npx.cmd tsc -b
npm.cmd run build
```

### 브라우저 검증

- 메인 화면이 깨지지 않는지
- 오늘의 추천 뉴스가 왼쪽에 보이는지
- 당신을 위한 뉴스 5개가 오른쪽에 보이는지
- 모바일에서 1열로 자연스럽게 내려오는지
- 클릭 후 상세 페이지 이동이 정상인지
- 메인 복귀 후 추천 API가 다시 호출되는지

## 완료 기준

이 작업은 아래 조건이 모두 만족되면 완료로 본다.

- Supabase에서 `impression` 로그를 확인할 수 있다.
- impression-but-not-clicked 데이터를 SQL로 조회할 수 있다.
- `/api/recommendations`가 실제 세션 행동을 읽는다.
- 추천 5개가 행동 후 다시 계산된다.
- 학습 전에도 fallback 추천이 안정적으로 동작한다.
- 학습 후에는 attention 가중치가 로드된다.
- learned embedding backfill이 완료된다.
- 로컬 테스트와 프론트 빌드가 모두 통과한다.

## 당장 다음 액션

1. 로컬 백엔드/프론트를 실행한다.
2. 메인 화면에서 `/api/logs`, `/api/recommendations` 호출을 확인한다.
3. Supabase에서 `event_logs` 최신 30개를 확인한다.
4. 같은 세션으로 기사 3~5개를 클릭한다.
5. 메인 복귀 후 오른쪽 추천 5개가 다시 호출되는지 확인한다.
6. 하루 이상 로그를 쌓은 뒤 negative sample SQL 결과를 확인한다.
7. 데이터가 충분해지면 학습을 실행한다.
