# 추천 시스템 플로우차트

> ET_by_claude · 2026-07-03 기준 배포 상태

---

## Chart 1 — Attention Encoder 기반 추천 (Stage 1 · 현재 비활성)

UserEncoder가 사용자 행동 시퀀스를 벡터로 압축해 `learned_embedding` 공간에서 유사 기사를 탐색한다.
`torch` 미설치 환경(Render 무료 플랜)에서는 `is_model_ready()` → `False`이므로 항상 Stage 2로 폴백된다.

```mermaid
flowchart TD
    START([사용자 세션 시작]):::green --> HIST[행동 이력 조회\nevent_logs · get_recent_user_history]
    HIST --> D1{이력 ≥ 1건?}:::amber
    D1 -->|NO| FALL4[Stage 4\n최신 기사 폴백]:::fallback
    D1 -->|YES| ENC[Attention Encoder 상태 확인\nencoder_inference.is_model_ready]
    ENC --> D2{모델 준비됨?}:::amber
    D2 -->|NO\n현재 비활성\nRender 미지원| EMBD[[→ Stage 2 Embedding 기반 폴백]]:::ref
    D2 -->|YES| SEQ[행동 시퀀스 인코딩\nUserEncoder]:::inactive
    SEQ --> VEC[사용자 벡터 생성\nattention_encoders.pt]:::inactive
    VEC --> SRCH[learned_embedding 유사도 검색\narticles 테이블]:::inactive
    SRCH --> D3{결과 있음?}:::amber
    D3 -->|NO| EMBD2[[→ Stage 2 Embedding 기반 폴백]]:::ref
    D3 -->|YES| OUT([개인화 추천 반환]):::gray

    classDef green  fill:#1A3D22,stroke:#3FB950,color:#3FB950
    classDef amber  fill:#231D04,stroke:#E3B341,color:#E3B341
    classDef fallback fill:#1C1900,stroke:#E3B341,color:#E3B341,stroke-dasharray:5 3
    classDef ref    fill:#0C1929,stroke:#58A6FF,color:#58A6FF,stroke-dasharray:5 3
    classDef inactive fill:#1A1A2E,stroke:#F85149,color:#6E7681
    classDef gray   fill:#21262D,stroke:#6E7681,color:#8B949E
```

---

## Chart 2 — 임베딩 기반 추천 (Stage 2 · 현재 활성 / Encoder Fallback)

Gemini `embed_summary` 벡터의 가중 평균으로 사용자 프로필을 구성하고,
pgvector 코사인 유사도로 `article_chunks` 308개를 탐색한다.
Stage 2 실패 시 카테고리(Stage 3) → 최신 기사(Stage 4)로 단계적 폴백한다.

```mermaid
flowchart TD
    ENTRY([Encoder 폴백 진입]):::blue --> COL[히스토리 embed_summary 수집\narticles.embed_summary · 최근 20개]
    COL --> D1{유효 임베딩\n벡터 존재?}:::amber
    D1 -->|NO| S3A[[→ Stage 3 카테고리 폴백]]:::fallback
    D1 -->|YES| PROF[가중 평균 프로필 벡터 생성\nbuild_profile_vector · 최근 기사 가중치 ↑]
    PROF --> PGV[article_chunks 코사인 유사도 검색\npgvector ⟨=⟩ operator · 308개 청크 · min_score=0.5]
    PGV --> D2{유사도 ≥ 0.5\n결과 있음?}:::amber
    D2 -->|NO| S3B[[→ Stage 3 카테고리 폴백]]:::fallback
    D2 -->|YES| DEDUP[기사 단위 중복 제거\n청크 → 기사 dedup · search_similar_chunks]
    DEDUP --> OUT([임베딩 기반 추천 반환]):::green

    subgraph STAGE3 [Stage 3 — 카테고리 기반 폴백]
        CAT[카테고리 분포 분석 · 가중 샘플링\n_fallback_by_history_categories]
        D3{카테고리 결과\n있음?}:::amber
        OUT3([카테고리 기반 추천 반환]):::green
        FALL4([Stage 4 · 최신 기사 폴백]):::gray
        CAT --> D3
        D3 -->|YES| OUT3
        D3 -->|NO| FALL4
    end

    S3A --> CAT
    S3B --> CAT

    classDef blue   fill:#0C2340,stroke:#58A6FF,color:#58A6FF
    classDef amber  fill:#231D04,stroke:#E3B341,color:#E3B341
    classDef fallback fill:#1C1900,stroke:#E3B341,color:#E3B341,stroke-dasharray:5 3
    classDef green  fill:#1A3D22,stroke:#3FB950,color:#3FB950
    classDef gray   fill:#21262D,stroke:#6E7681,color:#8B949E
```

---

## 현재 배포 상태 요약 (2026-07-03)

| Stage | 방식 | 상태 | 근거 |
|-------|------|------|------|
| 1 | Attention Encoder (PyTorch) | ❌ 비활성 | torch 미설치 (Render 무료 플랜) |
| **2** | **임베딩 유사도 (pgvector)** | **✅ 활성** | embed_summary 전체 분석 완료, 청크 308개 |
| 3 | 카테고리 기반 | ✅ 대기 | 카테고리 6종 분류됨 |
| 4 | 최신 기사 | ✅ 항상 | 신규 사용자 폴백 |
