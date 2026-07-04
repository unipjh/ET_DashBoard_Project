-- 배경 작업 이력 테이블 (서버 재시작 후에도 작업 상태 유지)
CREATE TABLE IF NOT EXISTS background_jobs (
    job_id             TEXT PRIMARY KEY,
    job_type           TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'running',
    current_step       TEXT,
    last_message       TEXT,
    articles_processed INTEGER DEFAULT 0,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ,
    error_text         TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_started_at
ON background_jobs (started_at DESC);
