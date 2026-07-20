CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE articles
ADD COLUMN IF NOT EXISTS learned_embedding vector(768);

ALTER TABLE event_logs
ADD COLUMN IF NOT EXISTS user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_event_logs_type_created_at
ON event_logs (event_type, created_at);

CREATE INDEX IF NOT EXISTS idx_event_logs_user_session
ON event_logs (user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_articles_learned_embedding
ON articles USING ivfflat (learned_embedding vector_cosine_ops)
WHERE learned_embedding IS NOT NULL;
