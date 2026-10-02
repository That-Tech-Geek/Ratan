ALTER TABLE consents
  ADD COLUMN IF NOT EXISTS evidence_ref TEXT;
CREATE TABLE IF NOT EXISTS deletion_queue (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
  queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  processed_at TIMESTAMPTZ,
  reason VARCHAR(64) NOT NULL DEFAULT 'consent_withdrawal'
);
CREATE INDEX IF NOT EXISTS idx_deletion_queue_pending ON deletion_queue(processed_at, queued_at);
