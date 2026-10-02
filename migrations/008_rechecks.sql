CREATE TABLE IF NOT EXISTS recheck_sessions (
  id BIGSERIAL PRIMARY KEY,
  original_session_id BIGINT NOT NULL REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
  student_id BIGINT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  question_ids JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_rechecks_student ON recheck_sessions(student_id, started_at DESC);
