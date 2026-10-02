ALTER TABLE likert_responses
  ADD COLUMN IF NOT EXISTS response_event_id BIGINT;
CREATE TABLE IF NOT EXISTS learning_preferences (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
  likert_session_id BIGINT NOT NULL REFERENCES likert_sessions(id) ON DELETE CASCADE,
  visual_score INTEGER NOT NULL DEFAULT 0,
  auditory_score INTEGER NOT NULL DEFAULT 0,
  reading_writing_score INTEGER NOT NULL DEFAULT 0,
  kinesthetic_score INTEGER NOT NULL DEFAULT 0,
  top_preference VARCHAR(32) NOT NULL,
  second_preference VARCHAR(32) NOT NULL,
  is_mixed BOOLEAN NOT NULL DEFAULT FALSE,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(likert_session_id)
);
CREATE INDEX IF NOT EXISTS idx_learning_preferences_student ON learning_preferences(student_id, computed_at DESC);
