CREATE TABLE IF NOT EXISTS schema_migrations (
  version VARCHAR(64) PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schools (
  id BIGSERIAL PRIMARY KEY,
  name VARCHAR(200) NOT NULL,
  district VARCHAR(100) NOT NULL,
  block VARCHAR(100) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS students (
  id BIGSERIAL PRIMARY KEY,
  school_id BIGINT NOT NULL REFERENCES schools(id),
  external_id VARCHAR(64) NOT NULL,
  class_no SMALLINT NOT NULL CHECK (class_no IN (8, 9)),
  gender VARCHAR(20) NOT NULL DEFAULT '',
  medium VARCHAR(20) NOT NULL,
  parent_phone_hash VARCHAR(128) NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(school_id, external_id)
);

CREATE TABLE IF NOT EXISTS consents (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES students(id),
  consent_type VARCHAR(32) NOT NULL,
  consent_version VARCHAR(32) NOT NULL,
  granted BOOLEAN NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  withdrawn_at TIMESTAMPTZ,
  UNIQUE(student_id, consent_type, consent_version)
);

CREATE TABLE IF NOT EXISTS diagnostic_sessions (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES students(id),
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  language VARCHAR(5) NOT NULL,
  device_type VARCHAR(32) NOT NULL,
  sync_status VARCHAR(16) NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS diagnostic_responses (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES diagnostic_sessions(id),
  question_id VARCHAR(64) NOT NULL,
  selected_option VARCHAR(128) NOT NULL,
  response_time_ms INTEGER,
  skipped BOOLEAN NOT NULL DEFAULT FALSE,
  synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  sync_event_id BIGINT,
  UNIQUE(session_id, question_id)
);

CREATE TABLE IF NOT EXISTS likert_sessions (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES students(id),
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  language VARCHAR(5) NOT NULL,
  sync_status VARCHAR(16) NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS likert_responses (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES likert_sessions(id),
  item_id VARCHAR(64) NOT NULL,
  score SMALLINT NOT NULL,
  skipped BOOLEAN NOT NULL DEFAULT FALSE,
  synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(session_id, item_id)
);

CREATE TABLE IF NOT EXISTS sync_events (
  id BIGSERIAL PRIMARY KEY,
  idempotency_key UUID NOT NULL UNIQUE,
  entity_type VARCHAR(64) NOT NULL,
  action VARCHAR(16) NOT NULL,
  payload_hash VARCHAR(64) NOT NULL,
  payload JSONB NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
  id BIGSERIAL PRIMARY KEY,
  actor_type VARCHAR(20) NOT NULL,
  actor_id VARCHAR(128) NOT NULL DEFAULT '',
  action VARCHAR(100) NOT NULL,
  entity_type VARCHAR(100) NOT NULL,
  entity_id VARCHAR(64) NOT NULL,
  school_id BIGINT,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_students_school_class ON students(school_id, class_no);
CREATE INDEX IF NOT EXISTS idx_diagnostic_sessions_student ON diagnostic_sessions(student_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_diagnostic_responses_session ON diagnostic_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_sync_events_received ON sync_events(received_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
