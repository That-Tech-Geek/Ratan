CREATE TABLE IF NOT EXISTS teachers (
  id BIGSERIAL PRIMARY KEY,
  school_id BIGINT NOT NULL REFERENCES schools(id) ON DELETE RESTRICT,
  firebase_uid VARCHAR(128) NOT NULL UNIQUE,
  display_name VARCHAR(200) NOT NULL DEFAULT '',
  role VARCHAR(20) NOT NULL DEFAULT 'teacher' CHECK (role IN ('teacher','volunteer','admin')),
  active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_teachers_school ON teachers(school_id);
CREATE INDEX IF NOT EXISTS idx_teachers_uid ON teachers(firebase_uid);
