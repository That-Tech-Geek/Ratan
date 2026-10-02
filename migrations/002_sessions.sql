ALTER TABLE diagnostic_sessions
  ADD COLUMN IF NOT EXISTS client_session_id UUID UNIQUE,
  ADD COLUMN IF NOT EXISTS session_token_hash VARCHAR(64),
  ADD COLUMN IF NOT EXISTS issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

UPDATE diagnostic_sessions
SET expires_at = COALESCE(expires_at, started_at + INTERVAL '24 hours')
WHERE expires_at IS NULL;

ALTER TABLE diagnostic_sessions
  ALTER COLUMN expires_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_diagnostic_sessions_expires ON diagnostic_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_diagnostic_sessions_client ON diagnostic_sessions(client_session_id);

ALTER TABLE diagnostic_responses
  ADD CONSTRAINT fk_diagnostic_response_sync_event
  FOREIGN KEY (sync_event_id) REFERENCES sync_events(id);
