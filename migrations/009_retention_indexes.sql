CREATE INDEX IF NOT EXISTS idx_responses_synced_at ON diagnostic_responses(synced_at);
CREATE INDEX IF NOT EXISTS idx_likert_responses_synced_at ON likert_responses(synced_at);
CREATE INDEX IF NOT EXISTS idx_consents_recorded_at ON consents(recorded_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
