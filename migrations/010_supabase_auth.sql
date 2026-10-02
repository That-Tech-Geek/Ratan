ALTER TABLE teachers RENAME COLUMN firebase_uid TO auth_user_id;
DROP INDEX IF EXISTS idx_teachers_uid;
CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_auth_user_id ON teachers(auth_user_id);
