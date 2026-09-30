CREATE TABLE user_profile (
    user_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    intake_json TEXT NOT NULL,
    preferences_json TEXT NOT NULL,
    clinician_thresholds_json TEXT
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    turn_count INTEGER NOT NULL,
    session_rating INTEGER,
    summary_template_id TEXT,
    task_template_id TEXT
);

CREATE TABLE belief_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    readiness TEXT NOT NULL,
    alliance REAL NOT NULL,
    risk_flag INTEGER NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    type TEXT NOT NULL,
    value INTEGER NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE move_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    move_id TEXT NOT NULL,
    context_vector BLOB,
    reward REAL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    instrument TEXT NOT NULL,
    item_scores TEXT NOT NULL,
    total_score INTEGER NOT NULL,
    timestamp INTEGER NOT NULL
);
