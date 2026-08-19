CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    display_name TEXT NOT NULL,
    extension TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    duration_ms INTEGER,
    storage_kind TEXT NOT NULL CHECK (storage_kind IN ('managed', 'legacy_external')),
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    current_run_id TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    preset_spk_num INTEGER,
    settings_json TEXT NOT NULL,
    model_name TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'waiting_speaker', 'completed', 'failed', 'cancelled', 'interrupted')),
    phase TEXT NOT NULL CHECK (phase IN ('funasr', 'evidence', 'speaker_review', 'cleaned', 'final', 'complete')),
    speaker_mapping_version INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(recording_id, version)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('funasr', 'continuation')),
    queue_seq INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed', 'interrupted', 'force_stopped')),
    phase TEXT NOT NULL CHECK (phase IN ('funasr', 'evidence', 'speaker_review', 'cleaned', 'final')),
    progress_completed INTEGER,
    progress_total INTEGER,
    cancel_requested_at TEXT,
    worker_generation INTEGER,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_per_recording
ON jobs(recording_id) WHERE status IN ('queued', 'running', 'cancel_requested');
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, queue_seq);

CREATE TABLE IF NOT EXISTS stage_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    stage TEXT NOT NULL CHECK (stage IN ('funasr', 'evidence', 'speaker_review', 'cleaned', 'final')),
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    status TEXT NOT NULL CHECK (status IN ('running', 'prepared', 'committed', 'abandoned', 'failed')),
    staging_dir TEXT NOT NULL,
    manifest_path TEXT,
    input_sha256 TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(run_id, stage, attempt_no)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES stage_attempts(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    variant TEXT NOT NULL DEFAULT 'canonical',
    storage_kind TEXT NOT NULL CHECK (storage_kind IN ('managed', 'legacy_external')),
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN ('prepared', 'committed')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(run_id, type, variant),
    UNIQUE(attempt_id, type, variant)
);

CREATE TABLE IF NOT EXISTS speaker_mapping_versions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(run_id, version)
);

CREATE TABLE IF NOT EXISTS speaker_mapping_entries (
    mapping_version_id TEXT NOT NULL REFERENCES speaker_mapping_versions(id) ON DELETE CASCADE,
    anonymous_label TEXT NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY(mapping_version_id, anonymous_label)
);

CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    event TEXT NOT NULL,
    completed INTEGER,
    total INTEGER,
    message TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS task_events_job_idx ON task_events(job_id, id);

CREATE TABLE IF NOT EXISTS deletion_operations (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    storage_kind TEXT NOT NULL CHECK (storage_kind IN ('managed', 'legacy_external')),
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'failed', 'completed')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
