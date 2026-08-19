from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .db import Database
from .state_machine import next_phase, require_job_transition, require_run_transition


ALLOWED_TASK_EVENTS = {
    ("funasr", "started"),
    ("funasr", "completed"),
    ("funasr", "failed"),
    ("evidence", "started"),
    ("evidence", "completed"),
    ("evidence", "failed"),
    ("speaker_review", "started"),
    ("speaker_review", "full_review_started"),
    ("speaker_review", "full_review_completed"),
    ("speaker_review", "risk_segments_started"),
    ("speaker_review", "risk_segment_completed"),
    ("speaker_review", "review_completed"),
    ("speaker_review", "completed"),
    ("speaker_review", "failed"),
    ("cleaned", "started"),
    ("cleaned", "completed"),
    ("cleaned", "failed"),
    ("final", "started"),
    ("final", "chunk_started"),
    ("final", "chunk_completed"),
    ("final", "completed"),
    ("final", "failed"),
}
SENSITIVE_EVENT_TOKENS = ("api_key", "authorization", "base_url", "prompt", "traceback", "url")
ALLOWED_EVENT_DETAIL_KEYS = {"sentence_count", "review_queue_count", "block_count", "fallback_chunk_count"}


@dataclass(frozen=True)
class PreparedArtifact:
    type: str
    relative_path: str
    sha256: str
    size_bytes: int
    variant: str = "canonical"


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def initialize(self) -> None:
        self.database.initialize()

    def recording_by_sha256(self, sha256: str) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM recordings WHERE sha256 = ?", (sha256,)).fetchone()

    def recording(self, recording_id: str) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()

    def create_or_get_recording(
        self,
        *,
        original_filename: str,
        display_name: str,
        extension: str,
        sha256: str,
        size_bytes: int,
        source_path: str,
        duration_ms: int | None = None,
    ) -> tuple[sqlite3.Row, bool]:
        with self.database.transaction(immediate=True) as connection:
            existing = connection.execute("SELECT * FROM recordings WHERE sha256 = ?", (sha256,)).fetchone()
            if existing is not None:
                return existing, False
            recording_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO recordings(id, original_filename, display_name, extension, sha256, size_bytes, duration_ms, storage_kind, source_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (recording_id, original_filename, display_name, extension, sha256, size_bytes, duration_ms, "managed", source_path),
            )
            row = connection.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
            assert row is not None
            return row, True

    def list_recordings(self, *, query: str | None, page: int, page_size: int) -> tuple[list[sqlite3.Row], int]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("分页参数非法")
        pattern = f"%{query.strip()}%" if query and query.strip() else None
        where = "WHERE (? IS NULL OR r.original_filename LIKE ? ESCAPE '\\' OR r.display_name LIKE ? ESCAPE '\\')"
        params = (pattern, pattern, pattern)
        with self.database.connection() as connection:
            total = connection.execute(f"SELECT COUNT(*) FROM recordings AS r {where}", params).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT r.*, run.status AS run_status, run.phase AS run_phase,
                       job.id AS job_id, job.status AS job_status, job.phase AS job_phase,
                       job.progress_completed, job.progress_total, job.error_code, job.error_message,
                       EXISTS(
                           SELECT 1 FROM artifacts AS artifact
                           WHERE artifact.run_id = r.current_run_id
                             AND artifact.type = 'final'
                             AND artifact.status = 'committed'
                       ) AS final_exists
                FROM recordings AS r
                LEFT JOIN runs AS run ON run.id = r.current_run_id
                LEFT JOIN jobs AS job ON job.id = (
                    SELECT candidate.id FROM jobs AS candidate
                    WHERE candidate.run_id = r.current_run_id
                    ORDER BY candidate.queue_seq DESC, candidate.id DESC LIMIT 1
                )
                {where}
                ORDER BY r.created_at DESC, r.id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return rows, total

    def recording_detail(self, recording_id: str) -> tuple[sqlite3.Row, sqlite3.Row | None, sqlite3.Row | None]:
        with self.database.connection() as connection:
            recording = connection.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
            if recording is None:
                raise KeyError(f"录音不存在：{recording_id}")
            run = None
            job = None
            if recording["current_run_id"] is not None:
                run = connection.execute("SELECT * FROM runs WHERE id = ?", (recording["current_run_id"],)).fetchone()
                job = connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ? ORDER BY queue_seq DESC, id DESC LIMIT 1",
                    (recording["current_run_id"],),
                ).fetchone()
            return recording, run, job

    def update_recording_source_path(self, recording_id: str, source_path: str) -> None:
        with self.database.transaction() as connection:
            if connection.execute("UPDATE recordings SET source_path = ? WHERE id = ?", (source_path, recording_id)).rowcount != 1:
                raise KeyError(f"录音不存在：{recording_id}")

    def update_recording_display_name(self, recording_id: str, display_name: str) -> sqlite3.Row:
        with self.database.transaction() as connection:
            if connection.execute("UPDATE recordings SET display_name = ? WHERE id = ?", (display_name, recording_id)).rowcount != 1:
                raise KeyError(f"录音不存在：{recording_id}")
            recording = connection.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
            assert recording is not None
            return recording

    def create_run_and_enqueue_funasr(
        self,
        *,
        recording_id: str,
        preset_spk_num: int | None,
        settings_snapshot: dict[str, Any],
        model_name: str | None = None,
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        if preset_spk_num is not None and (isinstance(preset_spk_num, bool) or not isinstance(preset_spk_num, int) or preset_spk_num <= 0):
            raise ValueError("preset_spk_num 必须为 null 或正整数")
        with self.database.transaction(immediate=True) as connection:
            recording = connection.execute("SELECT storage_kind FROM recordings WHERE id = ?", (recording_id,)).fetchone()
            if recording is None:
                raise KeyError(f"录音不存在：{recording_id}")
            if recording["storage_kind"] != "managed":
                raise ValueError("录音不是受控录音")
            active = connection.execute(
                "SELECT id FROM jobs WHERE recording_id = ? AND status IN ('queued', 'running', 'cancel_requested')",
                (recording_id,),
            ).fetchone()
            if active is not None:
                raise ValueError(f"录音已有活动任务：{active['id']}")
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM runs WHERE recording_id = ?", (recording_id,)
            ).fetchone()[0]
            run_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO runs(id, recording_id, version, preset_spk_num, settings_json, model_name, status, phase)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', 'funasr')
                """,
                (run_id, recording_id, version, preset_spk_num, json.dumps(settings_snapshot, ensure_ascii=False, sort_keys=True), model_name),
            )
            connection.execute("UPDATE recordings SET current_run_id = ? WHERE id = ?", (run_id, recording_id))
            queue_seq = connection.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM jobs").fetchone()[0]
            job_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO jobs(id, recording_id, run_id, kind, queue_seq, status, phase)
                VALUES (?, ?, ?, 'funasr', ?, 'queued', 'funasr')
                """,
                (job_id, recording_id, run_id, queue_seq),
            )
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert run is not None and job is not None
            return run, job

    def create_runs_and_enqueue_funasr(
        self,
        *,
        recording_ids: list[str],
        preset_spk_num: int | None,
        settings_snapshot: dict[str, Any],
    ) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
        if not recording_ids or len(recording_ids) != len(set(recording_ids)):
            raise ValueError("批量录音列表非法")
        if preset_spk_num is not None and (isinstance(preset_spk_num, bool) or not isinstance(preset_spk_num, int) or preset_spk_num <= 0):
            raise ValueError("preset_spk_num 必须为 null 或正整数")
        with self.database.transaction(immediate=True) as connection:
            placeholders = ", ".join("?" for _ in recording_ids)
            found = connection.execute(f"SELECT id, storage_kind FROM recordings WHERE id IN ({placeholders})", recording_ids).fetchall()
            if len(found) != len(recording_ids):
                raise KeyError("批量请求包含不存在的录音")
            if any(recording["storage_kind"] != "managed" for recording in found):
                raise ValueError("录音不是受控录音")
            active = connection.execute(
                f"SELECT recording_id FROM jobs WHERE recording_id IN ({placeholders}) AND status IN ('queued', 'running', 'cancel_requested')",
                recording_ids,
            ).fetchone()
            if active is not None:
                raise ValueError("批量请求包含活动任务")
            queue_seq = connection.execute("SELECT COALESCE(MAX(queue_seq), 0) FROM jobs").fetchone()[0]
            result = []
            for recording_id in recording_ids:
                version = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM runs WHERE recording_id = ?", (recording_id,)
                ).fetchone()[0]
                run_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
                queue_seq += 1
                connection.execute(
                    "INSERT INTO runs(id, recording_id, version, preset_spk_num, settings_json, status, phase) VALUES (?, ?, ?, ?, ?, 'queued', 'funasr')",
                    (run_id, recording_id, version, preset_spk_num, json.dumps(settings_snapshot, ensure_ascii=False, sort_keys=True)),
                )
                connection.execute("UPDATE recordings SET current_run_id = ? WHERE id = ?", (run_id, recording_id))
                connection.execute(
                    "INSERT INTO jobs(id, recording_id, run_id, kind, queue_seq, status, phase) VALUES (?, ?, ?, 'funasr', ?, 'queued', 'funasr')",
                    (job_id, recording_id, run_id, queue_seq),
                )
                result.append((
                    connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone(),
                    connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone(),
                ))
            return result

    def queued_jobs(self) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(
                """
                SELECT jobs.*, recordings.display_name, recordings.original_filename
                FROM jobs
                JOIN recordings ON recordings.id = jobs.recording_id
                WHERE jobs.status = 'queued'
                ORDER BY jobs.queue_seq, jobs.id
                """
            ).fetchall()

    def reorder_queued_jobs(self, job_ids: list[str]) -> list[sqlite3.Row]:
        if not job_ids or len(job_ids) != len(set(job_ids)):
            raise ValueError("队列顺序非法")
        with self.database.transaction(immediate=True) as connection:
            queued = connection.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_seq, id").fetchall()
            if {job["id"] for job in queued} != set(job_ids):
                raise ValueError("队列必须包含全部且仅包含未开始任务")
            for index, job_id in enumerate(job_ids, start=1):
                connection.execute("UPDATE jobs SET queue_seq = ? WHERE id = ?", (-index, job_id))
            base = max(0, connection.execute("SELECT COALESCE(MAX(queue_seq), 0) FROM jobs").fetchone()[0])
            for index, job_id in enumerate(job_ids, start=1):
                connection.execute(
                    "UPDATE jobs SET queue_seq = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                    (base + index, job_id),
                )
            return [connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone() for job_id in job_ids]

    def create_run(
        self,
        *,
        recording_id: str,
        preset_spk_num: int | None,
        settings_snapshot: dict[str, Any],
        model_name: str | None = None,
    ) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM recordings WHERE id = ?", (recording_id,)).fetchone() is None:
                raise KeyError(f"录音不存在：{recording_id}")
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM runs WHERE recording_id = ?", (recording_id,)
            ).fetchone()[0]
            run_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO runs(id, recording_id, version, preset_spk_num, settings_json, model_name, status, phase)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', 'funasr')
                """,
                (run_id, recording_id, version, preset_spk_num, json.dumps(settings_snapshot, ensure_ascii=False, sort_keys=True), model_name),
            )
            connection.execute("UPDATE recordings SET current_run_id = ? WHERE id = ?", (run_id, recording_id))
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            assert row is not None
            return row

    def run(self, run_id: str) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def enqueue_job(self, *, recording_id: str, run_id: str, kind: str, phase: str) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            active = connection.execute(
                "SELECT id FROM jobs WHERE recording_id = ? AND status IN ('queued', 'running', 'cancel_requested')", (recording_id,)
            ).fetchone()
            if active is not None:
                raise ValueError(f"录音已有活动任务：{active['id']}")
            queue_seq = connection.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM jobs").fetchone()[0]
            job_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO jobs(id, recording_id, run_id, kind, queue_seq, status, phase) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (job_id, recording_id, run_id, kind, queue_seq, phase),
            )
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert row is not None
            return row

    def job(self, job_id: str) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def claim_next_job(self, worker_generation: int) -> sqlite3.Row | None:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_seq, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            changed = connection.execute(
                """
                UPDATE jobs
                SET status = 'running', worker_generation = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ? AND status = 'queued'
                """,
                (worker_generation, row["id"]),
            ).rowcount
            if changed != 1:
                return None
            connection.execute(
                "UPDATE runs SET status = 'running', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (row["run_id"],),
            )
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()

    def transition_job(self, job_id: str, target: str, *, error_code: str | None = None, error_message: str | None = None) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"任务不存在：{job_id}")
            require_job_transition(row["status"], target)
            connection.execute(
                """
                UPDATE jobs SET status = ?, error_code = ?, error_message = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (target, error_code, error_message, job_id),
            )
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def complete_initial_job_and_enqueue_continuation(self, job_id: str) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"任务不存在：{job_id}")
            if job["kind"] != "funasr":
                raise ValueError("只有 FunASR 任务可以自动排入后续处理")
            require_job_transition(job["status"], "succeeded")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (job["run_id"],)).fetchone()
            if run is None:
                raise KeyError(f"运行不存在：{job['run_id']}")
            require_run_transition(run["status"], "queued")
            connection.execute(
                "UPDATE jobs SET status = 'succeeded', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (job_id,),
            )
            connection.execute(
                "UPDATE runs SET status = 'queued', phase = 'speaker_review', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (run["id"],),
            )
            queue_seq = connection.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM jobs").fetchone()[0]
            continuation_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO jobs(id, recording_id, run_id, kind, queue_seq, status, phase)
                VALUES (?, ?, ?, 'continuation', ?, 'queued', 'speaker_review')
                """,
                (continuation_id, job["recording_id"], run["id"], queue_seq),
            )
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (continuation_id,)).fetchone()

    def set_job_phase(self, job_id: str, phase: str) -> None:
        with self.database.transaction() as connection:
            if connection.execute(
                "UPDATE jobs SET phase = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (phase, job_id),
            ).rowcount != 1:
                raise KeyError(f"任务不存在：{job_id}")

    def running_jobs(self, worker_generation: int | None = None) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            if worker_generation is None:
                return connection.execute("SELECT * FROM jobs WHERE status IN ('running', 'cancel_requested')").fetchall()
            return connection.execute(
                "SELECT * FROM jobs WHERE status IN ('running', 'cancel_requested') AND worker_generation = ?",
                (worker_generation,),
            ).fetchall()

    def request_cancel(self, job_id: str) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"任务不存在：{job_id}")
            if row["status"] in {"cancel_requested", "cancelled"}:
                return row
            target = "cancelled" if row["status"] == "queued" else "cancel_requested"
            require_job_transition(row["status"], target)
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, cancel_requested_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (target, job_id),
            )
            if target == "cancelled":
                run = connection.execute("SELECT * FROM runs WHERE id = ?", (row["run_id"],)).fetchone()
                if run is None:
                    raise KeyError(f"运行不存在：{row['run_id']}")
                require_run_transition(run["status"], "cancelled")
                connection.execute(
                    "UPDATE runs SET status = 'cancelled', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                    (run["id"],),
                )
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def recovery_recommendation(self, job_id: str):
        with self.database.connection() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"任务不存在：{job_id}")
            artifacts = connection.execute(
                "SELECT type FROM artifacts WHERE run_id = ? AND status = 'committed'", (job["run_id"],)
            ).fetchall()
            return next_phase(artifact["type"] for artifact in artifacts)

    def requeue_recoverable_job(self, job_id: str) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"任务不存在：{job_id}")
            if job["status"] not in {"failed", "interrupted", "force_stopped"}:
                raise ValueError("当前任务不能恢复")
            artifacts = connection.execute(
                "SELECT type FROM artifacts WHERE run_id = ? AND status = 'committed'", (job["run_id"],)
            ).fetchall()
            recommendation = next_phase(artifact["type"] for artifact in artifacts)
            if recommendation.action == "none" or recommendation.phase is None:
                raise ValueError("当前任务不需要恢复")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (job["run_id"],)).fetchone()
            if run is None:
                raise KeyError(f"运行不存在：{job['run_id']}")
            require_job_transition(job["status"], "queued")
            require_run_transition(run["status"], "queued")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', phase = ?, worker_generation = NULL, error_code = NULL, error_message = NULL,
                    cancel_requested_at = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (recommendation.phase, job_id),
            )
            connection.execute(
                "UPDATE runs SET status = 'queued', phase = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (recommendation.phase, run["id"]),
            )
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def transition_run(self, run_id: str, target: str, *, phase: str | None = None) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"运行不存在：{run_id}")
            require_run_transition(row["status"], target)
            connection.execute(
                "UPDATE runs SET status = ?, phase = COALESCE(?, phase), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (target, phase, run_id),
            )
            return connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def start_stage_attempt(self, *, run_id: str, job_id: str | None, stage: str, staging_dir: str, input_sha256: str | None) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            attempt_no = connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM stage_attempts WHERE run_id = ? AND stage = ?", (run_id, stage)
            ).fetchone()[0]
            attempt_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO stage_attempts(id, run_id, job_id, stage, attempt_no, status, staging_dir, input_sha256)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (attempt_id, run_id, job_id, stage, attempt_no, staging_dir, input_sha256),
            )
            return connection.execute("SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)).fetchone()

    def attempt(self, attempt_id: str) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)).fetchone()

    def prepare_attempt(self, attempt_id: str, manifest_path: str, artifacts: Iterable[PreparedArtifact]) -> None:
        artifacts = list(artifacts)
        if not artifacts:
            raise ValueError("阶段至少需要一个 artifact")
        with self.database.transaction(immediate=True) as connection:
            attempt = connection.execute("SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(f"阶段尝试不存在：{attempt_id}")
            if attempt["status"] != "running":
                raise ValueError(f"阶段尝试不能准备提交：{attempt['status']}")
            for artifact in artifacts:
                connection.execute(
                    """
                    INSERT INTO artifacts(id, run_id, attempt_id, type, variant, storage_kind, relative_path, sha256, size_bytes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')
                    """,
                    (str(uuid.uuid4()), attempt["run_id"], attempt_id, artifact.type, artifact.variant, "managed", artifact.relative_path, artifact.sha256, artifact.size_bytes),
                )
            connection.execute(
                """
                UPDATE stage_attempts
                SET status = 'prepared', manifest_path = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (manifest_path, attempt_id),
            )

    def commit_attempt(self, attempt_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            attempt = connection.execute("SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(f"阶段尝试不存在：{attempt_id}")
            if attempt["status"] != "prepared":
                raise ValueError(f"阶段尝试不能提交：{attempt['status']}")
            connection.execute("UPDATE artifacts SET status = 'committed' WHERE attempt_id = ?", (attempt_id,))
            connection.execute(
                "UPDATE stage_attempts SET status = 'committed', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (attempt_id,),
            )

    def abandon_attempt(self, attempt_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE stage_attempts SET status = 'abandoned', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ? AND status IN ('running', 'prepared')",
                (attempt_id,),
            )

    def pending_attempts(self) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT * FROM stage_attempts WHERE status IN ('running', 'prepared') ORDER BY created_at, id"
            ).fetchall()

    def committed_artifacts(self, run_id: str) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND status = 'committed' ORDER BY type, variant", (run_id,)
            ).fetchall()

    def committed_artifact(self, *, run_id: str, artifact_type: str, variant: str = "canonical") -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND type = ? AND variant = ? AND status = 'committed'",
                (run_id, artifact_type, variant),
            ).fetchone()

    def prepared_artifacts(self, attempt_id: str) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute("SELECT * FROM artifacts WHERE attempt_id = ? ORDER BY type, variant", (attempt_id,)).fetchall()

    def save_speaker_mapping(self, run_id: str, entries: dict[str, str]) -> sqlite3.Row:
        if not entries or any(not re.fullmatch(r"SPEAKER_[0-9]+", label) for label in entries):
            raise ValueError("speaker 映射必须使用匿名 SPEAKER_ 标签")
        if any(not isinstance(name, str) or len(name) > 120 or any(ord(character) < 32 for character in name) for name in entries.values()):
            raise ValueError("speaker 显示名非法")
        with self.database.transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"运行不存在：{run_id}")
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM speaker_mapping_versions WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            mapping_id = str(uuid.uuid4())
            connection.execute("INSERT INTO speaker_mapping_versions(id, run_id, version) VALUES (?, ?, ?)", (mapping_id, run_id, version))
            connection.executemany(
                "INSERT INTO speaker_mapping_entries(mapping_version_id, anonymous_label, display_name) VALUES (?, ?, ?)",
                [(mapping_id, label, name) for label, name in sorted(entries.items())],
            )
            connection.execute("UPDATE runs SET speaker_mapping_version = ? WHERE id = ?", (version, run_id))
            return connection.execute("SELECT * FROM speaker_mapping_versions WHERE id = ?", (mapping_id,)).fetchone()

    def latest_speaker_mapping(self, run_id: str) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
        with self.database.connection() as connection:
            mapping = connection.execute(
                "SELECT * FROM speaker_mapping_versions WHERE run_id = ? ORDER BY version DESC LIMIT 1", (run_id,)
            ).fetchone()
            if mapping is None:
                return None, []
            entries = connection.execute(
                "SELECT anonymous_label, display_name FROM speaker_mapping_entries WHERE mapping_version_id = ? ORDER BY anonymous_label",
                (mapping["id"],),
            ).fetchall()
            return mapping, entries

    def append_event(
        self,
        *,
        job_id: str,
        stage: str,
        event: str,
        completed: int | None = None,
        total: int | None = None,
        message: str | None = None,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        if (stage, event) not in ALLOWED_TASK_EVENTS:
            raise ValueError("任务事件不在白名单内")
        if message is not None and ("\n" in message or "\r" in message or len(message) > 240):
            raise ValueError("任务事件消息格式非法")
        details = details or {}
        if set(details) - ALLOWED_EVENT_DETAIL_KEYS:
            raise ValueError("任务事件详情包含非白名单字段")
        if not all(isinstance(value, (int, float, bool, type(None))) for value in details.values()):
            raise ValueError("任务事件详情只支持数值或布尔值")
        if message is not None and any(token in message.lower() for token in SENSITIVE_EVENT_TOKENS):
            raise ValueError("任务事件包含敏感字段")
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO task_events(job_id, stage, event, completed, total, message, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, stage, event, completed, total, message, json.dumps(details, ensure_ascii=False, sort_keys=True)),
            )

    def task_events(self, job_id: str, *, limit: int = 100) -> list[sqlite3.Row]:
        if not 1 <= limit <= 200:
            raise ValueError("事件数量非法")
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT id, stage, event, completed, total, message, details_json, created_at FROM task_events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()

    def create_deletion_operation(self, *, recording_id: str, manifest: dict[str, Any]) -> sqlite3.Row:
        with self.database.transaction(immediate=True) as connection:
            operation_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO deletion_operations(id, recording_id, storage_kind, manifest_json, status) VALUES (?, ?, 'managed', ?, 'pending')",
                (operation_id, recording_id, json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
            )
            return connection.execute("SELECT * FROM deletion_operations WHERE id = ?", (operation_id,)).fetchone()

    def fail_deletion_operation(self, operation_id: str, error_message: str) -> None:
        with self.database.transaction() as connection:
            if connection.execute(
                """
                UPDATE deletion_operations
                SET status = 'failed', error_message = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ? AND status IN ('pending', 'failed')
                """,
                (error_message, operation_id),
            ).rowcount != 1:
                raise KeyError(f"删除操作不存在或已完成：{operation_id}")

    def deletion_operations(self, recording_id: str) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT * FROM deletion_operations WHERE recording_id = ? ORDER BY created_at, id", (recording_id,)
            ).fetchall()

    def delete_recording_row(self, recording_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            if connection.execute("DELETE FROM recordings WHERE id = ?", (recording_id,)).rowcount != 1:
                raise KeyError(f"录音不存在：{recording_id}")
