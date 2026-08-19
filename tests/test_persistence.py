import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funasr_e2e.persistence.db import Database
from funasr_e2e.persistence.files import ManagedFileStore, StagedArtifact
from funasr_e2e.persistence.repository import PreparedArtifact, Repository
from funasr_e2e.persistence.state_machine import next_phase, require_job_transition
from funasr_e2e.worker.supervisor import WorkerSupervisor


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.app_data_dir = root / "app_data"
        self.repository = Repository(Database(self.app_data_dir / "app.sqlite3"))
        self.repository.initialize()
        self.store = ManagedFileStore(self.app_data_dir, self.repository)
        self.store.initialize()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def create_recording(self, contents: bytes = b"audio"):
        digest = hashlib.sha256(contents).hexdigest()
        return self.repository.create_or_get_recording(
            original_filename="sample.wav",
            display_name="sample",
            extension=".wav",
            sha256=digest,
            size_bytes=len(contents),
            source_path="runtime/uploads/pending.part",
        )

    def test_v1_migration_removes_legacy_records_without_touching_external_files(self):
        database = Database(Path(self.directory.name) / "legacy.sqlite3")
        external_file = Path(self.directory.name) / "external.wav"
        external_file.write_bytes(b"legacy-audio")
        expected_mtime = external_file.stat().st_mtime_ns
        legacy_sha256 = hashlib.sha256(b"legacy-audio").hexdigest()
        initial_schema = PROJECT_DIR / "funasr_e2e" / "persistence" / "migrations" / "001_initial.sql"
        with database.connection() as connection:
            connection.executescript(initial_schema.read_text(encoding="utf-8"))
            connection.execute("PRAGMA user_version = 1")
            connection.execute(
                """
                INSERT INTO recordings(id, original_filename, display_name, extension, sha256, size_bytes, storage_kind, source_path)
                VALUES ('legacy-recording', 'legacy.wav', 'legacy', '.wav', ?, 12, 'legacy_external', ?)
                """,
                (legacy_sha256, str(external_file)),
            )
            connection.execute(
                """
                INSERT INTO runs(id, recording_id, version, preset_spk_num, settings_json, status, phase)
                VALUES ('legacy-run', 'legacy-recording', 1, NULL, '{}', 'completed', 'complete')
                """,
            )
            connection.execute("UPDATE recordings SET current_run_id = 'legacy-run' WHERE id = 'legacy-recording'")
            connection.execute(
                """
                INSERT INTO stage_attempts(id, run_id, job_id, stage, attempt_no, status, staging_dir)
                VALUES ('legacy-attempt', 'legacy-run', NULL, 'funasr', 1, 'committed', 'legacy/staging')
                """,
            )
            connection.execute(
                """
                INSERT INTO artifacts(id, run_id, attempt_id, type, variant, storage_kind, relative_path, sha256, size_bytes, status)
                VALUES ('legacy-artifact', 'legacy-run', 'legacy-attempt', 'raw_json', 'canonical', 'legacy_external', ?, 'artifact-hash', 1, 'committed')
                """,
                (str(external_file),),
            )
            connection.execute(
                """
                INSERT INTO recordings(id, original_filename, display_name, extension, sha256, size_bytes, storage_kind, source_path)
                VALUES ('managed-recording', 'managed.wav', 'managed', '.wav', 'managed-hash', 1, 'managed', 'recordings/managed/source/audio.wav')
                """,
            )

        database.initialize()

        with database.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM recordings WHERE id = 'legacy-recording'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs WHERE id = 'legacy-run'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM artifacts WHERE id = 'legacy-artifact'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM recordings WHERE id = 'managed-recording'").fetchone()[0], 1)
        self.assertEqual(external_file.read_bytes(), b"legacy-audio")
        self.assertEqual(external_file.stat().st_mtime_ns, expected_mtime)

        repository = Repository(database)
        replacement, created = repository.create_or_get_recording(
            original_filename="replacement.wav",
            display_name="replacement",
            extension=".wav",
            sha256=legacy_sha256,
            size_bytes=12,
            source_path="runtime/uploads/pending.part",
        )
        self.assertTrue(created)
        self.assertEqual(replacement["storage_kind"], "managed")

    def test_duplicate_hash_is_released_only_after_managed_delete(self):
        recording, created = self.create_recording()
        self.assertTrue(created)
        upload = self.store.stream_upload(io.BytesIO(b"audio"))
        self.store.commit_source(recording_id=recording["id"], upload_path=upload.path, extension=".wav")

        duplicate, created = self.create_recording()
        self.assertFalse(created)
        self.assertEqual(duplicate["id"], recording["id"])

        self.store.delete_managed_recording(recording["id"])
        replacement, created = self.create_recording()
        self.assertTrue(created)
        self.assertNotEqual(replacement["id"], recording["id"])

    def test_failed_managed_delete_keeps_recording_and_hash_reservation(self):
        recording, _ = self.create_recording()
        upload = self.store.stream_upload(io.BytesIO(b"audio"))
        self.store.commit_source(recording_id=recording["id"], upload_path=upload.path, extension=".wav")

        with patch("funasr_e2e.persistence.files.shutil.rmtree", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                self.store.delete_managed_recording(recording["id"])
        self.assertIsNotNone(self.repository.recording(recording["id"]))
        self.assertEqual(self.repository.deletion_operations(recording["id"])[0]["status"], "failed")
        _, created = self.create_recording()
        self.assertFalse(created)

    def test_claim_is_fifo_and_one_recording_cannot_have_two_active_jobs(self):
        first, _ = self.create_recording(b"first")
        second, _ = self.create_recording(b"second")
        first_run = self.repository.create_run(recording_id=first["id"], preset_spk_num=None, settings_snapshot={})
        second_run = self.repository.create_run(recording_id=second["id"], preset_spk_num=2, settings_snapshot={})
        first_job = self.repository.enqueue_job(recording_id=first["id"], run_id=first_run["id"], kind="funasr", phase="funasr")
        self.repository.enqueue_job(recording_id=second["id"], run_id=second_run["id"], kind="funasr", phase="funasr")

        with self.assertRaisesRegex(ValueError, "活动任务"):
            self.repository.enqueue_job(recording_id=first["id"], run_id=first_run["id"], kind="continuation", phase="evidence")
        self.assertEqual(self.repository.claim_next_job(1)["id"], first_job["id"])
        self.assertEqual(self.repository.claim_next_job(1)["recording_id"], second["id"])

    def test_unprepared_staging_is_abandoned_and_never_becomes_visible(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        attempt_id, staging_dir = self.store.create_attempt(
            recording_id=recording["id"],
            run_id=run["id"],
            job_id=None,
            stage="funasr",
            input_sha256=None,
        )
        (staging_dir / "raw.json").write_text('{"partial": true}', encoding="utf-8")

        self.assertEqual(self.store.recover_attempts(), [])
        self.assertEqual(self.repository.attempt(attempt_id)["status"], "abandoned")
        self.assertEqual(self.repository.committed_artifacts(run["id"]), [])

    def test_manifest_recovery_commits_only_hash_verified_artifacts(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        attempt_id, staging_dir = self.store.create_attempt(
            recording_id=recording["id"],
            run_id=run["id"],
            job_id=None,
            stage="funasr",
            input_sha256=None,
        )
        source = staging_dir / "raw.json"
        source.write_text('{"sentence_info": []}', encoding="utf-8")
        digest = self.store.sha256_file(source)
        target = self.store.run_dir(recording["id"], run["id"]) / "artifacts" / "raw_json" / "raw.json"
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "attempt_id": attempt_id,
            "run_id": run["id"],
            "stage": "funasr",
            "artifacts": [{
                "type": "raw_json",
                "variant": "canonical",
                "filename": "raw.json",
                "sha256": digest,
                "size_bytes": source.stat().st_size,
                "relative_path": self.store._relative(target),
            }],
        }), encoding="utf-8")
        self.repository.prepare_attempt(
            attempt_id,
            self.store._relative(manifest_path),
            [PreparedArtifact("raw_json", self.store._relative(target), digest, source.stat().st_size)],
        )

        self.assertEqual(self.store.recover_attempts(), [attempt_id])
        self.assertTrue(target.exists())
        self.assertEqual(self.store.sha256_file(target), digest)
        self.assertEqual(self.repository.attempt(attempt_id)["status"], "committed")
        self.assertEqual(self.repository.committed_artifacts(run["id"])[0]["type"], "raw_json")

    def test_commit_attempt_publishes_artifact_after_prepared_state(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        attempt_id, staging_dir = self.store.create_attempt(
            recording_id=recording["id"],
            run_id=run["id"],
            job_id=None,
            stage="evidence",
            input_sha256="input-hash",
        )
        evidence = staging_dir / "evidence.txt"
        evidence.write_text("证据内容\n", encoding="utf-8")

        targets = self.store.commit_attempt(attempt_id, [StagedArtifact("evidence", evidence)])
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].is_file())
        self.assertEqual(self.repository.attempt(attempt_id)["status"], "committed")
        artifact = self.repository.committed_artifacts(run["id"])[0]
        self.assertEqual(artifact["sha256"], self.store.sha256_file(targets[0]))

    def test_initial_job_atomically_enqueues_continuation(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        initial = self.repository.enqueue_job(recording_id=recording["id"], run_id=run["id"], kind="funasr", phase="funasr")
        self.assertEqual(self.repository.claim_next_job(1)["id"], initial["id"])

        continuation = self.repository.complete_initial_job_and_enqueue_continuation(initial["id"])
        self.assertEqual(continuation["kind"], "continuation")
        self.assertEqual(continuation["status"], "queued")
        self.assertEqual(self.repository.job(initial["id"])["status"], "succeeded")
        self.assertEqual(self.repository.run(run["id"])["status"], "queued")
        self.assertEqual(self.repository.claim_next_job(1)["id"], continuation["id"])

    def test_supervisor_marks_only_its_interrupted_worker_job(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        job = self.repository.enqueue_job(recording_id=recording["id"], run_id=run["id"], kind="funasr", phase="funasr")
        self.repository.claim_next_job(42)
        supervisor = WorkerSupervisor(
            app_data_dir=self.store.root,
            project_dir=Path(self.directory.name),
            settings_path=Path(self.directory.name) / "settings.yaml",
        )

        supervisor._mark_worker_interrupted(42, force_stopped=True)
        self.assertEqual(self.repository.job(job["id"])["status"], "force_stopped")
        self.assertEqual(self.repository.run(run["id"])["status"], "interrupted")

    def test_task_events_only_accept_whitelisted_non_sensitive_fields(self):
        recording, _ = self.create_recording()
        run = self.repository.create_run(recording_id=recording["id"], preset_spk_num=None, settings_snapshot={})
        job = self.repository.enqueue_job(recording_id=recording["id"], run_id=run["id"], kind="funasr", phase="funasr")
        self.repository.append_event(
            job_id=job["id"],
            stage="funasr",
            event="completed",
            details={"sentence_count": 3},
        )
        with self.assertRaisesRegex(ValueError, "非白名单"):
            self.repository.append_event(
                job_id=job["id"],
                stage="funasr",
                event="completed",
                details={"prompt": 3},
            )

    def test_state_machine_provides_one_recommended_next_action(self):
        self.assertEqual(next_phase([]).action, "run_funasr")
        self.assertEqual(next_phase(["raw_json", "evidence"]).phase, "speaker_review")
        self.assertEqual(next_phase(["raw_json", "evidence", "speaker_review", "reviewed", "cleaned", "final", "final_audit"]).action, "none")
        with self.assertRaisesRegex(ValueError, "不允许"):
            require_job_transition("queued", "succeeded")


if __name__ == "__main__":
    unittest.main()
