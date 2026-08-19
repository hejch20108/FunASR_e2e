from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funasr_e2e.persistence.db import Database
from funasr_e2e.persistence.files import ManagedFileStore
from funasr_e2e.persistence.repository import Repository
from funasr_e2e.worker.main import PipelineWorker
from funasr_e2e.worker.supervisor import WorkerSupervisor


class FakeFunASRModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [{
            "sentence_info": [
                {"text": "第一句", "start": 0, "end": 800, "spk": 0},
                {"text": "第二句", "start": 900, "end": 1500, "spk": 1},
            ]
        }]


def settings_snapshot(prompt_dir: Path, *, preset_spk_num: int | None = None) -> dict[str, object]:
    return {
        "paths": {
            "env_file": str(prompt_dir.parent / ".env"),
            "prompt_dir": str(prompt_dir),
        },
        "funasr": {
            "model": "fake-model",
            "vad_model": "fake-vad",
            "punc_model": "fake-punc",
            "spk_model": "fake-speaker",
            "device": "cpu",
            "batch_size_s": 300,
            "batch_size_threshold_s": 60,
            "max_single_segment_time": 60000,
            "preset_spk_num": preset_spk_num,
        },
        "postprocess": {
            "max_gap_ms": 2000,
            "max_chars": 400,
            "speaker_prefix": "快照说话人",
            "keep_time": True,
        },
        "llm": {
            "model": None,
            "chunk_size": 8,
            "max_workers": 1,
            "max_retries": 1,
            "enable_thinking": False,
            "api_key_env": "API_KEY",
            "base_url_env": "BASE_URL",
            "model_name_env": "MODEL_NAME",
        },
        "speaker_review": {
            "enabled": True,
            "model": None,
            "context_size": 1,
            "max_risk_core_sentences": 2,
            "max_boundary_candidates": 2,
            "max_workers": 1,
            "max_retries": 1,
            "request_timeout_s": 1,
            "auto_apply_confidence": 0.9,
            "allowed_speakers": None,
            "unknown_label": "unknown",
            "overlap_label": "overlap",
            "failure_policy": "keep_original",
            "api_key_env": "API_KEY",
            "base_url_env": "BASE_URL",
            "model_name_env": "MODEL_NAME",
        },
    }


class PipelineWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.app_data_dir = self.root / "app_data"
        self.project_dir = self.root / "project"
        self.prompt_dir = self.project_dir / "prompt"
        self.prompt_dir.mkdir(parents=True)
        (self.prompt_dir / "hotwords.txt").write_text("术语\n", encoding="utf-8")
        self.repository = Repository(Database(self.app_data_dir / "app.sqlite3"))
        self.repository.initialize()
        self.store = ManagedFileStore(self.app_data_dir, self.repository)
        self.store.initialize()
        self.worker = PipelineWorker(
            app_data_dir=self.app_data_dir,
            project_dir=self.project_dir,
            settings_path=self.project_dir / "settings.yaml",
            generation=17,
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def create_claimed_job(self, *, preset_spk_num: int | None = 3):
        upload = self.store.stream_upload(io.BytesIO(b"audio"))
        recording, created = self.repository.create_or_get_recording(
            original_filename="sample.wav",
            display_name="sample",
            extension=".wav",
            sha256=upload.sha256,
            size_bytes=upload.size_bytes,
            source_path="runtime/uploads/pending.part",
        )
        self.assertTrue(created)
        self.store.commit_source(recording_id=recording["id"], upload_path=upload.path, extension=".wav")
        run = self.repository.create_run(
            recording_id=recording["id"],
            preset_spk_num=preset_spk_num,
            settings_snapshot=settings_snapshot(self.prompt_dir, preset_spk_num=99),
        )
        self.repository.enqueue_job(recording_id=recording["id"], run_id=run["id"], kind="funasr", phase="funasr")
        job = self.repository.claim_next_job(self.worker.generation)
        self.assertIsNotNone(job)
        return recording, run, job

    def test_initial_job_uses_run_snapshot_and_enqueues_continuation(self) -> None:
        _, run, initial_job = self.create_claimed_job()
        model = FakeFunASRModel()

        with patch("funasr_e2e.worker.main.build_model", return_value=model):
            self.worker._run_job(initial_job)

        artifacts = {artifact["type"]: artifact for artifact in self.repository.committed_artifacts(run["id"])}
        self.assertEqual(set(artifacts), {"raw_json", "evidence"})
        evidence = self.store.artifact_path(artifacts["evidence"]["relative_path"])
        self.assertIn("快照说话人0", evidence.read_text(encoding="utf-8"))
        self.assertEqual(model.calls[0]["preset_spk_num"], 3)
        self.assertEqual(self.repository.job(initial_job["id"])["status"], "succeeded")
        continuation = self.repository.claim_next_job(self.worker.generation)
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation["kind"], "continuation")
        self.assertEqual(continuation["phase"], "speaker_review")

    def test_continuation_completes_without_speaker_mapping(self) -> None:
        _, run, initial_job = self.create_claimed_job()
        with patch("funasr_e2e.worker.main.build_model", return_value=FakeFunASRModel()):
            self.worker._run_job(initial_job)
        continuation = self.repository.claim_next_job(self.worker.generation)
        self.assertIsNotNone(continuation)

        def write_review(**kwargs: object) -> None:
            Path(str(kwargs["speaker_review_path"])).write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
            Path(str(kwargs["reviewed_path"])).write_text("已复核\n", encoding="utf-8")

        def write_cleaned(**kwargs: object) -> None:
            Path(str(kwargs["cleaned_path"])).write_text("已清洗\n", encoding="utf-8")

        def write_final(**kwargs: object) -> None:
            Path(str(kwargs["final_path"])).write_text("最终稿\n", encoding="utf-8")
            Path(str(kwargs["final_audit_path"])).write_text("{}\n", encoding="utf-8")

        with (
            patch("funasr_e2e.worker.main.load_api_credentials", return_value=("key", "base", "model")),
            patch("funasr_e2e.worker.main.run_speaker_review_stage", side_effect=write_review),
            patch("funasr_e2e.worker.main.generate_cleaned_stage", side_effect=write_cleaned),
            patch("funasr_e2e.worker.main.generate_final_stage", side_effect=write_final),
        ):
            self.worker._run_job(continuation)

        self.assertEqual(self.repository.run(run["id"])["status"], "completed")
        self.assertEqual(self.repository.job(continuation["id"])["status"], "succeeded")
        artifacts = self.repository.committed_artifacts(run["id"])
        self.assertEqual(
            {artifact["type"] for artifact in artifacts},
            {"raw_json", "evidence", "speaker_review", "reviewed", "cleaned", "final", "final_audit"},
        )
        attempts = {artifact["type"]: artifact["attempt_id"] for artifact in artifacts}
        self.assertEqual(attempts["final"], attempts["final_audit"])

    def test_missing_funasr_runtime_is_visible_and_abandons_staging(self) -> None:
        _, run, job = self.create_claimed_job()
        with patch(
            "funasr_e2e.worker.main.build_model",
            side_effect=ModuleNotFoundError("No module named 'funasr'", name="funasr"),
        ):
            self.worker._run_job(job)

        failed_job = self.repository.job(job["id"])
        self.assertEqual(failed_job["status"], "failed")
        self.assertEqual(failed_job["error_code"], "FUNASR_RUNTIME_UNAVAILABLE")
        self.assertEqual(failed_job["error_message"], "FunASR 运行环境不可用")
        self.assertEqual(self.repository.run(run["id"])["status"], "failed")
        self.assertEqual(self.repository.pending_attempts(), [])
        self.assertEqual(self.repository.task_events(job["id"])[0]["event"], "failed")

    def test_cancel_after_funasr_only_keeps_committed_raw_artifact(self) -> None:
        _, run, job = self.create_claimed_job()

        def cancel_after_writing_raw(**kwargs: object) -> None:
            Path(str(kwargs["raw_json_path"])).write_text('{"sentence_info": []}', encoding="utf-8")
            self.repository.request_cancel(job["id"])

        with (
            patch("funasr_e2e.worker.main.build_model", return_value=FakeFunASRModel()),
            patch("funasr_e2e.worker.main.run_funasr_stage", side_effect=cancel_after_writing_raw),
        ):
            self.worker._run_job(job)

        self.assertEqual(self.repository.job(job["id"])["status"], "cancelled")
        self.assertEqual(self.repository.run(run["id"])["status"], "cancelled")
        self.assertEqual({artifact["type"] for artifact in self.repository.committed_artifacts(run["id"])}, {"raw_json"})
        self.assertEqual(len(self.repository.pending_attempts()), 0)

    def test_stop_request_prevents_worker_from_claiming_next_job(self) -> None:
        stop_request = self.worker._stop_request_path()
        stop_request.parent.mkdir(parents=True, exist_ok=True)
        stop_request.write_text("job-id\n", encoding="utf-8")
        with patch.object(self.worker.repository, "claim_next_job") as claim:
            self.worker.run_forever()
        claim.assert_not_called()


@unittest.skipUnless(os.name == "nt", "仅在 Windows 验证 Job Object 监督")
class WorkerSupervisorIntegrationTest(unittest.TestCase):
    def test_supervisor_starts_and_stops_idle_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_data_dir = Path(directory) / "app_data"
            project_dir = Path(__file__).resolve().parents[1]
            supervisor = WorkerSupervisor(
                app_data_dir=app_data_dir,
                project_dir=project_dir,
                settings_path=project_dir / "settings.yaml",
            )
            try:
                status = supervisor.start()
                self.assertTrue(status.running)
                self.assertIsNotNone(status.pid)
                self.assertIsNotNone(status.generation)
            finally:
                supervisor.stop()
            self.assertFalse(supervisor.status().running)


if __name__ == "__main__":
    unittest.main()
