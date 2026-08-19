from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from funasr_e2e.pipeline.control import PipelineCancelled, PipelineEvent
from funasr_e2e.pipeline.service import (
    LLMCredentials,
    generate_cleaned_stage,
    generate_evidence_stage,
    generate_final_stage,
    run_funasr_stage,
    run_speaker_review_stage,
)
from funasr_e2e.persistence.db import Database
from funasr_e2e.persistence.files import ManagedFileStore, StagedArtifact
from funasr_e2e.persistence.repository import Repository
from scripts.run_funasr_full_pipeline import build_model, load_api_credentials, load_settings, resolve_project_path


class PipelineWorker:
    def __init__(
        self,
        *,
        app_data_dir: Path,
        project_dir: Path,
        settings_path: Path,
        generation: int,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.app_data_dir = app_data_dir.resolve()
        self.project_dir = project_dir
        self.settings_path = settings_path
        self.repository = Repository(Database(self.app_data_dir / "app.sqlite3"))
        self.store = ManagedFileStore(self.app_data_dir, self.repository)
        self.generation = generation
        self.poll_interval_s = poll_interval_s
        self._model: Any | None = None
        self._model_config: str | None = None

    def run_forever(self) -> None:
        self.repository.initialize()
        self.store.initialize()
        self.store.recover_attempts()
        while True:
            if self._stop_request_path().exists():
                return
            job = self.repository.claim_next_job(self.generation)
            if job is None:
                time.sleep(self.poll_interval_s)
                continue
            self._run_job(job)

    def _stop_request_path(self) -> Path:
        return self.app_data_dir / "runtime" / f"worker-stop-{self.generation}.request"

    def _run_job(self, job: Any) -> None:
        try:
            run = self.repository.run(job["run_id"])
            recording = self.repository.recording(job["recording_id"])
            if run is None or recording is None:
                raise RuntimeError("任务引用的录音或运行不存在")
            settings = self._settings_for_run(run)
            artifacts = self._artifacts(run["id"])
            if job["kind"] == "funasr":
                self._run_initial_stages(job, run, recording, artifacts, settings)
            else:
                self._run_continuation_stages(job, run, recording, artifacts, settings)
        except PipelineCancelled:
            self.repository.transition_job(job["id"], "cancelled")
            run = self.repository.run(job["run_id"])
            if run is not None and run["status"] == "running":
                self.repository.transition_run(run["id"], "cancelled")
        except Exception as error:
            error_code, error_message = self._failure_details(error)
            self.store.recover_attempts()
            self.repository.transition_job(job["id"], "failed", error_code=error_code, error_message=error_message)
            self.repository.append_event(job_id=job["id"], stage=job["phase"], event="failed", message=error_message)
            run = self.repository.run(job["run_id"])
            if run is not None and run["status"] == "running":
                self.repository.transition_run(run["id"], "failed")

    @staticmethod
    def _failure_details(error: Exception) -> tuple[str, str]:
        if isinstance(error, ModuleNotFoundError) and error.name == "funasr":
            return "FUNASR_RUNTIME_UNAVAILABLE", "FunASR 运行环境不可用"
        return "PIPELINE_ERROR", "处理失败"

    def _run_initial_stages(
        self,
        job: Any,
        run: Any,
        recording: Any,
        artifacts: dict[str, Path],
        settings: dict[str, Any],
    ) -> None:
        source_path = self.store.artifact_path(recording["source_path"])
        if "raw_json" not in artifacts:
            self.repository.set_job_phase(job["id"], "funasr")
            attempt_id, staging_dir = self.store.create_attempt(
                recording_id=recording["id"],
                run_id=run["id"],
                job_id=job["id"],
                stage="funasr",
                input_sha256=self.store.sha256_file(source_path),
            )
            raw_path = staging_dir / "raw.json"
            run_funasr_stage(
                audio_path=source_path,
                raw_json_path=raw_path,
                model=self._funasr_model(settings),
                funasr_config=settings["funasr"],
                prompt_dir=self._prompt_dir(settings),
                progress_callback=self._progress_callback(job["id"]),
                cancel_check=self._cancel_check(job["id"]),
            )
            self.store.commit_attempt(attempt_id, [StagedArtifact("raw_json", raw_path)])
            artifacts = self._artifacts(run["id"])
        self._check_cancel(job["id"])
        if "evidence" not in artifacts:
            self.repository.set_job_phase(job["id"], "evidence")
            raw_path = artifacts["raw_json"]
            attempt_id, staging_dir = self.store.create_attempt(
                recording_id=recording["id"],
                run_id=run["id"],
                job_id=job["id"],
                stage="evidence",
                input_sha256=self.store.sha256_file(raw_path),
            )
            evidence_path = staging_dir / "evidence.txt"
            generate_evidence_stage(
                raw_json_path=raw_path,
                evidence_path=evidence_path,
                speaker_prefix=settings["postprocess"]["speaker_prefix"],
                keep_time=settings["postprocess"]["keep_time"],
                progress_callback=self._progress_callback(job["id"]),
                cancel_check=self._cancel_check(job["id"]),
            )
            self.store.commit_attempt(attempt_id, [StagedArtifact("evidence", evidence_path)])
        self._check_cancel(job["id"])
        self.repository.complete_initial_job_and_enqueue_continuation(job["id"])

    def _run_continuation_stages(
        self,
        job: Any,
        run: Any,
        recording: Any,
        artifacts: dict[str, Path],
        settings: dict[str, Any],
    ) -> None:
        raw_path = artifacts.get("raw_json")
        if raw_path is None or "evidence" not in artifacts:
            raise RuntimeError("缺少已提交的 FunASR 或 evidence artifact")
        speaker_credentials = self._credentials(settings["speaker_review"], settings)
        final_credentials = self._credentials(settings["llm"], settings)
        if "speaker_review" not in artifacts or "reviewed" not in artifacts:
            self.repository.set_job_phase(job["id"], "speaker_review")
            attempt_id, staging_dir = self.store.create_attempt(
                recording_id=recording["id"],
                run_id=run["id"],
                job_id=job["id"],
                stage="speaker_review",
                input_sha256=self.store.sha256_file(raw_path),
            )
            review_path = staging_dir / "speaker_review.json"
            reviewed_path = staging_dir / "reviewed.txt"
            run_speaker_review_stage(
                raw_json_path=raw_path,
                speaker_review_path=review_path,
                reviewed_path=reviewed_path,
                prompt_dir=self._prompt_dir(settings),
                speaker_review_config=settings["speaker_review"],
                postprocess_config=settings["postprocess"],
                credentials=speaker_credentials,
                progress_callback=self._progress_callback(job["id"]),
                cancel_check=self._cancel_check(job["id"]),
            )
            self.store.commit_attempt(
                attempt_id,
                [StagedArtifact("speaker_review", review_path), StagedArtifact("reviewed", reviewed_path)],
            )
            artifacts = self._artifacts(run["id"])
        self._check_cancel(job["id"])
        if "cleaned" not in artifacts:
            self.repository.set_job_phase(job["id"], "cleaned")
            attempt_id, staging_dir = self.store.create_attempt(
                recording_id=recording["id"],
                run_id=run["id"],
                job_id=job["id"],
                stage="cleaned",
                input_sha256=self.store.sha256_file(artifacts["speaker_review"]),
            )
            cleaned_path = staging_dir / "cleaned.txt"
            generate_cleaned_stage(
                speaker_review_path=artifacts["speaker_review"],
                cleaned_path=cleaned_path,
                prompt_dir=self._prompt_dir(settings),
                postprocess_config=settings["postprocess"],
                progress_callback=self._progress_callback(job["id"]),
                cancel_check=self._cancel_check(job["id"]),
            )
            self.store.commit_attempt(attempt_id, [StagedArtifact("cleaned", cleaned_path)])
            artifacts = self._artifacts(run["id"])
        self._check_cancel(job["id"])
        if "final" not in artifacts or "final_audit" not in artifacts:
            self.repository.set_job_phase(job["id"], "final")
            attempt_id, staging_dir = self.store.create_attempt(
                recording_id=recording["id"],
                run_id=run["id"],
                job_id=job["id"],
                stage="final",
                input_sha256=self.store.sha256_file(artifacts["speaker_review"]),
            )
            final_path = staging_dir / "final.txt"
            final_audit_path = staging_dir / "final_audit.json"
            generate_final_stage(
                raw_json_path=raw_path,
                speaker_review_path=artifacts["speaker_review"],
                final_path=final_path,
                final_audit_path=final_audit_path,
                prompt_dir=self._prompt_dir(settings),
                postprocess_config=settings["postprocess"],
                llm_config=settings["llm"],
                credentials=final_credentials,
                progress_callback=self._progress_callback(job["id"]),
                cancel_check=self._cancel_check(job["id"]),
            )
            self.store.commit_attempt(
                attempt_id,
                [StagedArtifact("final", final_path), StagedArtifact("final_audit", final_audit_path)],
            )
        self._check_cancel(job["id"])
        self.repository.transition_run(run["id"], "completed", phase="complete")
        self.repository.transition_job(job["id"], "succeeded")

    def _artifacts(self, run_id: str) -> dict[str, Path]:
        return {artifact["type"]: self.store.artifact_path(artifact["relative_path"]) for artifact in self.repository.committed_artifacts(run_id)}

    def _settings_for_run(self, run: Any) -> dict[str, Any]:
        try:
            settings = json.loads(run["settings_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("运行配置快照无效") from error
        if not isinstance(settings, dict):
            raise RuntimeError("运行配置快照无效")
        try:
            funasr_config = settings["funasr"]
            settings["paths"]
            settings["postprocess"]
            settings["llm"]
            settings["speaker_review"]
        except KeyError as error:
            raise RuntimeError("运行配置快照不完整") from error
        if not isinstance(funasr_config, dict):
            raise RuntimeError("运行配置快照不完整")
        funasr_config["preset_spk_num"] = run["preset_spk_num"]
        return settings

    def _funasr_model(self, settings: dict[str, Any]) -> Any:
        model_config = json.dumps(settings["funasr"], ensure_ascii=False, sort_keys=True)
        if self._model is None or self._model_config != model_config:
            self._model = None
            gc.collect()
            self._model = build_model(settings)
            self._model_config = model_config
        return self._model

    def _prompt_dir(self, settings: dict[str, Any]) -> Path:
        path = resolve_project_path(self.project_dir, settings["paths"]["prompt_dir"])
        if path is None:
            raise RuntimeError("prompt_dir 不能为空")
        return path

    def _credentials(self, config: dict[str, Any], settings: dict[str, Any]) -> LLMCredentials:
        api_key, base_url, model = load_api_credentials(self.project_dir, settings["paths"], config)
        return LLMCredentials(api_key, base_url, model)

    def _cancel_check(self, job_id: str):
        return lambda: self._check_cancel(job_id)

    def _check_cancel(self, job_id: str) -> None:
        job = self.repository.job(job_id)
        if job is None or job["status"] == "cancel_requested":
            raise PipelineCancelled()

    def _progress_callback(self, job_id: str):
        def callback(event: PipelineEvent) -> None:
            self.repository.append_event(
                job_id=job_id,
                stage=event.stage,
                event=event.event,
                completed=event.completed,
                total=event.total,
                message=event.message,
                details=event.details,
            )

        return callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FunASR_e2e 本机流水线 worker")
    parser.add_argument("--app-data", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    PipelineWorker(
        app_data_dir=Path(args.app_data).resolve(),
        project_dir=Path(args.project_dir).resolve(),
        settings_path=Path(args.settings).resolve(),
        generation=args.generation,
        poll_interval_s=args.poll_interval,
    ).run_forever()


if __name__ == "__main__":
    main()
