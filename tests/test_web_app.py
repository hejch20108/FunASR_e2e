from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from funasr_e2e.persistence.files import StagedArtifact
from funasr_e2e.worker.supervisor import WorkerStatus
from funasr_e2e.web.app import create_app


class FakeSupervisor:
    def __init__(self, **_: object) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> WorkerStatus:
        self.started = True
        return WorkerStatus(pid=1234, generation=1, running=True)

    def refresh(self) -> WorkerStatus:
        return WorkerStatus(pid=1234 if self.started and not self.stopped else None, generation=1 if self.started and not self.stopped else None, running=self.started and not self.stopped)

    def force_stop_job(self, job_id: str) -> WorkerStatus:
        self.stopped = True
        return WorkerStatus(pid=4321, generation=2, running=True)

    def stop(self) -> None:
        self.stopped = True


class WebAppTest(unittest.TestCase):
    def create_app(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        supervisor = FakeSupervisor()
        settings_path = Path(directory.name) / "settings.yaml"
        settings_path.write_text("{}\n", encoding="utf-8")
        app = create_app(
            app_data_dir=Path(directory.name) / "app_data",
            project_dir=Path(directory.name),
            settings_path=settings_path,
            supervisor_factory=lambda **_: supervisor,
        )
        return app, supervisor

    def add_committed_final(self, app, final_text: str = "SPEAKER_0：\n最终稿\n"):
        services = app.state.services
        upload = services.store.stream_upload(io.BytesIO(b"audio-content"))
        recording, _ = services.repository.create_or_get_recording(
            original_filename="sample.wav", display_name="sample", extension=".wav", sha256=upload.sha256,
            size_bytes=upload.size_bytes, source_path="runtime/uploads/pending.part",
        )
        services.store.commit_source(recording_id=recording["id"], upload_path=upload.path, extension=".wav")
        run = services.repository.create_run(
            recording_id=recording["id"],
            preset_spk_num=None,
            settings_snapshot={"postprocess": {"speaker_prefix": "说话人"}},
        )
        attempt_id, staging = services.store.create_attempt(recording_id=recording["id"], run_id=run["id"], job_id=None, stage="final", input_sha256=None)
        final_path = staging / "final.txt"
        final_path.write_text(final_text, encoding="utf-8")
        services.store.commit_attempt(attempt_id, [StagedArtifact("final", final_path)])
        return recording["id"]

    def test_health_runs_worker_lifecycle(self) -> None:
        app, supervisor = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"worker_running": True})
            self.assertTrue(supervisor.started)
        self.assertTrue(supervisor.stopped)

    def test_production_frontend_serves_assets_and_spa_routes(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        project_dir = Path(directory.name)
        settings_path = project_dir / "settings.yaml"
        settings_path.write_text("{}\n", encoding="utf-8")
        dist = project_dir / "frontend" / "dist"
        assets = dist / "assets"
        assets.mkdir(parents=True)
        (dist / "index.html").write_text("<main>FunASR_e2e</main>", encoding="utf-8")
        (assets / "app.js").write_text("console.log('ok')", encoding="utf-8")
        app = create_app(
            app_data_dir=project_dir / "app_data",
            project_dir=project_dir,
            settings_path=settings_path,
            supervisor_factory=lambda **_: FakeSupervisor(),
        )

        with TestClient(app, base_url="http://127.0.0.1") as client:
            root = client.get("/")
            spa_route = client.get("/recordings/example")
            asset = client.get("/assets/app.js")
            unknown_api = client.get("/api/not-a-route")

        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.text, "<main>FunASR_e2e</main>")
        self.assertEqual(spa_route.status_code, 200)
        self.assertEqual(spa_route.text, "<main>FunASR_e2e</main>")
        self.assertEqual(asset.status_code, 200)
        self.assertEqual(asset.text, "console.log('ok')")
        self.assertEqual(unknown_api.status_code, 404)
        self.assertEqual(unknown_api.json()["error"]["code"], "API_NOT_FOUND")

    def test_local_middleware_rejects_untrusted_host_and_origin(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            host_response = client.get("/api/health", headers={"Host": "example.invalid"})
            origin_response = client.post("/api/not-a-route", headers={"Origin": "https://example.invalid"})
        self.assertEqual(host_response.status_code, 421)
        self.assertEqual(host_response.json()["error"]["code"], "HOST_REJECTED")
        self.assertEqual(origin_response.status_code, 403)
        self.assertEqual(origin_response.json()["error"]["code"], "ORIGIN_REJECTED")

    def test_artifact_audio_and_zip_downloads_use_committed_files(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            artifact = client.get(f"/api/recordings/{recording_id}/artifacts/final")
            ranged_audio = client.get(f"/api/recordings/{recording_id}/audio", headers={"Range": "bytes=1-4"})
            archive = client.get(f"/api/recordings/{recording_id}/download/all")
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.content.decode("utf-8"), "SPEAKER_0：\r\n最终稿\r\n")
        self.assertEqual(ranged_audio.status_code, 206)
        self.assertEqual(ranged_audio.content, b"udio")
        self.assertEqual(ranged_audio.headers["content-range"], "bytes 1-4/13")
        self.assertEqual(archive.status_code, 200)
        self.assertTrue(archive.content.startswith(b"PK"))

    def test_speaker_mapping_is_versioned_without_artifact_mutation(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            initial = client.get(f"/api/recordings/{recording_id}/speaker-mapping")
            saved = client.post(f"/api/recordings/{recording_id}/speaker-mapping", json={"entries": {"SPEAKER_0": "甲", "SPEAKER_1": ""}})
            current = client.get(f"/api/recordings/{recording_id}/speaker-mapping")
            artifact = client.get(f"/api/recordings/{recording_id}/artifacts/final")
            display_export = client.get(f"/api/recordings/{recording_id}/download/final?display_names=true")
        self.assertEqual(initial.json()["version"], None)
        self.assertEqual(saved.json()["version"], 1)
        self.assertEqual(current.json()["entries"], {"SPEAKER_0": "甲", "SPEAKER_1": ""})
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(current.json()["speaker_prefix"], "说话人")
        self.assertEqual(display_export.text, "甲：\n最终稿\n")

    def test_display_export_replaces_prefixed_labels_only(self) -> None:
        app, _ = self.create_app()
        canonical_final = "[00:00] 说话人0【待回听】：第一句\n正文中的说话人0不应替换。\nSPEAKER_1：第二句\n"
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app, canonical_final)
            saved = client.post(
                f"/api/recordings/{recording_id}/speaker-mapping",
                json={"entries": {"SPEAKER_0": "甲", "SPEAKER_1": "乙"}},
            )
            display_export = client.get(f"/api/recordings/{recording_id}/download/final?display_names=true")
            canonical_export = client.get(f"/api/recordings/{recording_id}/download/final")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(
            display_export.text,
            "[00:00] 甲【待回听】：第一句\n正文中的说话人0不应替换。\n乙：第二句\n",
        )
        self.assertEqual(canonical_export.text, canonical_final.replace("\n", "\r\n"))

    def test_speaker_summary_reads_verified_raw_artifact(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            services = app.state.services
            _, run, _ = services.repository.recording_detail(recording_id)
            assert run is not None
            attempt_id, staging = services.store.create_attempt(
                recording_id=recording_id,
                run_id=run["id"],
                job_id=None,
                stage="funasr",
                input_sha256=None,
            )
            raw_path = staging / "raw.json"
            raw_path.write_text(json.dumps([{"sentence_info": [
                {"start": 0, "end": 1200, "spk": 0, "text": "第一句"},
                {"start": 1400, "end": 2600, "spk": 0, "text": "第二句较长"},
                {"start": 2700, "end": 3200, "spk": 1, "text": "另一位说话人"},
            ]}], ensure_ascii=False), encoding="utf-8")
            services.store.commit_attempt(attempt_id, [StagedArtifact("raw_json", raw_path)])
            response = client.get(f"/api/recordings/{recording_id}/speaker-summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["anonymous_label"], "SPEAKER_0")
        self.assertEqual(response.json()["items"][0]["occurrence_count"], 2)
        self.assertEqual(response.json()["items"][1]["excerpts"][0]["text"], "另一位说话人")

    def test_upload_list_start_and_cancel_job(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            upload = client.post(
                "/api/recordings/uploads",
                data={"auto_start": "false"},
                files={"file": ("访谈.wav", b"audio", "audio/wav")},
            )
            self.assertEqual(upload.status_code, 200)
            payload = upload.json()
            self.assertTrue(payload["created"])
            recording_id = payload["recording"]["id"]
            self.assertIsNone(payload["recording"]["job"])

            listing = client.get("/api/recordings", params={"query": "访谈"})
            self.assertEqual(listing.status_code, 200)
            self.assertEqual(listing.json()["total"], 1)
            self.assertEqual(listing.json()["items"][0]["display_name"], "访谈")

            started = client.post(f"/api/recordings/{recording_id}/funasr-jobs", json={"preset_spk_num": 2})
            self.assertEqual(started.status_code, 200)
            job_id = started.json()["job"]["id"]
            self.assertEqual(started.json()["job"]["status"], "queued")
            self.assertEqual(client.post(f"/api/recordings/{recording_id}/funasr-jobs", json={}).status_code, 409)

            cancelled = client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["job"]["status"], "cancelled")
            self.assertEqual(client.post(f"/api/jobs/{job_id}/cancel").json()["job"]["status"], "cancelled")
            self.assertEqual(client.get(f"/api/recordings/{recording_id}").json()["recording"]["run"]["status"], "cancelled")
            renamed = client.patch(f"/api/recordings/{recording_id}", json={"display_name": "已归档访谈"})
            self.assertEqual(renamed.status_code, 200)
            self.assertEqual(renamed.json()["display_name"], "已归档访谈")

    def test_force_stop_requires_cancel_grace_before_restarting_worker(self) -> None:
        app, supervisor = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            uploaded = client.post("/api/recordings/uploads", data={"auto_start": "true"}, files={"file": ("blocked.wav", b"audio", "audio/wav")})
            job_id = uploaded.json()["recording"]["job"]["id"]
            app.state.services.repository.claim_next_job(1)
            self.assertEqual(client.post(f"/api/jobs/{job_id}/cancel").status_code, 200)
            too_early = client.post(f"/api/jobs/{job_id}/force-stop")
            self.assertEqual(too_early.status_code, 409)
            self.assertEqual(too_early.json()["error"]["code"], "FORCE_STOP_GRACE_REQUIRED")
            with app.state.services.repository.database.transaction() as connection:
                connection.execute("UPDATE jobs SET cancel_requested_at = '2000-01-01T00:00:00.000Z' WHERE id = ?", (job_id,))
            forced = client.post(f"/api/jobs/{job_id}/force-stop")
        self.assertEqual(forced.status_code, 200)
        self.assertEqual(forced.json(), {"worker_running": True})
        self.assertTrue(supervisor.stopped)

    def test_batch_start_and_reorder_preserves_requested_fifo(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            first = client.post("/api/recordings/uploads", data={"auto_start": "false"}, files={"file": ("first.wav", b"first", "audio/wav")})
            second = client.post("/api/recordings/uploads", data={"auto_start": "false"}, files={"file": ("second.wav", b"second", "audio/wav")})
            recording_ids = [first.json()["recording"]["id"], second.json()["recording"]["id"]]
            batch = client.post("/api/recordings/batch/funasr-jobs", json={"recording_ids": recording_ids})
            self.assertEqual(batch.status_code, 200)
            jobs = batch.json()["jobs"]
            job_ids = [item["job"]["id"] for item in jobs]
            queued = client.get("/api/jobs/queue")
            reordered = client.post("/api/jobs/reorder", json={"job_ids": list(reversed(job_ids))})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual([item["id"] for item in queued.json()["jobs"]], job_ids)
        self.assertEqual(reordered.status_code, 200)
        self.assertEqual([item["id"] for item in reordered.json()["jobs"]], list(reversed(job_ids)))

    def test_removed_import_routes_return_api_not_found(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            scan = client.post("/api/imports/scan")
            register = client.post("/api/imports/register", json={"paths": []})
        self.assertEqual(scan.status_code, 404)
        self.assertEqual(scan.json()["error"]["code"], "API_NOT_FOUND")
        self.assertEqual(register.status_code, 404)
        self.assertEqual(register.json()["error"]["code"], "API_NOT_FOUND")

    def test_tampered_managed_artifact_is_not_downloadable(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            _, run, _ = app.state.services.repository.recording_detail(recording_id)
            assert run is not None
            artifact = app.state.services.repository.committed_artifact(run_id=run["id"], artifact_type="final")
            assert artifact is not None
            app.state.services.store.artifact_path(artifact["relative_path"]).write_text("已篡改\n", encoding="utf-8")
            response = client.get(f"/api/recordings/{recording_id}/artifacts/final")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "ARTIFACT_INTEGRITY_FAILED")

    def test_managed_delete_requires_confirmation_and_releases_hash(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            recording_dir = app.state.services.store.recording_dir(recording_id)
            missing_confirmation = client.delete(f"/api/recordings/{recording_id}")
            deleted = client.delete(f"/api/recordings/{recording_id}?confirm=true")
            replacement = client.post(
                "/api/recordings/uploads",
                data={"auto_start": "false"},
                files={"file": ("replacement.wav", b"audio-content", "audio/wav")},
            )
        self.assertEqual(missing_confirmation.status_code, 400)
        self.assertEqual(missing_confirmation.json()["error"]["code"], "DELETE_CONFIRMATION_REQUIRED")
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertFalse(recording_dir.exists())
        self.assertIsNone(app.state.services.repository.recording(recording_id))
        self.assertTrue(replacement.json()["created"])

    def test_recommended_recovery_requeues_failed_job_at_safe_phase(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            uploaded = client.post("/api/recordings/uploads", data={"auto_start": "true"}, files={"file": ("recover.wav", b"audio", "audio/wav")})
            job_id = uploaded.json()["recording"]["job"]["id"]
            services = app.state.services
            services.repository.claim_next_job(1)
            services.repository.transition_job(job_id, "failed")
            services.repository.transition_run(uploaded.json()["recording"]["run"]["id"], "failed")

            recommendation = client.get(f"/api/jobs/{job_id}/recommended-recovery")
            self.assertEqual(recommendation.json(), {"action": "run_funasr", "phase": "funasr"})
            recovered = client.post(f"/api/jobs/{job_id}/recommended-recovery")
            self.assertEqual(recovered.status_code, 200)
            self.assertEqual(recovered.json()["job"]["status"], "queued")
            self.assertEqual(recovered.json()["job"]["phase"], "funasr")

    def test_duplicate_upload_reuses_existing_recording(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            first = client.post("/api/recordings/uploads", data={"auto_start": "false"}, files={"file": ("a.wav", b"same", "audio/wav")})
            second = client.post("/api/recordings/uploads", data={"auto_start": "true"}, files={"file": ("b.wav", b"same", "audio/wav")})
        self.assertTrue(first.json()["created"])
        self.assertFalse(second.json()["created"])
        self.assertEqual(first.json()["recording"]["id"], second.json()["recording"]["id"])

    def test_audit_summary_exposes_only_counts(self) -> None:
        app, _ = self.create_app()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            recording_id = self.add_committed_final(app)
            services = app.state.services
            _, run, _ = services.repository.recording_detail(recording_id)
            assert run is not None
            attempt_id, staging = services.store.create_attempt(
                recording_id=recording_id, run_id=run["id"], job_id=None, stage="final", input_sha256=None
            )
            audit_path = staging / "final_audit.json"
            audit_path.write_text(json.dumps({
                "schema_version": 2,
                "integrity": {"fallback_chunk_count": 1, "warning_chunk_count": 2, "warning_count": 3},
            }), encoding="utf-8")
            services.store.commit_attempt(attempt_id, [StagedArtifact("final_audit", audit_path)])
            response = client.get(f"/api/recordings/{recording_id}/audit-summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["final"], {
            "fallback_chunk_count": 1, "warning_chunk_count": 2, "warning_count": 3,
        })

    def test_unexpected_errors_are_sanitized(self) -> None:
        app, _ = self.create_app()

        @app.get("/api/failure")
        def failure() -> None:
            raise RuntimeError("api_key=canary base_url=https://secret.invalid traceback")

        with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
            response = client.get("/api/failure")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": {"code": "INTERNAL_ERROR", "message": "服务内部错误"}})
        self.assertNotIn("canary", response.text)


if __name__ == "__main__":
    unittest.main()
