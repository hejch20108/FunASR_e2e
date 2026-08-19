from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from funasr_e2e.persistence.db import Database
from funasr_e2e.persistence.files import ManagedFileStore
from funasr_e2e.persistence.repository import Repository

from .windows_job import WindowsJob


@dataclass(frozen=True)
class WorkerStatus:
    pid: int | None
    generation: int | None
    running: bool


class WorkerSupervisor:
    def __init__(self, *, app_data_dir: Path, project_dir: Path, settings_path: Path) -> None:
        self.app_data_dir = app_data_dir.resolve()
        self.project_dir = project_dir.resolve()
        self.settings_path = settings_path.resolve()
        self.repository = Repository(Database(self.app_data_dir / "app.sqlite3"))
        self.store = ManagedFileStore(self.app_data_dir, self.repository)
        self._process: subprocess.Popen[bytes] | None = None
        self._job: WindowsJob | None = None
        self._generation: int | None = None

    def start(self) -> WorkerStatus:
        self.repository.initialize()
        self.store.initialize()
        self.refresh()
        if self._process is not None:
            return self.status()
        generation = time.time_ns()
        arguments = [
            sys.executable,
            "-m",
            "funasr_e2e.worker.main",
            "--app-data",
            str(self.app_data_dir),
            "--project-dir",
            str(self.project_dir),
            "--settings",
            str(self.settings_path),
            "--generation",
            str(generation),
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            arguments,
            cwd=self.project_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        job: WindowsJob | None = None
        if os.name == "nt":
            try:
                job = WindowsJob()
                job.assign(process._handle)
            except OSError:
                if job is not None:
                    job.close()
                job = None
        self._process = process
        self._job = job
        self._generation = generation
        return self.status()

    def refresh(self) -> WorkerStatus:
        if self._process is not None and self._process.poll() is not None:
            generation = self._generation
            self._close_job()
            self._process = None
            self._generation = None
            if generation is not None:
                self._mark_worker_interrupted(generation, force_stopped=False)
        return self.status()

    def force_stop_job(self, job_id: str) -> WorkerStatus:
        status = self.refresh()
        if not status.running or status.generation is None:
            raise ValueError("worker 未运行")
        active = self.repository.running_jobs(status.generation)
        if len(active) != 1 or active[0]["id"] != job_id or active[0]["status"] != "cancel_requested":
            raise ValueError("指定任务不是当前已请求取消的任务")
        request_path = self.app_data_dir / "runtime" / f"worker-stop-{status.generation}.request"
        with request_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(job_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        time.sleep(0.2)
        active = self.repository.running_jobs(status.generation)
        if len(active) != 1 or active[0]["id"] != job_id or active[0]["status"] != "cancel_requested":
            request_path.unlink(missing_ok=True)
            raise ValueError("任务已停止或 worker 已切换任务")
        try:
            return self.force_stop()
        finally:
            request_path.unlink(missing_ok=True)

    def force_stop(self, *, restart: bool = True) -> WorkerStatus:
        self.refresh()
        if self._process is None or self._generation is None:
            return self.status()
        process = self._process
        generation = self._generation
        self._close_job()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True, timeout=20)
                process.wait(timeout=10)
            else:
                process.kill()
                process.wait(timeout=10)
        self._process = None
        self._generation = None
        self._mark_worker_interrupted(generation, force_stopped=True)
        self.store.recover_attempts()
        return self.start() if restart else self.status()

    def stop(self) -> None:
        self.force_stop(restart=False)

    def status(self) -> WorkerStatus:
        if self._process is None or self._process.poll() is not None:
            return WorkerStatus(pid=None, generation=None, running=False)
        return WorkerStatus(pid=self._process.pid, generation=self._generation, running=True)

    def _mark_worker_interrupted(self, generation: int, *, force_stopped: bool) -> None:
        target = "force_stopped" if force_stopped else "interrupted"
        for job in self.repository.running_jobs(generation):
            self.repository.transition_job(job["id"], target)
            run = self.repository.run(job["run_id"])
            if run is not None and run["status"] == "running":
                self.repository.transition_run(run["id"], "interrupted")

    def _close_job(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None
