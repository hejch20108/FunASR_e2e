from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from .repository import PreparedArtifact, Repository


_ARTIFACT_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class FileDigest:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class StagedArtifact:
    type: str
    path: Path
    variant: str = "canonical"


class ManagedFileStore:
    def __init__(self, root: Path, repository: Repository) -> None:
        self.root = root.resolve()
        self.repository = repository
        self.runtime_dir = self.root / "runtime"
        self.recordings_dir = self.root / "recordings"

    def initialize(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_artifact_type(value: str) -> str:
        if not _ARTIFACT_TYPE.fullmatch(value):
            raise ValueError(f"artifact 类型非法：{value}")
        return value

    @staticmethod
    def _safe_filename(value: str) -> str:
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("artifact 文件名非法")
        return value

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _path_from_relative(self, value: str) -> Path:
        path = (self.root / value).resolve()
        path.relative_to(self.root)
        return path

    def recording_dir(self, recording_id: str) -> Path:
        return self.recordings_dir / recording_id

    def run_dir(self, recording_id: str, run_id: str) -> Path:
        return self.recording_dir(recording_id) / "runs" / run_id

    def artifact_path(self, relative_path: str) -> Path:
        return self._path_from_relative(relative_path)

    def stream_upload(self, source: BinaryIO, *, suffix: str = ".part", minimum_free_bytes: int = 0) -> FileDigest:
        self.initialize()
        uploads_dir = self.runtime_dir / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=uploads_dir)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(fd, "wb") as target:
                while block := source.read(1024 * 1024):
                    if shutil.disk_usage(uploads_dir).free < minimum_free_bytes + len(block):
                        raise OSError("可用磁盘空间不足")
                    target.write(block)
                    digest.update(block)
                    size_bytes += len(block)
                target.flush()
                os.fsync(target.fileno())
            return FileDigest(Path(temp_name), digest.hexdigest(), size_bytes)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def commit_source(self, *, recording_id: str, upload_path: Path, extension: str) -> Path:
        if not extension.startswith(".") or len(extension) > 16 or any(char in extension for char in "/\\"):
            raise ValueError("音频扩展名非法")
        target = self.recording_dir(recording_id) / "source" / f"audio{extension.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if self.sha256_file(target) != self.sha256_file(upload_path):
                raise FileExistsError(f"录音源文件已存在：{target}")
            upload_path.unlink(missing_ok=True)
        else:
            os.replace(upload_path, target)
        self.repository.update_recording_source_path(recording_id, self._relative(target))
        return target

    def create_attempt(self, *, recording_id: str, run_id: str, job_id: str | None, stage: str, input_sha256: str | None) -> tuple[str, Path]:
        staging_base = self.run_dir(recording_id, run_id) / "staging"
        staging_base.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=staging_base))
        attempt = self.repository.start_stage_attempt(
            run_id=run_id,
            job_id=job_id,
            stage=stage,
            staging_dir=self._relative(temporary_dir),
            input_sha256=input_sha256,
        )
        return attempt["id"], temporary_dir

    def commit_attempt(self, attempt_id: str, artifacts: Iterable[StagedArtifact]) -> list[Path]:
        attempt = self.repository.attempt(attempt_id)
        if attempt is None:
            raise KeyError(f"阶段尝试不存在：{attempt_id}")
        if attempt["status"] != "running":
            raise ValueError(f"阶段尝试不能提交：{attempt['status']}")
        run = self.repository.run(attempt["run_id"])
        if run is None:
            raise KeyError(f"运行不存在：{attempt['run_id']}")
        staging_dir = self._path_from_relative(attempt["staging_dir"])
        staged = list(artifacts)
        if not staged:
            raise ValueError("阶段至少需要一个 artifact")
        manifest_artifacts = []
        prepared = []
        destinations = []
        seen = set()
        for artifact in staged:
            self._safe_artifact_type(artifact.type)
            key = (artifact.type, artifact.variant)
            if key in seen:
                raise ValueError("阶段存在重复 artifact")
            seen.add(key)
            source = artifact.path.resolve()
            source.relative_to(staging_dir.resolve())
            if not source.is_file():
                raise FileNotFoundError(f"staging artifact 不存在：{source}")
            filename = self._safe_filename(source.name)
            target = self.run_dir(run["recording_id"], run["id"]) / "artifacts" / artifact.type / filename
            digest = FileDigest(source, self.sha256_file(source), source.stat().st_size)
            relative_path = self._relative(target)
            manifest_artifacts.append({
                "type": artifact.type,
                "variant": artifact.variant,
                "filename": filename,
                "sha256": digest.sha256,
                "size_bytes": digest.size_bytes,
                "relative_path": relative_path,
            })
            prepared.append(PreparedArtifact(artifact.type, relative_path, digest.sha256, digest.size_bytes, artifact.variant))
            destinations.append((source, target, digest))
        manifest = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "run_id": run["id"],
            "stage": attempt["stage"],
            "artifacts": manifest_artifacts,
        }
        manifest_path = staging_dir / "manifest.json"
        self._write_json_durable(manifest_path, manifest)
        self.repository.prepare_attempt(attempt_id, self._relative(manifest_path), prepared)
        for source, target, digest in destinations:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if self.sha256_file(target) != digest.sha256:
                    raise FileExistsError(f"不可变 artifact 已存在且 hash 不匹配：{target}")
                source.unlink()
            else:
                os.replace(source, target)
            if self.sha256_file(target) != digest.sha256 or target.stat().st_size != digest.size_bytes:
                raise RuntimeError(f"artifact 提交后的 hash 校验失败：{target}")
        self.repository.commit_attempt(attempt_id)
        return [target for _, target, _ in destinations]

    def recover_attempts(self) -> list[str]:
        recovered = []
        for attempt in self.repository.pending_attempts():
            staging_dir = self._path_from_relative(attempt["staging_dir"])
            manifest_path = staging_dir / "manifest.json"
            if attempt["status"] == "running" or not manifest_path.is_file():
                self.repository.abandon_attempt(attempt["id"])
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema_version") != 1 or manifest.get("attempt_id") != attempt["id"]:
                    raise ValueError("manifest 不匹配")
                for item in manifest["artifacts"]:
                    source = staging_dir / self._safe_filename(item["filename"])
                    target = self._path_from_relative(item["relative_path"])
                    if target.exists():
                        if self.sha256_file(target) != item["sha256"] or target.stat().st_size != item["size_bytes"]:
                            raise ValueError("已提交 artifact 校验失败")
                    elif source.is_file() and self.sha256_file(source) == item["sha256"] and source.stat().st_size == item["size_bytes"]:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, target)
                    else:
                        raise ValueError("artifact 不完整")
                self.repository.commit_attempt(attempt["id"])
                recovered.append(attempt["id"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                self.repository.abandon_attempt(attempt["id"])
        return recovered

    def delete_managed_recording(self, recording_id: str) -> None:
        recording = self.repository.recording(recording_id)
        if recording is None:
            raise KeyError(f"录音不存在：{recording_id}")
        if recording["storage_kind"] != "managed":
            raise ValueError("仅能通过 managed 删除流程删除受控录音")
        directory = self.recording_dir(recording_id).resolve()
        directory.relative_to(self.recordings_dir.resolve())
        files = [self._relative(path) for path in sorted(directory.rglob("*")) if path.is_file()] if directory.exists() else []
        operation = self.repository.create_deletion_operation(
            recording_id=recording_id,
            manifest={"schema_version": 1, "files": files},
        )
        try:
            if directory.exists():
                shutil.rmtree(directory)
        except OSError:
            self.repository.fail_deletion_operation(operation["id"], "受控文件删除失败")
            raise
        self.repository.delete_recording_row(recording_id)

    @staticmethod
    def _write_json_durable(path: Path, value: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
