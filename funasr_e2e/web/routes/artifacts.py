from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Header, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from funasr_e2e.web.errors import ApiError

from .recordings import _services


router = APIRouter(prefix="/api/recordings", tags=["artifacts"])

_AUDIO_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}


def _current_run(request: Request, recording_id: str):
    try:
        _, run, _ = _services(request).repository.recording_detail(recording_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    if run is None:
        raise ApiError(status_code=404, code="RUN_NOT_FOUND", message="录音尚无运行版本")
    return run


def _render_final_display(canonical_text: str, mapping: dict[str, str], speaker_prefix: str | None) -> str:
    rendered = canonical_text
    for anonymous_label, display_name in mapping.items():
        if not display_name or not anonymous_label.startswith("SPEAKER_"):
            continue
        suffix = anonymous_label.removeprefix("SPEAKER_")
        labels = [anonymous_label]
        if speaker_prefix is not None:
            labels.append(f"{speaker_prefix}{suffix}")
        for label in labels:
            pattern = rf"(?m)^((?:\[[^\n]+\]\s+)?){re.escape(label)}(?=(?:【待回听】)?：)"
            rendered = re.sub(pattern, lambda match: f"{match.group(1)}{display_name}", rendered)
    return rendered


def _verified_artifact_path(request: Request, run_id: str, artifact_type: str, variant: str = "canonical") -> tuple[Any, Path]:
    services = _services(request)
    artifact = services.repository.committed_artifact(run_id=run_id, artifact_type=artifact_type, variant=variant)
    if artifact is None:
        raise ApiError(status_code=404, code="ARTIFACT_NOT_FOUND", message="产物不存在")
    try:
        path = services.store.artifact_path(artifact["relative_path"])
    except ValueError:
        raise ApiError(status_code=409, code="ARTIFACT_INTEGRITY_FAILED", message="产物完整性校验失败") from None
    if not path.is_file() or path.stat().st_size != artifact["size_bytes"] or services.store.sha256_file(path) != artifact["sha256"]:
        raise ApiError(status_code=409, code="ARTIFACT_INTEGRITY_FAILED", message="产物完整性校验失败")
    return artifact, path


def _single_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise ApiError(status_code=416, code="INVALID_RANGE", message="仅支持单个字节范围")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator or (not start_text and not end_text):
        raise ApiError(status_code=416, code="INVALID_RANGE", message="字节范围非法")
    try:
        if start_text:
            start = int(start_text)
            end = size - 1 if not end_text else int(end_text)
        else:
            length = int(end_text)
            start, end = max(0, size - length), size - 1
    except ValueError as error:
        raise ApiError(status_code=416, code="INVALID_RANGE", message="字节范围非法") from error
    if start < 0 or end < start or start >= size:
        raise ApiError(status_code=416, code="INVALID_RANGE", message="字节范围不可用")
    return start, min(end, size - 1)


def _range_stream(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


@router.get("/{recording_id}/artifacts")
def list_artifacts(recording_id: str, request: Request) -> dict[str, Any]:
    run = _current_run(request, recording_id)
    artifacts = _services(request).repository.committed_artifacts(run["id"])
    return {"run_id": run["id"], "items": [
        {"id": item["id"], "type": item["type"], "variant": item["variant"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in artifacts
    ]}


@router.get("/{recording_id}/artifacts/{artifact_type}")
def download_artifact(recording_id: str, artifact_type: str, request: Request) -> FileResponse:
    run = _current_run(request, recording_id)
    _, path = _verified_artifact_path(request, run["id"], artifact_type)
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/{recording_id}/download/final")
def download_final(recording_id: str, request: Request, display_names: bool = False) -> Response:
    run = _current_run(request, recording_id)
    _, path = _verified_artifact_path(request, run["id"], "final")
    if not display_names:
        return FileResponse(path, media_type="text/plain; charset=utf-8", filename="final.txt")
    _, entries = _services(request).repository.latest_speaker_mapping(run["id"])
    mapping = {entry["anonymous_label"]: entry["display_name"] for entry in entries if entry["display_name"]}
    try:
        speaker_prefix = json.loads(run["settings_json"])["postprocess"]["speaker_prefix"]
    except (KeyError, TypeError, json.JSONDecodeError):
        speaker_prefix = None
    rendered = _render_final_display(
        path.read_text(encoding="utf-8"),
        mapping,
        speaker_prefix if isinstance(speaker_prefix, str) else None,
    )
    return Response(
        rendered,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="final-display.txt"'},
    )


@router.get("/{recording_id}/audit-summary")
def audit_summary(recording_id: str, request: Request) -> dict[str, Any]:
    run = _current_run(request, recording_id)
    services = _services(request)
    artifact_types = {item["type"] for item in services.repository.committed_artifacts(run["id"])}
    result: dict[str, Any] = {"speaker_review": None, "final": None}
    if "speaker_review" in artifact_types:
        try:
            _, review_path = _verified_artifact_path(request, run["id"], "speaker_review")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            reviewed_spans = review.get("reviewed_spans", [])
            review_queue = review.get("review_queue", [])
            full_review = review.get("full_review", {})
            result["speaker_review"] = {
                "review_required_count": sum(bool(item.get("review_required")) for item in reviewed_spans if isinstance(item, dict)),
                "review_queue_count": len(review_queue) if isinstance(review_queue, list) else 0,
                "full_review_fallback": bool(full_review.get("failed")) if isinstance(full_review, dict) else False,
            }
        except (OSError, ValueError, json.JSONDecodeError):
            result["speaker_review"] = {"available": False}
    if "final_audit" in artifact_types:
        try:
            _, audit_path = _verified_artifact_path(request, run["id"], "final_audit")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            integrity = audit.get("integrity", {})
            result["final"] = {
                "fallback_chunk_count": int(integrity.get("fallback_chunk_count", 0)) if isinstance(integrity, dict) else 0,
                "warning_chunk_count": int(integrity.get("warning_chunk_count", 0)) if isinstance(integrity, dict) else 0,
                "warning_count": int(integrity.get("warning_count", 0)) if isinstance(integrity, dict) else 0,
            }
        except (OSError, ValueError, json.JSONDecodeError):
            result["final"] = {"available": False}
    return result


@router.get("/{recording_id}/audio")
def stream_audio(recording_id: str, request: Request, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    services = _services(request)
    try:
        recording, _, _ = services.repository.recording_detail(recording_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    try:
        path = services.store.artifact_path(recording["source_path"])
    except ValueError:
        raise ApiError(status_code=404, code="AUDIO_NOT_FOUND", message="音频不存在") from None
    if not path.is_file():
        raise ApiError(status_code=404, code="AUDIO_NOT_FOUND", message="音频不存在")
    size = path.stat().st_size
    selected = _single_range(range_header, size)
    headers = {"Accept-Ranges": "bytes"}
    media_type = _AUDIO_MEDIA_TYPES.get(recording["extension"].lower(), "application/octet-stream")
    if selected is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_range_stream(path, 0, size), headers=headers, media_type=media_type)
    start, end = selected
    length = end - start + 1
    headers.update({"Content-Length": str(length), "Content-Range": f"bytes {start}-{end}/{size}"})
    return StreamingResponse(_range_stream(path, start, length), status_code=206, headers=headers, media_type=media_type)


@router.get("/{recording_id}/download/all")
def download_all(recording_id: str, request: Request) -> FileResponse:
    run = _current_run(request, recording_id)
    services = _services(request)
    artifacts = services.repository.committed_artifacts(run["id"])
    if not artifacts:
        raise ApiError(status_code=404, code="ARTIFACT_NOT_FOUND", message="尚无可下载产物")
    exports_dir = services.store.runtime_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="export-", suffix=".zip", dir=exports_dir)
    os.close(descriptor)
    archive_path = Path(name)
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in artifacts:
                _, path = _verified_artifact_path(request, run["id"], artifact["type"], artifact["variant"])
                archive.write(path, arcname=f"artifacts/{artifact['type']}/{artifact['variant']}/{path.name}")
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename="artifacts.zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )
