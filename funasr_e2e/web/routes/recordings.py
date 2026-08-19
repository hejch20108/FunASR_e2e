from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from pydantic import BaseModel, Field

from funasr_e2e.web.errors import ApiError


router = APIRouter(prefix="/api/recordings", tags=["recordings"])


def _services(request: Request):
    return request.app.state.services


class RunRequest(BaseModel):
    preset_spk_num: int | None = Field(default=None, ge=1)


class BatchRunRequest(BaseModel):
    recording_ids: list[str] = Field(min_length=1, max_length=200)
    preset_spk_num: int | None = Field(default=None, ge=1)


class DisplayNameRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=160)


def _recording_payload(recording: Any, run: Any | None = None, job: Any | None = None) -> dict[str, Any]:
    return {
        "id": recording["id"],
        "original_filename": recording["original_filename"],
        "display_name": recording["display_name"],
        "extension": recording["extension"],
        "size_bytes": recording["size_bytes"],
        "duration_ms": recording["duration_ms"],
        "created_at": recording["created_at"],
        "run": None if run is None else {
            "id": run["id"], "version": run["version"], "status": run["status"], "phase": run["phase"],
            "preset_spk_num": run["preset_spk_num"],
        },
        "job": None if job is None else _job_payload(job),
    }


def _job_payload(job: Any) -> dict[str, Any]:
    return {
        "id": job["id"], "run_id": job["run_id"], "kind": job["kind"], "queue_seq": job["queue_seq"],
        "status": job["status"], "phase": job["phase"], "progress_completed": job["progress_completed"],
        "progress_total": job["progress_total"], "error_code": job["error_code"], "error_message": job["error_message"],
        "cancel_requested_at": job["cancel_requested_at"], "created_at": job["created_at"], "updated_at": job["updated_at"],
    }


def _validated_filename(filename: str | None) -> tuple[str, str, str]:
    if not filename or Path(filename).name != filename or any(ord(character) < 32 for character in filename):
        raise ApiError(status_code=400, code="INVALID_FILENAME", message="上传文件名非法")
    suffix = Path(filename).suffix.lower()
    if not suffix:
        raise ApiError(status_code=400, code="INVALID_EXTENSION", message="上传文件必须包含扩展名")
    display_name = Path(filename).stem.strip() or "未命名录音"
    return filename, display_name, suffix


@router.post("/uploads")
def upload_recording(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    auto_start: Annotated[bool, Form()] = True,
    preset_spk_num: Annotated[int | None, Form(ge=1)] = None,
) -> dict[str, Any]:
    services = _services(request)
    original_filename, display_name, extension = _validated_filename(file.filename)
    supported_extensions = {str(value).lower() for value in services.settings["audio"]["supported_extensions"]}
    if extension not in supported_extensions:
        raise ApiError(status_code=400, code="UNSUPPORTED_AUDIO", message="不支持的音频格式")
    upload = services.store.stream_upload(file.file)
    try:
        recording, created = services.repository.create_or_get_recording(
            original_filename=original_filename,
            display_name=display_name,
            extension=extension,
            sha256=upload.sha256,
            size_bytes=upload.size_bytes,
            source_path="runtime/uploads/pending.part",
        )
        if created:
            services.store.commit_source(recording_id=recording["id"], upload_path=upload.path, extension=extension)
            recording = services.repository.recording(recording["id"])
            assert recording is not None
        else:
            upload.path.unlink(missing_ok=True)
        run = job = None
        if auto_start and created:
            run, job = services.repository.create_run_and_enqueue_funasr(
                recording_id=recording["id"],
                preset_spk_num=preset_spk_num,
                settings_snapshot=services.settings,
            )
        return {"created": created, "recording": _recording_payload(recording, run, job)}
    except ValueError as error:
        upload.path.unlink(missing_ok=True)
        raise ApiError(status_code=409, code="RECORDING_CONFLICT", message=str(error)) from error
    finally:
        file.file.close()


@router.get("")
def list_recordings(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    query: str | None = None,
) -> dict[str, Any]:
    rows, total = _services(request).repository.list_recordings(query=query, page=page, page_size=page_size)
    return {
        "items": [{
            "id": row["id"], "original_filename": row["original_filename"], "display_name": row["display_name"],
            "extension": row["extension"], "size_bytes": row["size_bytes"], "duration_ms": row["duration_ms"],
            "created_at": row["created_at"], "run_status": row["run_status"], "phase": row["job_phase"] or row["run_phase"],
            "job_status": row["job_status"], "progress_completed": row["progress_completed"],
            "progress_total": row["progress_total"], "error_code": row["error_code"],
            "error_message": row["error_message"], "final_exists": bool(row["final_exists"]),
        } for row in rows],
        "page": page, "page_size": page_size, "total": total,
    }


@router.get("/{recording_id}")
def recording_detail(recording_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    try:
        recording, run, job = services.repository.recording_detail(recording_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    artifacts = [] if run is None else [{
        "id": artifact["id"], "type": artifact["type"], "variant": artifact["variant"],
        "size_bytes": artifact["size_bytes"], "sha256": artifact["sha256"],
    } for artifact in services.repository.committed_artifacts(run["id"])]
    return {"recording": _recording_payload(recording, run, job), "artifacts": artifacts}


@router.patch("/{recording_id}")
def update_recording(recording_id: str, body: DisplayNameRequest, request: Request) -> dict[str, Any]:
    display_name = body.display_name.strip()
    if not display_name or any(ord(character) < 32 for character in display_name):
        raise ApiError(status_code=400, code="INVALID_DISPLAY_NAME", message="显示名称非法")
    services = _services(request)
    try:
        recording, _, job = services.repository.recording_detail(recording_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    if job is not None and job["status"] in {"queued", "running", "cancel_requested"}:
        raise ApiError(status_code=409, code="ACTIVE_JOB_EXISTS", message="处理中不能修改显示名称")
    updated = services.repository.update_recording_display_name(recording["id"], display_name)
    return {"id": updated["id"], "display_name": updated["display_name"]}


@router.delete("/{recording_id}")
def delete_recording(
    recording_id: str,
    request: Request,
    confirm: bool = False,
) -> dict[str, bool]:
    if not confirm:
        raise ApiError(status_code=400, code="DELETE_CONFIRMATION_REQUIRED", message="必须明确确认删除")
    services = _services(request)
    try:
        recording, _, job = services.repository.recording_detail(recording_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    if job is not None and job["status"] in {"queued", "running", "cancel_requested"}:
        raise ApiError(status_code=409, code="ACTIVE_JOB_EXISTS", message="处理中不能删除录音")
    services.store.delete_managed_recording(recording_id)
    return {"deleted": True}


@router.post("/batch/funasr-jobs")
def start_batch_funasr(body: BatchRunRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    recordings = [services.repository.recording(recording_id) for recording_id in body.recording_ids]
    if any(recording is None for recording in recordings):
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="批量请求包含不存在的录音")
    try:
        created = services.repository.create_runs_and_enqueue_funasr(
            recording_ids=body.recording_ids,
            preset_spk_num=body.preset_spk_num,
            settings_snapshot=services.settings,
        )
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="批量请求包含不存在的录音") from error
    except ValueError as error:
        raise ApiError(status_code=409, code="BATCH_NOT_STARTABLE", message="批量请求不能启动") from error
    return {"jobs": [{"run_id": run["id"], "job": _job_payload(job)} for run, job in created]}


@router.post("/{recording_id}/funasr-jobs")
def start_funasr(recording_id: str, body: RunRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    try:
        run, job = services.repository.create_run_and_enqueue_funasr(
            recording_id=recording_id,
            preset_spk_num=body.preset_spk_num,
            settings_snapshot=services.settings,
        )
    except KeyError as error:
        raise ApiError(status_code=404, code="RECORDING_NOT_FOUND", message="录音不存在") from error
    except ValueError as error:
        raise ApiError(status_code=409, code="ACTIVE_JOB_EXISTS", message="录音已有活动任务") from error
    return {"run_id": run["id"], "job": _job_payload(job)}
