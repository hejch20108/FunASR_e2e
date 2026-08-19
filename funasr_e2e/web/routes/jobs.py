from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from funasr_e2e.web.errors import ApiError

from .recordings import _job_payload, _services


router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_FORCE_STOP_GRACE_SECONDS = 10


class ReorderRequest(BaseModel):
    job_ids: list[str] = Field(min_length=1, max_length=200)


@router.get("/queue")
def queue(request: Request) -> dict[str, Any]:
    return {"jobs": [
        {**_job_payload(job), "display_name": job["display_name"], "original_filename": job["original_filename"]}
        for job in _services(request).repository.queued_jobs()
    ]}


@router.post("/reorder")
def reorder_jobs(body: ReorderRequest, request: Request) -> dict[str, Any]:
    try:
        jobs = _services(request).repository.reorder_queued_jobs(body.job_ids)
    except ValueError as error:
        raise ApiError(status_code=409, code="QUEUE_REORDER_REJECTED", message="队列顺序不合法") from error
    return {"jobs": [_job_payload(job) for job in jobs]}


@router.get("/{job_id}")
def job_detail(job_id: str, request: Request) -> dict[str, Any]:
    job = _services(request).repository.job(job_id)
    if job is None:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在")
    return {"job": _job_payload(job)}


@router.get("/{job_id}/diagnostics")
def job_diagnostics(job_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    if services.repository.job(job_id) is None:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在")
    events = services.repository.task_events(job_id)
    return {"events": [
        {"id": event["id"], "stage": event["stage"], "event": event["event"], "completed": event["completed"],
         "total": event["total"], "message": event["message"], "details": json.loads(event["details_json"]), "created_at": event["created_at"]}
        for event in events
    ]}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        job = _services(request).repository.request_cancel(job_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在") from error
    except ValueError as error:
        raise ApiError(status_code=409, code="JOB_NOT_CANCELLABLE", message="当前任务不能取消") from error
    return {"job": _job_payload(job)}


@router.post("/{job_id}/force-stop")
def force_stop_job(job_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    job = services.repository.job(job_id)
    if job is None:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在")
    if job["status"] != "cancel_requested" or job["cancel_requested_at"] is None:
        raise ApiError(status_code=409, code="FORCE_STOP_NOT_READY", message="请先取消正在执行的任务")
    try:
        requested_at = datetime.fromisoformat(job["cancel_requested_at"].replace("Z", "+00:00"))
    except ValueError:
        raise ApiError(status_code=409, code="FORCE_STOP_NOT_READY", message="取消请求时间不可用") from None
    elapsed = (datetime.now(timezone.utc) - requested_at).total_seconds()
    if elapsed < _FORCE_STOP_GRACE_SECONDS:
        raise ApiError(
            status_code=409,
            code="FORCE_STOP_GRACE_REQUIRED",
            message=f"请在取消请求 {_FORCE_STOP_GRACE_SECONDS} 秒后再强制停止",
        )
    try:
        status = services.supervisor.force_stop_job(job_id)
    except ValueError as error:
        raise ApiError(status_code=409, code="FORCE_STOP_NOT_READY", message="任务不再适合强制停止") from error
    return {"worker_running": status.running}


@router.get("/{job_id}/recommended-recovery")
def recommended_recovery(job_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    try:
        recommendation = services.repository.recovery_recommendation(job_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在") from error
    return {"action": recommendation.action, "phase": recommendation.phase}


@router.post("/{job_id}/recommended-recovery")
def apply_recommended_recovery(job_id: str, request: Request) -> dict[str, Any]:
    try:
        job = _services(request).repository.requeue_recoverable_job(job_id)
    except KeyError as error:
        raise ApiError(status_code=404, code="JOB_NOT_FOUND", message="任务不存在") from error
    except ValueError as error:
        raise ApiError(status_code=409, code="JOB_NOT_RECOVERABLE", message="当前任务不能按推荐方式恢复") from error
    return {"job": _job_payload(job)}
