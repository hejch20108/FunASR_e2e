from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from funasr_e2e.pipeline.service import build_speaker_summaries
from funasr_e2e.web.errors import ApiError
from scripts.postprocess_funasr_transcript import load_sentences

from .artifacts import _current_run, _verified_artifact_path
from .recordings import _services


router = APIRouter(prefix="/api/recordings", tags=["speakers"])


class SpeakerMappingRequest(BaseModel):
    entries: dict[str, str] = Field(min_length=1, max_length=100)


def _speaker_prefix(run: Any) -> str | None:
    try:
        value = json.loads(run["settings_json"])["postprocess"]["speaker_prefix"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, str) else None


@router.get("/{recording_id}/speaker-summary")
def speaker_summary(recording_id: str, request: Request) -> dict[str, Any]:
    run = _current_run(request, recording_id)
    _, raw_path = _verified_artifact_path(request, run["id"], "raw_json")
    summaries = build_speaker_summaries(load_sentences(raw_path))
    return {"run_id": run["id"], "items": [
        {
            "anonymous_label": summary.anonymous_label,
            "occurrence_count": summary.occurrence_count,
            "start_ms": summary.start_ms,
            "end_ms": summary.end_ms,
            "excerpts": [
                {"start_ms": excerpt.start_ms, "end_ms": excerpt.end_ms, "text": excerpt.text}
                for excerpt in summary.excerpts
            ],
        }
        for summary in summaries
    ]}


@router.get("/{recording_id}/speaker-mapping")
def get_speaker_mapping(recording_id: str, request: Request) -> dict[str, Any]:
    run = _current_run(request, recording_id)
    mapping, entries = _services(request).repository.latest_speaker_mapping(run["id"])
    return {
        "run_id": run["id"],
        "version": None if mapping is None else mapping["version"],
        "speaker_prefix": _speaker_prefix(run),
        "entries": {entry["anonymous_label"]: entry["display_name"] for entry in entries},
    }


@router.post("/{recording_id}/speaker-mapping")
def save_speaker_mapping(recording_id: str, body: SpeakerMappingRequest, request: Request) -> dict[str, Any]:
    run = _current_run(request, recording_id)
    try:
        mapping = _services(request).repository.save_speaker_mapping(run["id"], body.entries)
    except ValueError as error:
        raise ApiError(status_code=400, code="INVALID_SPEAKER_MAPPING", message="speaker 映射格式非法") from error
    return {"run_id": run["id"], "version": mapping["version"]}
