from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


JOB_TRANSITIONS = {
    "queued": {"running", "cancelled"},
    "running": {"cancel_requested", "cancelled", "succeeded", "failed", "interrupted", "force_stopped"},
    "cancel_requested": {"cancelled", "succeeded", "failed", "interrupted", "force_stopped"},
    "interrupted": {"queued", "cancelled", "force_stopped"},
    "force_stopped": {"queued", "cancelled"},
    "failed": {"queued", "cancelled"},
    "cancelled": set(),
    "succeeded": set(),
}

RUN_TRANSITIONS = {
    "queued": {"running", "cancelled", "failed", "interrupted"},
    "running": {"queued", "waiting_speaker", "completed", "failed", "cancelled", "interrupted"},
    "waiting_speaker": {"queued", "running", "cancelled", "failed", "interrupted"},
    "interrupted": {"queued", "running", "cancelled", "failed"},
    "failed": {"queued", "cancelled"},
    "cancelled": set(),
    "completed": set(),
}

STAGE_ORDER = ("funasr", "evidence", "speaker_review", "cleaned", "final")


@dataclass(frozen=True)
class RecoveryRecommendation:
    action: str
    phase: str | None


def require_transition(current: str, target: str, transitions: dict[str, set[str]], label: str) -> None:
    if target not in transitions.get(current, set()):
        raise ValueError(f"不允许 {label} 状态从 {current} 变为 {target}")


def require_job_transition(current: str, target: str) -> None:
    require_transition(current, target, JOB_TRANSITIONS, "任务")


def require_run_transition(current: str, target: str) -> None:
    require_transition(current, target, RUN_TRANSITIONS, "运行")


def next_phase(committed_artifact_types: Iterable[str]) -> RecoveryRecommendation:
    artifacts = set(committed_artifact_types)
    if "raw_json" not in artifacts:
        return RecoveryRecommendation("run_funasr", "funasr")
    if "evidence" not in artifacts:
        return RecoveryRecommendation("generate_evidence", "evidence")
    if "speaker_review" not in artifacts or "reviewed" not in artifacts:
        return RecoveryRecommendation("continue_processing", "speaker_review")
    if "cleaned" not in artifacts:
        return RecoveryRecommendation("continue_processing", "cleaned")
    if "final" not in artifacts or "final_audit" not in artifacts:
        return RecoveryRecommendation("continue_processing", "final")
    return RecoveryRecommendation("none", None)
