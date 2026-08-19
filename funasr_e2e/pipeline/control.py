from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class PipelineCancelled(Exception):
    """在安全检查点终止当前流水线。"""


@dataclass(frozen=True)
class PipelineEvent:
    stage: str
    event: str
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    details: dict[str, str | int | float | bool | None] = field(default_factory=dict)


ProgressCallback = Callable[[PipelineEvent], None]
CancelCheck = Callable[[], None]


def check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


def report(progress_callback: ProgressCallback | None, event: PipelineEvent) -> None:
    if progress_callback is not None:
        progress_callback(event)
