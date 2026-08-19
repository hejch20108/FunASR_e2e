from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scripts.postprocess_funasr_transcript import (
    ReviewedSpan,
    Sentence,
    load_cleaning_config,
    load_sentences,
    merge_sentences,
    render_blocks,
    render_evidence_transcript,
    review_sentences,
    reviewed_spans_from_audit,
    write_final_transcript,
)
from scripts.review_funasr_speakers import (
    SpeakerReviewResult,
    run_speaker_review,
    sha256_file,
)

from .control import CancelCheck, PipelineEvent, ProgressCallback, check_cancel, report


class FunASRModel(Protocol):
    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class LLMCredentials:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class FunASRStageResult:
    raw_json_path: Path
    sentences: list[Sentence]


@dataclass(frozen=True)
class SpeakerExcerpt:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SpeakerSummary:
    anonymous_label: str
    occurrence_count: int
    start_ms: int
    end_ms: int
    excerpts: list[SpeakerExcerpt]


@dataclass(frozen=True)
class SpeakerReviewStageResult:
    review_result: SpeakerReviewResult
    speaker_review_path: Path
    reviewed_path: Path


def speaker_label(value: int | str) -> str:
    text = str(value)
    return text if text.startswith("SPEAKER_") else f"SPEAKER_{text}"


def _preset_speaker_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("preset_spk_num 必须为 null 或正整数")
    return value


def _load_hotwords(prompt_dir: Path) -> list[str]:
    path = prompt_dir / "hotwords.txt"
    if not path.exists():
        raise FileNotFoundError(f"热词文件不存在：{path}")
    seen: set[str] = set()
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#") and word not in seen:
            result.append(word)
            seen.add(word)
    return result


def run_funasr_stage(
    *,
    audio_path: Path,
    raw_json_path: Path,
    model: FunASRModel,
    funasr_config: dict[str, Any],
    prompt_dir: Path,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> FunASRStageResult:
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="funasr", event="started"))
    generate_kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "batch_size_s": funasr_config["batch_size_s"],
        "batch_size_threshold_s": funasr_config["batch_size_threshold_s"],
    }
    preset_spk_num = _preset_speaker_count(funasr_config.get("preset_spk_num"))
    if preset_spk_num is not None:
        generate_kwargs["preset_spk_num"] = preset_spk_num
    hotwords = _load_hotwords(prompt_dir)
    if hotwords:
        generate_kwargs["hotword"] = " ".join(hotwords)
    result = model.generate(**generate_kwargs)
    check_cancel(cancel_check)
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sentences = load_sentences(raw_json_path)
    if not sentences:
        raise ValueError("没有从 FunASR JSON 中解析到 sentence_info")
    review_sentences(sentences)
    report(progress_callback, PipelineEvent(stage="funasr", event="completed", details={"sentence_count": len(sentences)}))
    return FunASRStageResult(raw_json_path=raw_json_path, sentences=sentences)


def generate_evidence_stage(
    *,
    raw_json_path: Path,
    evidence_path: Path,
    speaker_prefix: str,
    keep_time: bool,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[Sentence]:
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="evidence", event="started"))
    sentences = load_sentences(raw_json_path)
    if not sentences:
        raise ValueError("没有从 FunASR JSON 中解析到 sentence_info")
    review_sentences(sentences)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(render_evidence_transcript(sentences, speaker_prefix, keep_time), encoding="utf-8")
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="evidence", event="completed", details={"sentence_count": len(sentences)}))
    return sentences


def build_speaker_summaries(sentences: list[Sentence], excerpt_limit: int = 3) -> list[SpeakerSummary]:
    grouped: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        if sentence.text.strip():
            grouped.setdefault(speaker_label(sentence.spk), []).append(sentence)
    summaries = []
    for label, items in sorted(grouped.items()):
        candidates = [items[0], max(items, key=lambda item: len(item.text)), items[-1]]
        excerpts: list[SpeakerExcerpt] = []
        seen: set[str] = set()
        for sentence in candidates:
            marker = sentence.source_id
            if marker not in seen and len(excerpts) < excerpt_limit:
                excerpts.append(SpeakerExcerpt(sentence.start, sentence.end, sentence.text))
                seen.add(marker)
        summaries.append(SpeakerSummary(label, len(items), items[0].start, items[-1].end, excerpts))
    return summaries


def run_speaker_review_stage(
    *,
    raw_json_path: Path,
    speaker_review_path: Path,
    reviewed_path: Path,
    prompt_dir: Path,
    speaker_review_config: dict[str, Any],
    postprocess_config: dict[str, Any],
    credentials: LLMCredentials,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> SpeakerReviewStageResult:
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="speaker_review", event="started"))
    sentences = load_sentences(raw_json_path)
    review_sentences(sentences)
    result = run_speaker_review(
        json_path=raw_json_path,
        sentences=sentences,
        prompt_dir=prompt_dir,
        config=speaker_review_config,
        base_url=credentials.base_url,
        api_key=credentials.api_key,
        default_model=credentials.model,
        speaker_prefix=postprocess_config["speaker_prefix"],
        keep_time=postprocess_config["keep_time"],
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    check_cancel(cancel_check)
    speaker_review_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_review_path.write_text(json.dumps(result.audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviewed_path.write_text(result.reviewed_text, encoding="utf-8")
    report(progress_callback, PipelineEvent(stage="speaker_review", event="completed", details={"review_queue_count": result.audit["integrity"]["review_queue_count"]}))
    return SpeakerReviewStageResult(result, speaker_review_path, reviewed_path)


def generate_cleaned_stage(
    *,
    speaker_review_path: Path,
    cleaned_path: Path,
    prompt_dir: Path,
    postprocess_config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> int:
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="cleaned", event="started"))
    audit = json.loads(speaker_review_path.read_text(encoding="utf-8"))
    spans = reviewed_spans_from_audit(audit)
    blocks = merge_sentences(spans, postprocess_config["max_gap_ms"], postprocess_config["max_chars"])
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text(
        render_blocks(
            blocks,
            postprocess_config["speaker_prefix"],
            postprocess_config["keep_time"],
            cleaned=True,
            cleaning_config=load_cleaning_config(prompt_dir),
        ),
        encoding="utf-8",
    )
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="cleaned", event="completed", details={"block_count": len(blocks)}))
    return len(blocks)


def validate_final_prerequisites(*, raw_json_path: Path, speaker_review_path: Path) -> list[ReviewedSpan]:
    audit = json.loads(speaker_review_path.read_text(encoding="utf-8"))
    integrity = audit.get("integrity")
    required = (
        "source_hash_verified",
        "per_source_reconstruction_passed",
        "global_reconstruction_passed",
        "order_passed",
        "coverage_passed",
        "allowed_speaker_passed",
        "unknown_from_explicit_review_passed",
    )
    if audit.get("schema_version") != 3 or not isinstance(integrity, dict) or not all(integrity.get(key) is True for key in required):
        raise ValueError("最终阅读整理只接受完整性校验全部通过的 schema v3 speaker review")
    if audit.get("run", {}).get("source_json_sha256") != sha256_file(raw_json_path):
        raise ValueError("原始 FunASR JSON 已变化")
    return reviewed_spans_from_audit(audit)


def generate_final_stage(
    *,
    raw_json_path: Path,
    speaker_review_path: Path,
    final_path: Path,
    final_audit_path: Path,
    prompt_dir: Path,
    postprocess_config: dict[str, Any],
    llm_config: dict[str, Any],
    credentials: LLMCredentials,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="final", event="started"))
    spans = validate_final_prerequisites(raw_json_path=raw_json_path, speaker_review_path=speaker_review_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    audit = write_final_transcript(
        spans=spans,
        final_path=final_path,
        final_audit_path=final_audit_path,
        max_gap_ms=postprocess_config["max_gap_ms"],
        max_chars=postprocess_config["max_chars"],
        speaker_prefix=postprocess_config["speaker_prefix"],
        keep_time=postprocess_config["keep_time"],
        prompt_dir=prompt_dir,
        base_url=credentials.base_url,
        api_key=credentials.api_key,
        model=credentials.model,
        enable_thinking=llm_config["enable_thinking"],
        chunk_size=llm_config["chunk_size"],
        max_retries=llm_config["max_retries"],
        source_json_sha256=sha256_file(raw_json_path),
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    check_cancel(cancel_check)
    report(progress_callback, PipelineEvent(stage="final", event="completed", details={"fallback_chunk_count": audit["integrity"]["fallback_chunk_count"]}))
    return audit
