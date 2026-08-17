#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from postprocess_funasr_transcript import (
    ReviewedSpan,
    Sentence,
    call_openai_compatible_chat,
    format_time,
    normalize_speaker,
)


OPERATIONS = {"KEEP", "REASSIGN", "SPLIT", "REVIEW_REQUIRED", "OVERLAP"}
FULL_OVERRIDE_OPERATIONS = {"REASSIGN", "REVIEW_REQUIRED", "OVERLAP"}
RISK_TYPES = {
    "ROLE_DISCONTINUITY",
    "AMBIGUOUS_TURN",
    "OVERLAP_OR_INTERRUPTION",
    "INTRA_SENTENCE_SWITCH",
}
REASON_CODES = {
    "SOURCE_SPK_PRIOR",
    "ROLE_CONSISTENCY",
    "STANCE_CONTINUITY",
    "TURN_TAKING",
    "QUESTION_RESPONSE",
    "ADDRESS_RESPONSE",
    "SHORT_ANSWER",
    "INTERRUPTION_CUE",
    "TIMESTAMP_GAP",
    "OVERLAP_SIGNAL",
    "AMBIGUOUS_CONTEXT",
    "INSUFFICIENT_BOUNDARY",
    "PASS_DISAGREEMENT",
}
PUNCTUATION = set("，,。！？；：、（）()“”‘’\"'…—- ")
STRONG_PUNCTUATION = set("。！？；：…—-")
INTERRUPTION_CUES = ("但是", "不是", "那", "对", "好", "可以", "不可以")
HIGH_IMPACT_FLAGS = {
    "涉及金额",
    "涉及日期",
    "劳动争议内容",
    "接受立场",
    "拒绝立场",
    "否定或限制",
    "承诺或履约",
}
ANSWER_HIGH_IMPACT_FLAGS = {"涉及金额", "接受立场", "拒绝立场", "否定或限制", "承诺或履约"}
SEMANTIC_FLAGS = HIGH_IMPACT_FLAGS | {"疑问表达", "机构立场", "个人诉求"}
SEMANTIC_PROMOTION_TYPES = {
    "STRUCTURE_HIGH_IMPACT",
    "SAME_LABEL_HIGH_IMPACT_QA",
    "SAME_LABEL_STANCE_CONFLICT",
    "SAME_LABEL_POSITION_CONFLICT",
}
CONSERVATIVE_KEEP_GUARD_SIGNAL_TYPES = {
    "SAME_LABEL_HIGH_IMPACT_QA",
    "SAME_LABEL_STANCE_CONFLICT",
    "SAME_LABEL_POSITION_CONFLICT",
    "LOCAL_LABEL_TURBULENCE_HIGH_IMPACT",
}
CONSERVATIVE_KEEP_GUARD_FLAGS = HIGH_IMPACT_FLAGS | {"机构立场", "个人诉求"}
MAX_SAME_LABEL_QA_LOOKBACK = 6
SAME_LABEL_QA_LEAD_IN = 13
SAME_LABEL_QA_FOLLOW_THROUGH = 8
MAX_LABEL_TURBULENCE_WINDOW_MS = 12_000
MIN_LABEL_TURBULENCE_TRANSITIONS = 3


@dataclass(frozen=True)
class BoundaryCandidate:
    boundary_id: str
    source_id: str
    char_offset: int
    estimated_time_ms: int | None
    time_method: str
    reliable: bool = True
    candidate_type: str = "endpoint"


@dataclass(frozen=True)
class RiskSignal:
    source_indexes: tuple[int, ...]
    signal_type: str
    score: int
    primary: bool


@dataclass(frozen=True)
class RiskSegment:
    segment_id: str
    component_id: str
    chunk_index: int
    chunk_count: int
    core_indexes: tuple[int, ...]
    context_indexes: tuple[int, ...]
    score: int
    signals: tuple[str, ...]
    cut_rule: str
    forced_cut: bool


@dataclass
class SpeakerReviewResult:
    spans: list[ReviewedSpan]
    audit: dict[str, Any]
    reviewed_text: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def speaker_label(spk: int | str) -> str:
    value = str(spk)
    return value if value.startswith("SPEAKER_") else f"SPEAKER_{value}"


def normalize_review_config(config: dict[str, Any], sentences: list[Sentence]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("speaker_review 配置必须是对象")
    if config.get("model") is not None:
        raise ValueError("speaker_review.model 必须为 null，模型只能从 .env 的 MODEL_NAME 读取")

    allowed_value = config.get("allowed_speakers")
    if allowed_value is None:
        allowed_speakers = sorted({speaker_label(sentence.spk) for sentence in sentences})
    elif isinstance(allowed_value, list) and all(isinstance(item, str) and item.strip() for item in allowed_value):
        allowed_speakers = [item.strip() for item in allowed_value]
    else:
        raise ValueError("speaker_review.allowed_speakers 必须为 null 或非空字符串列表")
    if not allowed_speakers or len(set(allowed_speakers)) != len(allowed_speakers):
        raise ValueError("speaker_review.allowed_speakers 必须包含不重复的至少一个说话人")

    unknown_label = config.get("unknown_label", "unknown")
    overlap_label = config.get("overlap_label", "overlap")
    if not isinstance(unknown_label, str) or not unknown_label:
        raise ValueError("speaker_review.unknown_label 必须为非空字符串")
    if not isinstance(overlap_label, str) or not overlap_label:
        raise ValueError("speaker_review.overlap_label 必须为非空字符串")
    if unknown_label == overlap_label or unknown_label in allowed_speakers or overlap_label in allowed_speakers:
        raise ValueError("unknown、overlap 与 allowed_speakers 必须互不重复")

    normalized = {
        "enabled": bool(config.get("enabled", True)),
        "enable_thinking": config.get("enable_thinking", False),
        "context_size": config.get("context_size", 4),
        "max_risk_core_sentences": config.get("max_risk_core_sentences", 12),
        "max_boundary_candidates": config.get("max_boundary_candidates", 16),
        "max_workers": config.get("max_workers", 4),
        "max_retries": config.get("max_retries", 3),
        "request_timeout_s": config.get("request_timeout_s", 90),
        "auto_apply_confidence": config.get("auto_apply_confidence", 0.90),
        "allowed_speakers": allowed_speakers,
        "unknown_label": unknown_label,
        "overlap_label": overlap_label,
        "failure_policy": config.get("failure_policy", "keep_original"),
    }
    for key in ("max_risk_core_sentences", "max_boundary_candidates", "max_workers", "max_retries", "request_timeout_s"):
        if isinstance(normalized[key], bool) or not isinstance(normalized[key], int) or normalized[key] <= 0:
            raise ValueError(f"speaker_review.{key} 必须为正整数")
    if normalized["max_boundary_candidates"] < 3:
        raise ValueError("speaker_review.max_boundary_candidates 至少为 3")
    if isinstance(normalized["context_size"], bool) or not isinstance(normalized["context_size"], int) or normalized["context_size"] < 0:
        raise ValueError("speaker_review.context_size 必须为非负整数")
    if not isinstance(normalized["enable_thinking"], bool):
        raise ValueError("speaker_review.enable_thinking 必须为布尔值")
    confidence = normalized["auto_apply_confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("speaker_review.auto_apply_confidence 必须在 0 到 1 之间")
    if normalized["failure_policy"] not in {"fail_closed", "keep_original"}:
        raise ValueError("speaker_review.failure_policy 只支持 fail_closed 或 keep_original")
    return normalized


def load_review_prompt_template(prompt_dir: Path) -> str:
    path = prompt_dir / "speaker_review_prompt_template.txt"
    if not path.exists():
        raise FileNotFoundError(f"说话人复核提示词模板不存在：{path}")
    template = path.read_text(encoding="utf-8")
    if "{{ review_input }}" not in template:
        raise ValueError(f"说话人复核提示词模板必须包含占位符 {{{{ review_input }}}}：{path}")
    return template


def build_review_prompt(template: str, review_input: dict[str, Any]) -> str:
    return template.replace("{{ review_input }}", json.dumps(review_input, ensure_ascii=False, separators=(",", ":")))


def parse_json_response(response: str) -> dict[str, Any]:
    value = response.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, count=1, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value, count=1)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("LLM 返回必须是 JSON 对象")
    return parsed


def call_json_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int,
    validator: Callable[[dict[str, Any]], Any],
    timeout_seconds: int,
    enable_thinking: bool = False,
) -> Any:
    last_error: Exception | None = None
    retry_prompt = prompt
    for attempt in range(1, max_retries + 1):
        try:
            response = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=retry_prompt,
                enable_thinking=enable_thinking,
                response_format={"type": "json_object"},
                timeout_seconds=timeout_seconds,
            )
            return validator(parse_json_response(response))
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                retry_prompt = f"{prompt}\n\n上次输出未通过本地校验。只修复以下错误并重新输出完整 JSON：{type(error).__name__}: {error}"
    raise RuntimeError(f"说话人复核请求失败，已重试 {max_retries} 次：{last_error}") from last_error


def serialize_sentence(sentence: Sentence, boundaries: list[BoundaryCandidate] | None = None, include_flags: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_id": sentence.source_id,
        "start_ms": sentence.start,
        "end_ms": sentence.end,
        "source_speaker": speaker_label(sentence.spk),
        "text": sentence.text,
    }
    if include_flags:
        value["review_flags"] = sentence.review_flags
    if boundaries is not None:
        value["boundary_candidates"] = [
            {
                "boundary_id": item.boundary_id,
                "char_offset": item.char_offset,
                "estimated_time_ms": item.estimated_time_ms,
                "time_method": item.time_method,
                "reliable": item.reliable,
                "candidate_type": item.candidate_type,
            }
            for item in boundaries
        ]
    return value


def validate_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item in REASON_CODES for item in value):
        raise ValueError("reason_codes 必须为非空且受限的字符串数组")
    return list(dict.fromkeys(value))


def validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("confidence 必须在 0 到 1 之间")
    return float(value)


def validate_registry_response(payload: dict[str, Any], valid_ids: set[str], allowed_speakers: set[str]) -> dict[str, Any]:
    raw_registry = payload.get("speaker_registry")
    if not isinstance(raw_registry, dict) or not isinstance(raw_registry.get("speakers"), list):
        raise ValueError("响应必须包含 speaker_registry.speakers")
    speakers = raw_registry["speakers"]
    if len(speakers) != len(allowed_speakers):
        raise ValueError("registry 必须恰好覆盖允许的所有 speaker")
    result = []
    seen = set()
    for item in speakers:
        if not isinstance(item, dict):
            raise ValueError("registry speaker 项必须是对象")
        speaker_id = item.get("speaker_id")
        role_summary = item.get("role_summary")
        evidence_ids = item.get("evidence_source_ids")
        if speaker_id not in allowed_speakers or speaker_id in seen:
            raise ValueError("registry speaker_id 非法或重复")
        if not isinstance(role_summary, str) or not role_summary or len(role_summary) > 240:
            raise ValueError("registry role_summary 必须为不超过 240 字符的非空字符串")
        if not isinstance(evidence_ids, list) or not 2 <= len(evidence_ids) <= 8:
            raise ValueError("registry evidence_source_ids 必须包含 2 到 8 个引用")
        if len(set(evidence_ids)) != len(evidence_ids) or not all(item_id in valid_ids for item_id in evidence_ids):
            raise ValueError("registry evidence_source_ids 非法或重复")
        result.append(
            {
                "speaker_id": speaker_id,
                "role_summary": role_summary,
                "evidence_source_ids": evidence_ids,
                "confidence": validate_confidence(item.get("confidence")),
                "reason_codes": validate_reason_codes(item.get("reason_codes")),
            }
        )
        seen.add(speaker_id)
    return {"speakers": result}


def validate_full_review_response(
    payload: dict[str, Any],
    sentences: list[Sentence],
    allowed_speakers: set[str],
    unknown_label: str,
    overlap_label: str,
    max_risk_core_sentences: int,
) -> dict[str, Any]:
    valid_ids = {sentence.source_id for sentence in sentences}
    indexes = {sentence.source_id: index for index, sentence in enumerate(sentences)}
    sources = {sentence.source_id: sentence for sentence in sentences}
    registry = validate_registry_response(payload, valid_ids, allowed_speakers)

    raw_overrides = payload.get("overrides")
    if not isinstance(raw_overrides, list):
        raise ValueError("full_review 响应必须包含 overrides 数组")
    overrides: dict[str, dict[str, Any]] = {}
    for raw in raw_overrides:
        if not isinstance(raw, dict):
            raise ValueError("override 项必须是对象")
        source_id = raw.get("source_id")
        operation = raw.get("operation")
        if source_id not in sources or source_id in overrides:
            raise ValueError("override source_id 非法或重复")
        if operation not in FULL_OVERRIDE_OPERATIONS:
            raise ValueError("full_review override 只能使用 REASSIGN、REVIEW_REQUIRED 或 OVERLAP")
        confidence = validate_confidence(raw.get("confidence"))
        reason_codes = validate_reason_codes(raw.get("reason_codes"))
        target_speaker = raw.get("target_speaker")
        if operation == "REASSIGN":
            if target_speaker not in allowed_speakers:
                raise ValueError("REASSIGN target_speaker 非法")
            if target_speaker == speaker_label(sources[source_id].spk):
                raise ValueError("REASSIGN 不得指向原 speaker")
        elif operation == "REVIEW_REQUIRED":
            if target_speaker not in (None, unknown_label):
                raise ValueError("REVIEW_REQUIRED 只能使用 unknown")
            target_speaker = unknown_label
        else:
            if target_speaker not in (None, overlap_label):
                raise ValueError("OVERLAP 只能使用 overlap")
            target_speaker = overlap_label
        overrides[source_id] = {
            "source_id": source_id,
            "operation": operation,
            "target_speaker": target_speaker,
            "parts": [],
            "confidence": confidence,
            "reason_codes": reason_codes,
            "review_required": operation in {"REVIEW_REQUIRED", "OVERLAP"},
            "source": "full_review",
        }

    raw_risk_items = payload.get("risk_items")
    if not isinstance(raw_risk_items, list):
        raise ValueError("full_review 响应必须包含 risk_items 数组")
    risk_items = []
    discarded_risk_items = []
    seen_risks = set()
    for raw in raw_risk_items:
        if not isinstance(raw, dict):
            raise ValueError("risk_item 必须是对象")
        source_ids = raw.get("source_ids")
        risk_type = raw.get("risk_type")
        if isinstance(source_ids, list) and (not source_ids or len(source_ids) > max_risk_core_sentences):
            discarded_risk_items.append(
                {
                    "risk_type": risk_type,
                    "source_count": len(source_ids),
                    "reason": "source_ids 长度超出风险片段限制",
                }
            )
            continue
        if not isinstance(source_ids, list):
            raise ValueError("risk_item.source_ids 必须为数组")
        if len(set(source_ids)) != len(source_ids) or not all(source_id in indexes for source_id in source_ids):
            raise ValueError("risk_item.source_ids 非法或重复")
        source_indexes = [indexes[source_id] for source_id in source_ids]
        if source_indexes != list(range(source_indexes[0], source_indexes[0] + len(source_indexes))):
            raise ValueError("risk_item.source_ids 必须按时间线连续排列")
        if risk_type not in RISK_TYPES:
            raise ValueError("risk_item.risk_type 非法")
        key = (tuple(source_ids), risk_type)
        if key in seen_risks:
            raise ValueError("risk_item 重复")
        risk_items.append(
            {
                "source_ids": source_ids,
                "risk_type": risk_type,
                "confidence": validate_confidence(raw.get("confidence")),
                "reason_codes": validate_reason_codes(raw.get("reason_codes")),
            }
        )
        seen_risks.add(key)
    return {
        "speaker_registry": registry,
        "overrides": overrides,
        "risk_items": risk_items,
        "discarded_risk_items": discarded_risk_items,
    }


def implicit_keep_decision(sentence: Sentence) -> dict[str, Any]:
    return {
        "source_id": sentence.source_id,
        "operation": "KEEP",
        "target_speaker": speaker_label(sentence.spk),
        "parts": [],
        "confidence": 1.0,
        "reason_codes": ["SOURCE_SPK_PRIOR"],
        "review_required": False,
        "source": "implicit_keep",
    }


def fallback_decision(sentence: Sentence, policy: str, unknown_label: str) -> dict[str, Any]:
    if policy == "keep_original":
        decision = implicit_keep_decision(sentence)
        decision["confidence"] = 0.0
    else:
        decision = {
            "source_id": sentence.source_id,
            "operation": "REVIEW_REQUIRED",
            "target_speaker": unknown_label,
            "parts": [],
            "confidence": 0.0,
            "reason_codes": ["AMBIGUOUS_CONTEXT"],
            "review_required": True,
        }
    decision["source"] = "failure_fallback"
    return decision


def is_high_impact(sentence: Sentence) -> bool:
    return has_any_flag(sentence, CONSERVATIVE_KEEP_GUARD_FLAGS)


def preserve_baseline_decision(
    baseline: dict[str, Any],
    sentence: Sentence,
    source: str,
    review_required: bool = False,
) -> dict[str, Any]:
    decision = dict(baseline)
    decision.setdefault("source_id", sentence.source_id)
    decision["parts"] = [dict(part) for part in baseline.get("parts", [])]
    decision["reason_codes"] = list(baseline.get("reason_codes", ["SOURCE_SPK_PRIOR"]))
    decision.setdefault("review_required", False)
    if review_required:
        decision["review_required"] = True
        if "AMBIGUOUS_CONTEXT" not in decision["reason_codes"]:
            decision["reason_codes"].append("AMBIGUOUS_CONTEXT")
    decision["source"] = source
    return decision


def build_full_review_input(
    sentences: list[Sentence],
    allowed_speakers: list[str],
    max_risk_core_sentences: int,
) -> dict[str, Any]:
    return {
        "mode": "full_review",
        "allowed_speakers": allowed_speakers,
        "max_risk_core_sentences": max_risk_core_sentences,
        "source_timeline": [serialize_sentence(sentence) for sentence in sentences],
        "output_schema": {
            "speaker_registry": {
                "speakers": [
                    {
                        "speaker_id": "allowed speaker",
                        "role_summary": "stable anonymous role summary",
                        "evidence_source_ids": ["2 to 8 source IDs"],
                        "confidence": 0.0,
                        "reason_codes": ["limited reason code"],
                    }
                ]
            },
            "overrides": [
                {
                    "source_id": "only a sentence requiring change or conservative marking",
                    "operation": "REASSIGN | REVIEW_REQUIRED | OVERLAP",
                    "target_speaker": "required only for REASSIGN",
                    "confidence": 0.0,
                    "reason_codes": ["limited reason code"],
                }
            ],
            "risk_items": [
                {
                    "source_ids": ["one to max contiguous source IDs needing segment review"],
                    "risk_type": "ROLE_DISCONTINUITY | AMBIGUOUS_TURN | OVERLAP_OR_INTERRUPTION | INTRA_SENTENCE_SWITCH",
                    "confidence": 0.0,
                    "reason_codes": ["limited reason code"],
                }
            ],
        },
    }


def run_full_review(
    sentences: list[Sentence],
    config: dict[str, Any],
    template: str,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_review_prompt(
        template,
        build_full_review_input(
            sentences,
            config["allowed_speakers"],
            config["max_risk_core_sentences"],
        ),
    )
    print(f"正在执行全稿说话人复核，句段数：{len(sentences)}", flush=True)
    result = call_json_with_retries(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_retries=config["max_retries"],
        timeout_seconds=config["request_timeout_s"],
        enable_thinking=config["enable_thinking"],
        validator=lambda payload: validate_full_review_response(
            payload,
            sentences,
            set(config["allowed_speakers"]),
            config["unknown_label"],
            config["overlap_label"],
            config["max_risk_core_sentences"],
        ),
    )
    print("完成全稿说话人复核", flush=True)
    return result, {
        "request_id": "full-review-0001",
        "mode": "full_review",
        "status": "ok",
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "input_source_count": len(sentences),
        "model": model,
    }


def build_full_baseline(
    sentences: list[Sentence],
    overrides: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[int]]:
    baseline = {sentence.source_id: implicit_keep_decision(sentence) for sentence in sentences}
    forced_indexes: set[int] = set()
    for index, sentence in enumerate(sentences):
        override = overrides.get(sentence.source_id)
        if override is None:
            continue
        current = baseline[sentence.source_id]
        low_confidence = override["confidence"] < config["auto_apply_confidence"]
        if override["operation"] == "REASSIGN" and low_confidence:
            baseline[sentence.source_id] = preserve_baseline_decision(
                current,
                sentence,
                "low_confidence_full_override",
                review_required=is_high_impact(sentence),
            )
            forced_indexes.add(index)
        elif override["operation"] == "REVIEW_REQUIRED":
            if low_confidence or not is_high_impact(sentence):
                baseline[sentence.source_id] = preserve_baseline_decision(
                    current,
                    sentence,
                    "full_review_required_baseline",
                    review_required=is_high_impact(sentence),
                )
            else:
                baseline[sentence.source_id] = dict(override)
            forced_indexes.add(index)
        elif override["operation"] == "OVERLAP" and low_confidence:
            baseline[sentence.source_id] = preserve_baseline_decision(
                current,
                sentence,
                "low_confidence_full_overlap",
                review_required=is_high_impact(sentence),
            )
            forced_indexes.add(index)
        else:
            baseline[sentence.source_id] = dict(override)
    return baseline, forced_indexes


def is_fast_gap(left: Sentence, right: Sentence) -> bool:
    return right.start - left.end <= 1000


def is_short_answer(sentence: Sentence) -> bool:
    return "关键短答" in sentence.review_flags or len(sentence.text.strip()) <= 4


def has_any_flag(sentence: Sentence, flags: set[str]) -> bool:
    return bool(set(sentence.review_flags) & flags)


def has_conflicting_stance(left: Sentence, right: Sentence) -> bool:
    left_flags = set(left.review_flags)
    right_flags = set(right.review_flags)
    return (
        {"接受立场", "拒绝立场"} <= (left_flags | right_flags)
        or (("承诺或履约" in left_flags and "否定或限制" in right_flags)
            or ("否定或限制" in left_flags and "承诺或履约" in right_flags))
    )


def has_institution_personal_conflict(left: Sentence, right: Sentence) -> bool:
    left_flags = set(left.review_flags)
    right_flags = set(right.review_flags)
    return (
        ("机构立场" in left_flags and "个人诉求" in right_flags)
        or ("个人诉求" in left_flags and "机构立场" in right_flags)
    ) and (has_any_flag(left, HIGH_IMPACT_FLAGS) or has_any_flag(right, HIGH_IMPACT_FLAGS))


def is_fast_neighbor(sentences: list[Sentence], left_index: int, right_index: int) -> bool:
    return 0 <= left_index < right_index < len(sentences) and is_fast_gap(sentences[left_index], sentences[right_index])


def nearest_tight_same_label_question(sentences: list[Sentence], answer_index: int) -> int | None:
    speaker = sentences[answer_index].spk
    earliest = max(0, answer_index - MAX_SAME_LABEL_QA_LOOKBACK)
    for index in range(answer_index - 1, earliest - 1, -1):
        if sentences[index].spk != speaker or not is_fast_neighbor(sentences, index, index + 1):
            break
        if "疑问表达" in sentences[index].review_flags:
            return index
    return None


def same_label_lead_in_start(sentences: list[Sentence], question_index: int) -> int:
    start = question_index
    for _ in range(SAME_LABEL_QA_LEAD_IN):
        previous = start - 1
        if previous < 0 or sentences[previous].spk != sentences[question_index].spk or not is_fast_neighbor(sentences, previous, start):
            break
        start = previous
    return start


def same_label_follow_through_end(sentences: list[Sentence], answer_index: int) -> int:
    end = answer_index
    for _ in range(SAME_LABEL_QA_FOLLOW_THROUGH):
        following = end + 1
        if following >= len(sentences) or sentences[following].spk != sentences[answer_index].spk or not is_fast_neighbor(sentences, end, following):
            break
        end = following
    return end


def local_label_turbulence_start(sentences: list[Sentence], index: int) -> int | None:
    if not has_any_flag(sentences[index], CONSERVATIVE_KEEP_GUARD_FLAGS):
        return None
    start = index
    transitions = 0
    for previous in range(index - 1, -1, -1):
        if sentences[index].start - sentences[previous].start > MAX_LABEL_TURBULENCE_WINDOW_MS:
            break
        start = previous
        if (
            is_fast_neighbor(sentences, previous, previous + 1)
            and sentences[previous].spk != sentences[previous + 1].spk
        ):
            transitions += 1
    return start if transitions >= MIN_LABEL_TURBULENCE_TRANSITIONS else None


def collect_risk_signals(
    sentences: list[Sentence],
    full_result: dict[str, Any],
    forced_indexes: set[int],
) -> list[RiskSignal]:
    index_by_id = {sentence.source_id: index for index, sentence in enumerate(sentences)}
    signals: list[RiskSignal] = []
    for item in full_result["risk_items"]:
        indexes = tuple(index_by_id[source_id] for source_id in item["source_ids"])
        signals.append(RiskSignal(indexes, item["risk_type"], 60 + round(item["confidence"] * 40), True))
    for index in sorted(forced_indexes):
        signals.append(RiskSignal((index,), "LOW_CONFIDENCE_REASSIGN", 100, True))

    for index, sentence in enumerate(sentences):
        if "时间戳异常" in sentence.review_flags:
            signals.append(RiskSignal((index,), "TIMESTAMP_ANOMALY", 100, True))
        if "快速换人" in sentence.review_flags and len(sentence.text) >= 20 and sentence.end - sentence.start >= 3000:
            signals.append(RiskSignal((index,), "LONG_FAST_TURN", 80, True))
        for flag, score in (
            ("涉及金额", 5),
            ("涉及日期", 5),
            ("劳动争议内容", 5),
            ("关键短答", 20),
            ("疑问表达", 10),
            ("接受立场", 10),
            ("拒绝立场", 10),
            ("否定或限制", 10),
            ("承诺或履约", 10),
            ("机构立场", 5),
            ("个人诉求", 5),
        ):
            if flag in sentence.review_flags:
                signals.append(RiskSignal((index,), flag, score, False))

    for index in range(1, len(sentences)):
        previous, current = sentences[index - 1], sentences[index]
        if current.start - previous.end < 0:
            signals.append(RiskSignal((index - 1, index), "TIMESTAMP_OVERLAP", 100, True))
    for index in range(1, len(sentences) - 1):
        left, middle, right = sentences[index - 1], sentences[index], sentences[index + 1]
        if left.spk == right.spk and left.spk != middle.spk and is_fast_gap(left, middle) and is_fast_gap(middle, right) and is_short_answer(middle):
            signals.append(RiskSignal((index - 1, index, index + 1), "ABA_SHORT_TURN", 80, True))

    structural = tuple(signal for signal in signals if signal.primary)
    for signal in structural:
        start, end = min(signal.source_indexes), max(signal.source_indexes)
        for index in range(max(0, start - 1), min(len(sentences), end + 2)):
            if not has_any_flag(sentences[index], HIGH_IMPACT_FLAGS):
                continue
            if start <= index <= end or (index == start - 1 and is_fast_neighbor(sentences, index, start)) or (index == end + 1 and is_fast_neighbor(sentences, end, index)):
                signals.append(RiskSignal((min(index, start), max(index, end)), "STRUCTURE_HIGH_IMPACT", 35, True))

    for index in range(1, len(sentences)):
        previous, current = sentences[index - 1], sentences[index]
        if previous.spk != current.spk or not is_fast_gap(previous, current):
            continue
        if has_any_flag(current, ANSWER_HIGH_IMPACT_FLAGS):
            question_index = nearest_tight_same_label_question(sentences, index)
            if question_index is not None:
                lead_in_start = same_label_lead_in_start(sentences, question_index)
                follow_through_end = same_label_follow_through_end(sentences, index)
                signals.append(
                    RiskSignal(
                        tuple(range(lead_in_start, follow_through_end + 1)),
                        "SAME_LABEL_HIGH_IMPACT_QA",
                        90,
                        True,
                    )
                )
        if has_conflicting_stance(previous, current):
            signals.append(RiskSignal((index - 1, index), "SAME_LABEL_STANCE_CONFLICT", 85, True))
        if has_institution_personal_conflict(previous, current):
            signals.append(RiskSignal((index - 1, index), "SAME_LABEL_POSITION_CONFLICT", 75, True))

    for index in range(len(sentences)):
        start = local_label_turbulence_start(sentences, index)
        if start is not None:
            signals.append(
                RiskSignal(
                    tuple(range(start, index + 1)),
                    "LOCAL_LABEL_TURBULENCE_HIGH_IMPACT",
                    80,
                    True,
                )
            )
    return signals


def is_tight_bridge(sentences: list[Sentence], left_end: int, right_start: int) -> bool:
    if right_start != left_end + 2:
        return False
    bridge = left_end + 1
    return sentences[bridge].start - sentences[left_end].end <= 1000 and sentences[right_start].start - sentences[bridge].end <= 1000


def ends_with_strong_punctuation(sentence: Sentence) -> bool:
    return bool(sentence.text.rstrip()) and sentence.text.rstrip()[-1] in STRONG_PUNCTUATION


def signal_crosses_cut(signal: RiskSignal, cut_after: int) -> bool:
    return min(signal.source_indexes) <= cut_after < max(signal.source_indexes)


def select_component_cut(
    sentences: list[Sentence],
    primary_signals: list[RiskSignal],
    start: int,
    limit: int,
) -> tuple[int, str, bool]:
    candidates = [
        cut_after
        for cut_after in range(start, limit + 1)
        if not any(signal_crosses_cut(signal, cut_after) for signal in primary_signals)
    ]
    if not candidates:
        return limit, "forced_limit", True

    def ranking(cut_after: int) -> tuple[int, int, int, int]:
        next_index = cut_after + 1
        return (
            sentences[next_index].start - sentences[cut_after].end,
            int(sentences[cut_after].spk != sentences[next_index].spk),
            int(ends_with_strong_punctuation(sentences[cut_after])),
            cut_after,
        )

    return max(candidates, key=ranking), "safe_boundary", False


def partition_risk_component(
    sentences: list[Sentence],
    primary_signals: list[RiskSignal],
    start: int,
    end: int,
    max_core_size: int,
) -> list[tuple[tuple[int, ...], str, bool]]:
    chunks = []
    cursor = start
    while cursor <= end:
        limit = min(end, cursor + max_core_size - 1)
        if limit == end:
            chunks.append((tuple(range(cursor, end + 1)), "component_end", False))
            break
        cut_after, cut_rule, forced_cut = select_component_cut(sentences, primary_signals, cursor, limit)
        chunks.append((tuple(range(cursor, cut_after + 1)), cut_rule, forced_cut))
        cursor = cut_after + 1
    return chunks


def build_risk_segments(
    sentences: list[Sentence],
    signals: list[RiskSignal],
    config: dict[str, Any],
) -> list[RiskSegment]:
    primary = [signal for signal in signals if signal.primary]
    if not primary:
        return []
    intervals = sorted((min(signal.source_indexes), max(signal.source_indexes)) for signal in primary)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue
        previous = merged[-1]
        if start <= previous[1] + 1 or is_tight_bridge(sentences, previous[1], start):
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    segments = []
    for component_index, (start, end) in enumerate(merged, start=1):
        component_id = f"component-{component_index:04d}"
        component_primary = [
            signal
            for signal in primary
            if min(signal.source_indexes) <= end and max(signal.source_indexes) >= start
        ]
        chunks = partition_risk_component(
            sentences,
            component_primary,
            start,
            end,
            config["max_risk_core_sentences"],
        )
        for chunk_index, (core_indexes, cut_rule, forced_cut) in enumerate(chunks, start=1):
            core_start, core_end = core_indexes[0], core_indexes[-1]
            related = [
                signal
                for signal in signals
                if min(signal.source_indexes) <= core_end and max(signal.source_indexes) >= core_start
            ]
            score = sum(signal.score for signal in related)
            signal_types = tuple(dict.fromkeys(signal.signal_type for signal in related))
            context_indexes = tuple(
                index
                for index in range(
                    max(0, core_start - config["context_size"]),
                    min(len(sentences), core_end + config["context_size"] + 1),
                )
                if index not in core_indexes
            )
            segments.append(
                RiskSegment(
                    f"risk-{component_index:04d}-{chunk_index:02d}",
                    component_id,
                    chunk_index,
                    len(chunks),
                    core_indexes,
                    context_indexes,
                    score,
                    signal_types,
                    cut_rule,
                    forced_cut,
                )
            )
    return segments


def speech_unit_count(text: str, end_offset: int) -> int:
    return sum(1 for char in text[:end_offset] if char not in PUNCTUATION)


def estimate_boundary_time(sentence: Sentence, char_offset: int) -> tuple[int | None, str]:
    text_units = speech_unit_count(sentence.text, len(sentence.text))
    unit_offset = speech_unit_count(sentence.text, char_offset)
    if text_units and len(sentence.timestamps) == text_units:
        if unit_offset == 0:
            return sentence.timestamps[0][0], "token_exact"
        if unit_offset == text_units:
            return sentence.timestamps[-1][1], "token_exact"
        left = sentence.timestamps[unit_offset - 1][1]
        right = sentence.timestamps[unit_offset][0]
        return (left + right) // 2, "token_exact"
    if len(sentence.text) and sentence.end >= sentence.start:
        ratio = char_offset / len(sentence.text)
        return round(sentence.start + (sentence.end - sentence.start) * ratio), "interpolated"
    return None, "unavailable"


def build_boundary_candidates(sentence: Sentence, max_candidates: int) -> list[BoundaryCandidate]:
    text = sentence.text
    offsets: dict[int, tuple[int, bool, str]] = {
        0: (100, True, "endpoint"),
        len(text): (100, True, "endpoint"),
    }

    def add(offset: int, priority: int, reliable: bool, candidate_type: str) -> None:
        if not 0 <= offset <= len(text):
            return
        old = offsets.get(offset)
        if old is None or priority > old[0]:
            offsets[offset] = (priority, reliable, candidate_type)

    for index, char in enumerate(text):
        if char in STRONG_PUNCTUATION:
            add(index + 1, 70, True, "strong_punctuation")
        elif char in PUNCTUATION:
            add(index + 1, 40, False, "weak_punctuation")
    for cue in INTERRUPTION_CUES:
        start = 0
        while True:
            position = text.find(cue, start)
            if position < 0:
                break
            add(position, 80, True, "interruption_cue")
            start = position + len(cue)

    unit_count = speech_unit_count(text, len(text))
    if unit_count and len(sentence.timestamps) == unit_count:
        spoken_offsets = [index + 1 for index, char in enumerate(text) if char not in PUNCTUATION]
        for unit_index in range(1, len(sentence.timestamps)):
            gap = sentence.timestamps[unit_index][0] - sentence.timestamps[unit_index - 1][1]
            if gap >= 300:
                add(spoken_offsets[unit_index - 1], 90 + min(gap, 1000) // 100, True, "token_pause")

    endpoint_offsets = {0, len(text)}
    interior = sorted(
        ((offset, value) for offset, value in offsets.items() if offset not in endpoint_offsets),
        key=lambda item: (-item[1][0], item[0]),
    )
    endpoint_values = [(offset, offsets[offset]) for offset in sorted(endpoint_offsets)]
    selected = endpoint_values + interior[:max(0, max_candidates - len(endpoint_values))]
    selected.sort(key=lambda item: item[0])
    candidates = []
    for offset, (_, reliable, candidate_type) in selected:
        estimated_time_ms, time_method = estimate_boundary_time(sentence, offset)
        candidates.append(
            BoundaryCandidate(
                boundary_id=f"{sentence.source_id}.o{offset:03d}",
                source_id=sentence.source_id,
                char_offset=offset,
                estimated_time_ms=estimated_time_ms,
                time_method=time_method,
                reliable=reliable,
                candidate_type=candidate_type,
            )
        )
    return candidates


def split_candidate_source_indexes(segment: RiskSegment, signals: list[RiskSignal]) -> set[int]:
    split_types = {"INTRA_SENTENCE_SWITCH", "OVERLAP_OR_INTERRUPTION", "LONG_FAST_TURN"}
    indexes = set()
    for signal in signals:
        if signal.signal_type in split_types:
            indexes.update(index for index in signal.source_indexes if index in segment.core_indexes)
    return indexes


def build_segment_candidates(
    sentences: list[Sentence],
    segment: RiskSegment,
    signals: list[RiskSignal],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, BoundaryCandidate]], set[str], dict[str, str]]:
    split_indexes = split_candidate_source_indexes(segment, signals)
    candidate_map: dict[str, dict[str, BoundaryCandidate]] = {}
    split_source_ids = set()
    split_eligibility: dict[str, str] = {}
    for index in segment.core_indexes:
        sentence = sentences[index]
        if index not in split_indexes:
            candidates = build_boundary_candidates(sentence, 2)
            split_eligibility[sentence.source_id] = "no_split_signal"
        else:
            candidates = build_boundary_candidates(sentence, config["max_boundary_candidates"])
            has_reliable_internal = any(
                candidate.reliable and 0 < candidate.char_offset < len(sentence.text)
                for candidate in candidates
            )
            if has_reliable_internal:
                split_source_ids.add(sentence.source_id)
                split_eligibility[sentence.source_id] = "eligible"
            else:
                split_eligibility[sentence.source_id] = "no_reliable_internal_boundary"
        candidate_map[sentence.source_id] = {candidate.boundary_id: candidate for candidate in candidates}
    return candidate_map, split_source_ids, split_eligibility


def validate_decision_response(
    payload: dict[str, Any],
    core_sentences: list[Sentence],
    candidate_map: dict[str, dict[str, BoundaryCandidate]],
    allowed_speakers: set[str],
    unknown_label: str,
    overlap_label: str,
) -> dict[str, dict[str, Any]]:
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("审核响应必须包含 decisions 数组")
    expected_ids = {sentence.source_id for sentence in core_sentences}
    source_by_id = {sentence.source_id: sentence for sentence in core_sentences}
    all_labels = allowed_speakers | {unknown_label, overlap_label}
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("decision 项必须是对象")
        source_id = raw.get("source_id")
        operation = raw.get("operation")
        if source_id not in expected_ids or source_id in result:
            raise ValueError("decision source_id 非法、非 core 或重复")
        if operation not in OPERATIONS:
            raise ValueError("operation 非法")
        confidence = validate_confidence(raw.get("confidence"))
        reason_codes = validate_reason_codes(raw.get("reason_codes"))
        sentence = source_by_id[source_id]
        target_speaker = raw.get("target_speaker")
        parts: list[dict[str, Any]] = []
        if operation == "KEEP":
            if target_speaker not in (None, speaker_label(sentence.spk)):
                raise ValueError("KEEP 不得改变说话人")
            target_speaker = speaker_label(sentence.spk)
        elif operation == "REASSIGN":
            if target_speaker not in allowed_speakers:
                raise ValueError("REASSIGN target_speaker 非法")
        elif operation == "REVIEW_REQUIRED":
            if target_speaker not in (None, unknown_label):
                raise ValueError("REVIEW_REQUIRED 只能使用 unknown")
            target_speaker = unknown_label
        elif operation == "OVERLAP":
            if target_speaker not in (None, overlap_label):
                raise ValueError("OVERLAP 只能使用 overlap")
            target_speaker = overlap_label
        else:
            raw_parts = raw.get("parts")
            if not isinstance(raw_parts, list) or len(raw_parts) < 2:
                raise ValueError("SPLIT 必须包含至少两个 parts")
            candidates = candidate_map[source_id]
            previous_end = 0
            for part in raw_parts:
                if not isinstance(part, dict):
                    raise ValueError("SPLIT part 必须是对象")
                start_id = part.get("start_boundary_id")
                end_id = part.get("end_boundary_id")
                speaker = part.get("speaker")
                if start_id not in candidates or end_id not in candidates or speaker not in all_labels:
                    raise ValueError("SPLIT 边界或 speaker 非法")
                start_offset = candidates[start_id].char_offset
                end_offset = candidates[end_id].char_offset
                if start_offset != previous_end or end_offset <= start_offset:
                    raise ValueError("SPLIT parts 必须连续、递增且非空")
                parts.append(
                    {
                        "start_boundary_id": start_id,
                        "end_boundary_id": end_id,
                        "speaker": speaker,
                        "char_start": start_offset,
                        "char_end": end_offset,
                    }
                )
                previous_end = end_offset
            if previous_end != len(sentence.text):
                raise ValueError("SPLIT parts 必须覆盖完整源句")
            target_speaker = None
        result[source_id] = {
            "source_id": source_id,
            "operation": operation,
            "target_speaker": target_speaker,
            "parts": parts,
            "confidence": confidence,
            "reason_codes": reason_codes,
            "review_required": operation in {"REVIEW_REQUIRED", "OVERLAP"},
            "source": "risk_segment",
        }
    if set(result) != expected_ids:
        missing = sorted(expected_ids - set(result))
        raise ValueError(f"decision 必须完整覆盖 core：缺失 {missing}")
    return result


def validate_risk_segment_response(
    payload: dict[str, Any],
    core_sentences: list[Sentence],
    candidate_map: dict[str, dict[str, BoundaryCandidate]],
    split_source_ids: set[str],
    allowed_speakers: set[str],
    unknown_label: str,
    overlap_label: str,
) -> dict[str, dict[str, Any]]:
    decisions = validate_decision_response(
        payload,
        core_sentences,
        candidate_map,
        allowed_speakers,
        unknown_label,
        overlap_label,
    )
    for source_id, decision in decisions.items():
        if decision["operation"] != "SPLIT":
            continue
        if source_id not in split_source_ids:
            raise ValueError("该 core 句未提供可拆分的可靠边界")
        candidates = candidate_map[source_id]
        for part in decision["parts"]:
            for boundary_id in (part["start_boundary_id"], part["end_boundary_id"]):
                candidate = candidates[boundary_id]
                if candidate.char_offset not in {0, len(next(sentence for sentence in core_sentences if sentence.source_id == source_id).text)} and not candidate.reliable:
                    raise ValueError("SPLIT 内部边界必须可靠")
    return decisions


def conservative_keep_guard_signals(
    sentence: Sentence,
    sentence_index: int,
    core_sentences: list[Sentence],
    core_indexes: tuple[int, ...],
    raw_decisions: dict[str, dict[str, Any]],
    signals: list[RiskSignal],
) -> list[str]:
    signal_types = []
    for signal in signals:
        if (
            signal.signal_type not in CONSERVATIVE_KEEP_GUARD_SIGNAL_TYPES
            or sentence_index not in signal.source_indexes
        ):
            continue
        members = [
            member
            for member, member_index in zip(core_sentences, core_indexes)
            if member_index in signal.source_indexes
        ]
        has_high_impact_member = any(
            has_any_flag(member, CONSERVATIVE_KEEP_GUARD_FLAGS)
            for member in members
        )
        has_ambiguous_same_speaker_member = any(
            member.source_id != sentence.source_id
            and member.spk == sentence.spk
            and raw_decisions[member.source_id]["operation"] in {"REVIEW_REQUIRED", "OVERLAP"}
            for member in members
        )
        if has_any_flag(sentence, CONSERVATIVE_KEEP_GUARD_FLAGS) or (
            has_high_impact_member and has_ambiguous_same_speaker_member
        ):
            signal_types.append(signal.signal_type)
    return list(dict.fromkeys(signal_types))


def normalize_segment_decision(
    decision: dict[str, Any],
    baseline: dict[str, Any],
    sentence: Sentence,
    config: dict[str, Any],
    keep_guard_signals: list[str],
) -> dict[str, Any]:
    operation = decision["operation"]
    low_confidence = decision["confidence"] < config["auto_apply_confidence"]
    if operation == "KEEP":
        return preserve_baseline_decision(
            baseline,
            sentence,
            "risk_segment",
            review_required=low_confidence and is_high_impact(sentence),
        )
    if operation in {"REASSIGN", "SPLIT"} and low_confidence:
        return preserve_baseline_decision(
            baseline,
            sentence,
            "low_confidence_segment",
            review_required=is_high_impact(sentence),
        )
    if operation == "REVIEW_REQUIRED" and (low_confidence or not is_high_impact(sentence)):
        return preserve_baseline_decision(
            baseline,
            sentence,
            "risk_review_required_baseline",
            review_required=is_high_impact(sentence),
        )
    if operation == "OVERLAP" and low_confidence:
        return preserve_baseline_decision(
            baseline,
            sentence,
            "low_confidence_overlap",
            review_required=is_high_impact(sentence),
        )
    return decision


def apply_unreviewed_high_impact_keep_guard(
    sentences: list[Sentence],
    decisions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": sentence.source_id,
            "review_flags": [
                flag for flag in sentence.review_flags
                if flag in CONSERVATIVE_KEEP_GUARD_FLAGS
            ],
        }
        for sentence in sentences
        if (
            decisions[sentence.source_id]["operation"] == "KEEP"
            and decisions[sentence.source_id]["source"] == "implicit_keep"
            and has_any_flag(sentence, CONSERVATIVE_KEEP_GUARD_FLAGS)
        )
    ]


def build_risk_segment_input(
    segment: RiskSegment,
    sentences: list[Sentence],
    registry: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, BoundaryCandidate]],
    split_source_ids: set[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    core = []
    for index in segment.core_indexes:
        sentence = sentences[index]
        source_id = sentence.source_id
        value = serialize_sentence(
            sentence,
            list(candidates[source_id].values()) if source_id in split_source_ids else None,
            include_flags=True,
        )
        value["allowed_operations"] = (
            ["KEEP", "REASSIGN", "SPLIT", "REVIEW_REQUIRED", "OVERLAP"]
            if source_id in split_source_ids
            else ["KEEP", "REASSIGN", "REVIEW_REQUIRED", "OVERLAP"]
        )
        value["baseline"] = {
            "operation": baseline[source_id]["operation"],
            "target_speaker": baseline[source_id]["target_speaker"],
            "confidence": baseline[source_id]["confidence"],
        }
        core.append(value)
    return {
        "mode": "risk_segment",
        "allowed_speakers": config["allowed_speakers"],
        "unknown_label": config["unknown_label"],
        "overlap_label": config["overlap_label"],
        "speaker_registry": registry,
        "risk_signals": list(segment.signals),
        "context": [serialize_sentence(sentences[index]) for index in segment.context_indexes],
        "core": core,
        "output_schema": {
            "decisions": [
                {
                    "source_id": "core source_id",
                    "operation": "only one of the sentence allowed_operations",
                    "target_speaker": "required only for REASSIGN",
                    "parts": [
                        {
                            "start_boundary_id": "required only for SPLIT",
                            "end_boundary_id": "required only for SPLIT",
                            "speaker": "allowed speaker, unknown_label or overlap_label",
                        }
                    ],
                    "confidence": 0.0,
                    "reason_codes": ["limited reason code"],
                }
            ]
        },
    }


def run_risk_segment_pass(
    segments: list[RiskSegment],
    sentences: list[Sentence],
    signals: list[RiskSignal],
    registry: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    config: dict[str, Any],
    template: str,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    raw_results: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    executable = list(segments)

    def execute(segment: RiskSegment) -> tuple[
        RiskSegment,
        dict[str, dict[str, Any]] | Exception,
        str,
        int,
        dict[str, dict[str, BoundaryCandidate]],
        set[str],
        dict[str, str],
    ]:
        core_sentences = [sentences[index] for index in segment.core_indexes]
        candidates, split_source_ids, split_eligibility = build_segment_candidates(sentences, segment, signals, config)
        prompt = build_review_prompt(
            template,
            build_risk_segment_input(segment, sentences, registry, baseline, candidates, split_source_ids, config),
        )
        prompt_bytes = len(prompt.encode("utf-8"))
        try:
            print(f"正在复核高风险片段 {segment.segment_id}，核心句段数：{len(core_sentences)}", flush=True)
            value = call_json_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_retries=config["max_retries"],
                timeout_seconds=config["request_timeout_s"],
                enable_thinking=config["enable_thinking"],
                validator=lambda payload: validate_risk_segment_response(
                    payload,
                    core_sentences,
                    candidates,
                    split_source_ids,
                    set(config["allowed_speakers"]),
                    config["unknown_label"],
                    config["overlap_label"],
                ),
            )
            print(f"完成高风险片段 {segment.segment_id}", flush=True)
            return segment, value, sha256_bytes(prompt.encode("utf-8")), prompt_bytes, candidates, split_source_ids, split_eligibility
        except Exception as error:
            return (
                segment,
                error,
                sha256_bytes(prompt.encode("utf-8")),
                prompt_bytes,
                candidates,
                split_source_ids,
                split_eligibility,
            )

    if executable:
        print(f"需要复核的高风险连续片段数：{len(executable)}", flush=True)
    workers = min(config["max_workers"], len(executable))
    if workers <= 1:
        completed = [execute(segment) for segment in executable]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            completed = list(executor.map(execute, executable))
    for segment, value, prompt_sha256, prompt_bytes, candidates, split_source_ids, split_eligibility in completed:
        core_sentences = [sentences[index] for index in segment.core_indexes]
        audit = {
            "request_id": segment.segment_id,
            "mode": "risk_segment",
            "component_id": segment.component_id,
            "chunk_index": segment.chunk_index,
            "chunk_count": segment.chunk_count,
            "cut_rule": segment.cut_rule,
            "forced_cut": segment.forced_cut,
            "core_source_ids": [sentence.source_id for sentence in core_sentences],
            "context_source_ids": [sentences[index].source_id for index in segment.context_indexes],
            "risk_score": segment.score,
            "risk_signals": list(segment.signals),
            "boundary_candidate_count": sum(len(value) for value in candidates.values()),
            "split_candidate_source_ids": sorted(split_source_ids),
            "split_eligibility": split_eligibility,
            "prompt_sha256": prompt_sha256,
            "prompt_bytes": prompt_bytes,
            "model": model,
        }
        if isinstance(value, Exception):
            fallback = {
                sentence.source_id: preserve_baseline_decision(
                    baseline[sentence.source_id],
                    sentence,
                    "risk_segment_failure_baseline",
                    review_required=is_high_impact(sentence),
                )
                for sentence in core_sentences
            }
            results.update(fallback)
            audit.update({"status": "failed", "error": str(value), "fallback": fallback})
        else:
            raw_results.update(value)
            keep_guards = {
                sentence.source_id: conservative_keep_guard_signals(
                    sentence,
                    index,
                    core_sentences,
                    segment.core_indexes,
                    value,
                    signals,
                )
                for sentence, index in zip(core_sentences, segment.core_indexes)
            }
            normalized = {
                sentence.source_id: normalize_segment_decision(
                    value[sentence.source_id],
                    baseline[sentence.source_id],
                    sentence,
                    config,
                    keep_guards[sentence.source_id],
                )
                for sentence in core_sentences
            }
            results.update(normalized)
            audit.update({
                "status": "ok",
                "raw_decisions": value,
                "normalized_decisions": normalized,
                "conservative_keep_guards": [
                    {
                        "source_id": sentence.source_id,
                        "signal_types": keep_guards[sentence.source_id],
                        "review_flags": [
                            flag for flag in sentence.review_flags
                            if flag in CONSERVATIVE_KEEP_GUARD_FLAGS
                        ],
                    }
                    for sentence in core_sentences
                    if keep_guards[sentence.source_id]
                ],
            })
        audits.append(audit)
    return results, sorted(audits, key=lambda item: item["request_id"]), raw_results


def estimate_span_interval(sentence: Sentence, char_start: int, char_end: int) -> tuple[int, int, str]:
    if char_start == 0 and char_end == len(sentence.text):
        return sentence.start, sentence.end, "source_range"
    start, start_method = estimate_boundary_time(sentence, char_start)
    end, end_method = estimate_boundary_time(sentence, char_end)
    if start is None or end is None:
        return sentence.start, sentence.end, "source_range"
    return min(start, end), max(start, end), start_method if start_method == end_method else "mixed"


def decision_to_spans(sentence: Sentence, decision: dict[str, Any], candidates: dict[str, BoundaryCandidate] | None = None) -> list[ReviewedSpan]:
    if decision["operation"] == "SPLIT":
        parts = decision["parts"]
    else:
        parts = [{"char_start": 0, "char_end": len(sentence.text), "speaker": decision["target_speaker"]}]
    spans = []
    for part in parts:
        char_start = part["char_start"]
        char_end = part["char_end"]
        text = sentence.text[char_start:char_end]
        if not text:
            raise RuntimeError(f"复核片段为空：{sentence.source_id}")
        start, end, time_method = estimate_span_interval(sentence, char_start, char_end)
        spans.append(
            ReviewedSpan(
                source_id=sentence.source_id,
                source_order=sentence.index,
                char_start=char_start,
                char_end=char_end,
                start=start,
                end=end,
                spk=part["speaker"],
                original_spk=sentence.spk,
                text=text,
                operation=decision["operation"],
                confidence=decision["confidence"],
                reason_codes=list(decision["reason_codes"]),
                review_required=decision["review_required"],
                review_flags=list(sentence.review_flags),
                time_method=time_method,
            )
        )
    return spans


def validate_spans(sentences: list[Sentence], spans: list[ReviewedSpan], allowed_labels: set[str]) -> dict[str, bool]:
    spans_by_source: dict[str, list[ReviewedSpan]] = {}
    for span in spans:
        spans_by_source.setdefault(span.source_id, []).append(span)
        if span.spk not in allowed_labels:
            raise RuntimeError(f"reviewed span 使用了非法 speaker：{span.spk}")
    source_ids = [sentence.source_id for sentence in sentences]
    if set(spans_by_source) != set(source_ids):
        raise RuntimeError("reviewed spans 未完整覆盖所有源句")
    for sentence in sentences:
        source_spans = sorted(spans_by_source[sentence.source_id], key=lambda item: item.char_start)
        if source_spans[0].char_start != 0 or source_spans[-1].char_end != len(sentence.text):
            raise RuntimeError(f"源句覆盖不完整：{sentence.source_id}")
        previous_end = 0
        for span in source_spans:
            if span.char_start != previous_end or span.char_end <= span.char_start:
                raise RuntimeError(f"源句片段边界非法：{sentence.source_id}")
            if span.text != sentence.text[span.char_start:span.char_end]:
                raise RuntimeError(f"源句片段文本不匹配：{sentence.source_id}")
            previous_end = span.char_end
        if "".join(span.text for span in source_spans) != sentence.text:
            raise RuntimeError(f"源句重建失败：{sentence.source_id}")
    ordered = sorted(spans, key=lambda item: (item.source_order, item.char_start))
    if ordered != spans:
        raise RuntimeError("reviewed spans 顺序不符合源句顺序")
    if "".join(span.text for span in spans) != "".join(sentence.text for sentence in sentences):
        raise RuntimeError("全局文本重建失败")
    return {
        "per_source_reconstruction_passed": True,
        "global_reconstruction_passed": True,
        "order_passed": True,
        "coverage_passed": True,
        "allowed_speaker_passed": True,
    }


def render_reviewed_transcript(spans: list[ReviewedSpan], speaker_prefix: str, keep_time: bool) -> str:
    rendered = []
    for span in spans:
        speaker = normalize_speaker(span.spk, speaker_prefix)
        review_marker = "【待回听】" if span.review_required else ""
        header = f"[{format_time(span.start)} - {format_time(span.end)}] {speaker}{review_marker}：" if keep_time else f"{speaker}{review_marker}："
        rendered.append(f"{header}\n{span.text}")
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def serialize_span(span: ReviewedSpan) -> dict[str, Any]:
    return asdict(span)


def component_partition_passed(segments: list[RiskSegment], max_core_size: int) -> bool:
    by_component: dict[str, list[RiskSegment]] = {}
    for segment in segments:
        by_component.setdefault(segment.component_id, []).append(segment)
    for component_segments in by_component.values():
        ordered = sorted(component_segments, key=lambda segment: segment.chunk_index)
        if [segment.chunk_index for segment in ordered] != list(range(1, len(ordered) + 1)):
            return False
        if any(segment.chunk_count != len(ordered) or len(segment.core_indexes) > max_core_size for segment in ordered):
            return False
        covered = [index for segment in ordered for index in segment.core_indexes]
        if not covered or covered != list(range(covered[0], covered[-1] + 1)):
            return False
    return True


def run_speaker_review(
    *,
    json_path: Path,
    sentences: list[Sentence],
    prompt_dir: Path,
    config: dict[str, Any],
    base_url: str,
    api_key: str,
    default_model: str,
    speaker_prefix: str,
    keep_time: bool,
) -> SpeakerReviewResult:
    if not sentences:
        raise ValueError("没有可供说话人复核的源句")
    normalized_config = normalize_review_config(config, sentences)
    if not normalized_config["enabled"]:
        raise ValueError("speaker_review.enabled=false 时不应调用 run_speaker_review")
    if not isinstance(default_model, str) or not default_model:
        raise ValueError("必须通过 .env 的 MODEL_NAME 提供模型")
    template = load_review_prompt_template(prompt_dir)
    source_json_sha256 = sha256_file(json_path)

    full_error = None
    try:
        full_result, full_audit = run_full_review(
            sentences, normalized_config, template, base_url, api_key, default_model
        )
        baseline, forced_indexes = build_full_baseline(sentences, full_result["overrides"], normalized_config)
        signals = collect_risk_signals(sentences, full_result, forced_indexes)
        segments = build_risk_segments(sentences, signals, normalized_config)
        segment_decisions, segment_audits, raw_segment_decisions = run_risk_segment_pass(
            segments,
            sentences,
            signals,
            full_result["speaker_registry"],
            baseline,
            normalized_config,
            template,
            base_url,
            api_key,
            default_model,
        )
    except Exception as error:
        full_error = error
        if normalized_config["failure_policy"] == "fail_closed":
            raise
        full_result = {
            "speaker_registry": {"speakers": []},
            "overrides": {},
            "risk_items": [],
            "discarded_risk_items": [],
        }
        baseline = {
            sentence.source_id: fallback_decision(sentence, normalized_config["failure_policy"], normalized_config["unknown_label"])
            for sentence in sentences
        }
        signals = []
        segments = []
        segment_decisions = {}
        raw_segment_decisions = {}
        full_audit = {
            "request_id": "full-review-0001",
            "mode": "full_review",
            "status": "failed",
            "error": str(error),
            "model": default_model,
        }
        segment_audits = []

    final_decisions = dict(baseline)
    final_decisions.update(segment_decisions)
    unreviewed_high_impact_guards = apply_unreviewed_high_impact_keep_guard(
        sentences,
        final_decisions,
        normalized_config,
    )
    if sha256_file(json_path) != source_json_sha256:
        raise RuntimeError("原始 FunASR JSON 在说话人复核期间发生变化")

    spans = []
    for sentence in sentences:
        spans.extend(decision_to_spans(sentence, final_decisions[sentence.source_id]))
    allowed_labels = set(normalized_config["allowed_speakers"]) | {normalized_config["unknown_label"], normalized_config["overlap_label"]}
    integrity = validate_spans(sentences, spans, allowed_labels)
    segment_by_index = {index: segment.segment_id for segment in segments for index in segment.core_indexes}
    risk_by_index = {
        index: {
            "score": sum(signal.score for signal in signals if index in signal.source_indexes),
            "signals": [signal.signal_type for signal in signals if index in signal.source_indexes],
            "segment_id": segment_by_index.get(index),
        }
        for index in range(len(sentences))
    }
    review_queue = [
        {
            "source_id": span.source_id,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "final_speaker": span.spk,
            "operation": span.operation,
            "reason_codes": span.reason_codes,
        }
        for span in spans
        if span.review_required or span.spk in {normalized_config["unknown_label"], normalized_config["overlap_label"]}
    ]
    counts = {operation: sum(1 for decision in final_decisions.values() if decision["operation"] == operation) for operation in sorted(OPERATIONS)}
    unknown_source_counts: dict[str, int] = {}
    for decision in final_decisions.values():
        if decision["target_speaker"] == normalized_config["unknown_label"]:
            source = decision["source"]
            unknown_source_counts[source] = unknown_source_counts.get(source, 0) + 1
    baseline_preserved_count = sum(
        final_decisions[sentence.source_id]["target_speaker"] == baseline[sentence.source_id]["target_speaker"]
        and final_decisions[sentence.source_id]["operation"] != "SPLIT"
        for sentence in sentences
    )
    total_prompt_bytes = full_audit.get("prompt_bytes", 0) + sum(item.get("prompt_bytes", 0) for item in segment_audits)
    total_boundaries = sum(item.get("boundary_candidate_count", 0) for item in segment_audits)
    semantic_flag_counts = {
        flag: sum(flag in sentence.review_flags for sentence in sentences)
        for flag in sorted(SEMANTIC_FLAGS)
    }
    promoted_signals = [
        {
            "signal_type": signal.signal_type,
            "source_ids": [sentences[index].source_id for index in signal.source_indexes],
        }
        for signal in signals
        if signal.signal_type in SEMANTIC_PROMOTION_TYPES
    ]
    partition_passed = component_partition_passed(segments, normalized_config["max_risk_core_sentences"])
    audit = {
        "schema_version": 3,
        "run": {
            "source_json_path": str(json_path),
            "source_json_sha256": source_json_sha256,
            "prompt_template_sha256": sha256_bytes(template.encode("utf-8")),
            "config_sha256": stable_json_hash(normalized_config),
            "model": default_model,
            "allowed_speakers": normalized_config["allowed_speakers"],
            "unknown_label": normalized_config["unknown_label"],
            "overlap_label": normalized_config["overlap_label"],
            "failure_policy": normalized_config["failure_policy"],
            "model_from_env_verified": True,
        },
        "full_review": {
            **full_audit,
            "overrides": list(full_result["overrides"].values()),
            "risk_items": full_result["risk_items"],
            "discarded_risk_items": full_result["discarded_risk_items"],
            "failed": full_error is not None,
        },
        "speaker_registry": full_result["speaker_registry"],
        "risk_coverage": {
            "semantic_flag_counts": semantic_flag_counts,
            "promoted_signals": promoted_signals,
            "unreviewed_high_impact_items": unreviewed_high_impact_guards,
        },
        "risk_segments": segment_audits,
        "decisions": [
            {
                "source_id": sentence.source_id,
                "original_spk": sentence.spk,
                "baseline": baseline[sentence.source_id],
                "risk": risk_by_index[index],
                "segment_override": segment_decisions.get(sentence.source_id),
                "raw_segment_override": raw_segment_decisions.get(sentence.source_id),
                "final": final_decisions[sentence.source_id],
            }
            for index, sentence in enumerate(sentences)
        ],
        "reviewed_spans": [serialize_span(span) for span in spans],
        "review_queue": review_queue,
        "integrity": {
            "source_sentence_count": len(sentences),
            "reviewed_source_count": len({span.source_id for span in spans}),
            "reviewed_span_count": len(spans),
            "operation_counts": counts,
            "baseline_preserved_count": baseline_preserved_count,
            "high_confidence_reassign_count": sum(
                decision["operation"] == "REASSIGN"
                and decision["confidence"] >= normalized_config["auto_apply_confidence"]
                for decision in final_decisions.values()
            ),
            "low_confidence_suggestion_ignored_count": sum(
                decision["source"] in {
                    "low_confidence_full_override",
                    "low_confidence_full_overlap",
                    "low_confidence_segment",
                    "low_confidence_overlap",
                }
                for decision in final_decisions.values()
            ),
            "unknown_count": sum(1 for span in spans if span.spk == normalized_config["unknown_label"]),
            "explicit_unknown_count": sum(unknown_source_counts.values()),
            "unknown_source_counts": unknown_source_counts,
            "unknown_from_explicit_review_passed": set(unknown_source_counts) <= {"full_review", "risk_segment"},
            "overlap_count": sum(1 for span in spans if span.spk == normalized_config["overlap_label"]),
            "review_queue_count": len(review_queue),
            "segment_failure_baseline_fallback_count": sum(
                decision["source"] == "risk_segment_failure_baseline"
                for decision in final_decisions.values()
            ),
            "source_hash_verified": True,
            "full_review_request_count": 1,
            "risk_segment_request_count": len(segment_audits),
            "total_request_count": 1 + len(segment_audits),
            "total_prompt_bytes": total_prompt_bytes,
            "boundary_candidate_count": total_boundaries,
            "unique_core_membership_passed": len(segment_by_index) == sum(len(segment.core_indexes) for segment in segments),
            "component_partition_passed": partition_passed,
            "max_core_size_passed": all(
                len(segment.core_indexes) <= normalized_config["max_risk_core_sentences"]
                for segment in segments
            ),
            "context_core_separation_passed": all(not set(segment.core_indexes) & set(segment.context_indexes) for segment in segments),
            "sparse_full_output_passed": True,
            **integrity,
        },
    }
    return SpeakerReviewResult(spans=spans, audit=audit, reviewed_text=render_reviewed_transcript(spans, speaker_prefix, keep_time))


def write_speaker_review_outputs(result: SpeakerReviewResult, review_json_path: Path, reviewed_path: Path) -> None:
    review_json_path.write_text(json.dumps(result.audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviewed_path.write_text(result.reviewed_text, encoding="utf-8")
