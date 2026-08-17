#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sentence:
    index: int
    source_id: str
    result_index: int
    sentence_index: int
    start: int
    end: int
    spk: int | str
    text: str
    timestamps: list[list[int]]
    review_flags: list[str]


@dataclass
class ReviewedSpan:
    source_id: str
    source_order: int
    char_start: int
    char_end: int
    start: int
    end: int
    spk: int | str
    original_spk: int | str
    text: str
    operation: str
    confidence: float
    reason_codes: list[str]
    review_required: bool
    review_flags: list[str]
    time_method: str


@dataclass
class Block:
    start: int
    end: int
    spk: int | str
    texts: list[str]
    review_flags: list[str]

    @property
    def text(self) -> str:
        return join_texts(self.texts)


@dataclass
class TextCleaningConfig:
    repeated_words: list[str]
    drop_words: set[str]
    filler_words: list[str]


@dataclass
class PolishBlock:
    header: str
    text: str


@dataclass
class ReadingUnit:
    unit_id: str
    start: int
    end: int
    spk: int | str
    texts: list[str]
    source_spans: list[dict[str, Any]]
    review_flags: list[str]
    review_required: bool

    @property
    def text(self) -> str:
        return join_texts(self.texts)


@dataclass(frozen=True)
class ProtectedAtom:
    kind: str
    value: str
    start: int
    end: int


class FinalValidationError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


SHORT_ANSWER_WORDS = {"对", "好", "是", "不是", "可以", "不可以", "同意", "不同意"}
LABOR_KEYWORDS = (
    "工资", "薪资", "补偿", "赔偿", "解除", "辞退", "离职", "仲裁",
    "社保", "公积金", "年假", "年休假", "劳动合同", "N+1", "N加一",
)
ACCEPTANCE_PHRASES = ("接受", "同意", "认可", "答应", "愿意", "可以", "没问题")
REJECTION_PHRASES = ("不同意", "不接受", "不认可", "不答应", "拒绝", "不可以", "不能", "不会", "无法", "不予", "没有那么多", "没那么多")
COMMITMENT_PHRASES = ("承诺", "保证", "会支付", "将支付", "会补", "将补", "安排", "处理", "答复", "落实", "补发", "补缴")
INSTITUTION_POSITION_PHRASES = ("公司", "单位", "我们这边", "本单位", "制度", "政策", "流程", "审批", "领导", "人事")
PERSONAL_POSITION_PHRASES = ("我的诉求", "我要求", "我希望", "我接受", "我同意", "我不同意", "我不接受", "我不要求", "我现在只要", "我认为", "我需要", "我要")
QUESTION_CUE_PATTERN = re.compile(r"[？?]|(?:是否|能否|可不可以|可以吗|多少|怎么算|怎么(?:样|办)?|何时|什么时候|为什么|凭什么)")
MONEY_PATTERN = re.compile(
    r"(?:[￥¥]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:元|块|万(?:元)?)|"
    r"[零〇一二两三四五六七八九十百千万]+(?:元|块|万(?:元)?)|"
    r"(?:\d{3,}|[零〇一二两三四五六七八九十百千万幺]{3,})\D{0,4}(?:金额|数额|基数|工资|薪资|补偿(?:金)?|赔偿(?:金)?)|"
    r"(?:工资|薪资|补偿(?:金)?|赔偿(?:金)?|基数)\D{0,8}(?:(?:\d{2,})|(?:[零〇一二两三四五六七八九幺]*(?:十|百|千|万)[零〇一二两三四五六七八九十百千万幺]*)))"
)
DATE_PATTERN = re.compile(
    r"(?:\d{2,4}年\d{1,2}月(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
)


def load_word_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"词表文件不存在：{path}")

    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    return words


def load_cleaning_config(prompt_dir: Path) -> TextCleaningConfig:
    return TextCleaningConfig(
        repeated_words=load_word_list(prompt_dir / "repeated_words.txt"),
        drop_words=set(load_word_list(prompt_dir / "drop_words.txt")),
        filler_words=load_word_list(prompt_dir / "filler_words.txt"),
    )


def load_prompt_template(prompt_dir: Path) -> str:
    template_path = prompt_dir / "polish_prompt_template.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"润色提示词模板不存在：{template_path}")
    template = template_path.read_text(encoding="utf-8")
    if "{{ chunk_text }}" not in template:
        raise ValueError(f"润色提示词模板必须包含占位符 {{{{ chunk_text }}}}：{template_path}")
    return template


def normalize_timestamps(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []

    timestamps = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            timestamps.append([int(float(item[0])), int(float(item[1]))])
        except (TypeError, ValueError):
            continue
    return timestamps


def iter_sentence_items(data: Any) -> list[tuple[int, int, dict[str, Any]]]:
    if isinstance(data, dict):
        results = [data]
    elif isinstance(data, list):
        results = data
    else:
        return []

    items = []
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        sentence_info = result.get("sentence_info") or result.get("segments") or []
        if not isinstance(sentence_info, list):
            continue
        for sentence_index, item in enumerate(sentence_info):
            if isinstance(item, dict):
                items.append((result_index, sentence_index, item))
    return items


def load_sentences(json_path: Path) -> list[Sentence]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sentences = []

    for index, (result_index, sentence_index, item) in enumerate(iter_sentence_items(data), start=1):
        text_value = item.get("text", item.get("sentence", ""))
        text = "" if text_value is None else str(text_value)
        if text == "":
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        spk = item.get("spk", item.get("speaker", "unknown"))
        sentences.append(
            Sentence(
                index=index,
                source_id=f"r{result_index:03d}.s{sentence_index:06d}",
                result_index=result_index,
                sentence_index=sentence_index,
                start=int(float(start)),
                end=int(float(end)),
                spk=spk,
                text=text,
                timestamps=normalize_timestamps(item.get("timestamp")),
                review_flags=[],
            )
        )

    return sentences


def add_review_flag(sentence: Sentence, flag: str) -> None:
    if flag not in sentence.review_flags:
        sentence.review_flags.append(flag)


def review_sentences(sentences: list[Sentence]) -> None:
    for sentence in sentences:
        normalized_text = re.sub(r"[\s，,。！？；：、]+", "", sentence.text)
        if normalized_text in SHORT_ANSWER_WORDS:
            add_review_flag(sentence, "关键短答")
        if MONEY_PATTERN.search(sentence.text):
            add_review_flag(sentence, "涉及金额")
        if DATE_PATTERN.search(sentence.text):
            add_review_flag(sentence, "涉及日期")
        if any(keyword in sentence.text for keyword in LABOR_KEYWORDS):
            add_review_flag(sentence, "劳动争议内容")
        if QUESTION_CUE_PATTERN.search(sentence.text):
            add_review_flag(sentence, "疑问表达")
        if any(phrase in sentence.text for phrase in REJECTION_PHRASES):
            add_review_flag(sentence, "拒绝立场")
            add_review_flag(sentence, "否定或限制")
        elif any(phrase in sentence.text for phrase in ACCEPTANCE_PHRASES):
            add_review_flag(sentence, "接受立场")
        if any(phrase in sentence.text for phrase in COMMITMENT_PHRASES):
            add_review_flag(sentence, "承诺或履约")
        if any(phrase in sentence.text for phrase in INSTITUTION_POSITION_PHRASES):
            add_review_flag(sentence, "机构立场")
        if any(phrase in sentence.text for phrase in PERSONAL_POSITION_PHRASES):
            add_review_flag(sentence, "个人诉求")
        if sentence.start < 0 or sentence.end < sentence.start:
            add_review_flag(sentence, "时间戳异常")

    for previous, current in zip(sentences, sentences[1:]):
        gap = current.start - previous.end
        if current.spk != previous.spk and gap <= 1000:
            add_review_flag(previous, "快速换人")
            add_review_flag(current, "快速换人")
        if gap < 0:
            add_review_flag(previous, "时间区间相交")
            add_review_flag(current, "时间区间相交")


def join_texts(texts: list[str]) -> str:
    result = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if not result:
            result = text
            continue
        if result[-1] in "，。！？；：、“‘（《" or text[0] in "，。！？；：、”’）》":
            result += text
        else:
            result += text
    return result


def merge_sentences(sentences: list[Sentence | ReviewedSpan], max_gap_ms: int, max_chars: int) -> list[Block]:
    blocks: list[Block] = []

    for sentence in sentences:
        if not blocks:
            blocks.append(
                Block(sentence.start, sentence.end, sentence.spk, [sentence.text], list(sentence.review_flags))
            )
            continue

        current = blocks[-1]
        gap = sentence.start - current.end
        merged_text = join_texts(current.texts + [sentence.text])
        should_merge = (
            sentence.spk == current.spk
            and gap <= max_gap_ms
            and len(merged_text) <= max_chars
        )

        if should_merge:
            current.end = max(current.end, sentence.end)
            current.texts.append(sentence.text)
            for flag in sentence.review_flags:
                if flag not in current.review_flags:
                    current.review_flags.append(flag)
        else:
            blocks.append(
                Block(sentence.start, sentence.end, sentence.spk, [sentence.text], list(sentence.review_flags))
            )

    return blocks


def format_time(ms: int) -> str:
    total_seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def normalize_speaker(spk: int | str, speaker_prefix: str) -> str:
    if isinstance(spk, str):
        if spk.startswith("SPEAKER_"):
            return spk
        if spk == "unknown":
            return "UNKNOWN"
        if spk == "overlap":
            return "OVERLAP"
    return f"{speaker_prefix}{spk}"


def build_filler_pattern(filler_words: list[str]) -> str:
    if not filler_words:
        return ""
    return "|".join(re.escape(word) for word in sorted(filler_words, key=len, reverse=True))


def clean_text(text: str, config: TextCleaningConfig) -> str:
    text = re.sub(r"\s+", "", text)
    filler_pattern = build_filler_pattern(config.filler_words)
    if filler_pattern:
        text = re.sub(rf"^(?:{filler_pattern})+[，,、。\s]*", "", text)
        text = re.sub(rf"[，,、\s]*(?:{filler_pattern})+[，,、。\s]*$", "", text)

    for word in config.repeated_words:
        pattern = f"(?:{re.escape(word)}){{2,}}"
        text = re.sub(pattern, word, text)

    text = re.sub(r"([一-鿿])\1{2,}", r"\1", text)
    text = re.sub(r"([，。！？；：、])\1+", r"\1", text)
    text = text.strip(" ，,、")
    return text


def should_drop_cleaned_text(text: str, config: TextCleaningConfig) -> bool:
    stripped = text.strip(" ，,、。！？；：")
    return stripped == "" or stripped in config.drop_words


def render_evidence_transcript(sentences: list[Sentence], speaker_prefix: str, keep_time: bool) -> str:
    rendered = []
    for sentence in sentences:
        speaker = normalize_speaker(sentence.spk, speaker_prefix)
        if keep_time:
            header = f"[{format_time(sentence.start)} - {format_time(sentence.end)}] {speaker}："
        else:
            header = f"{speaker}："
        rendered.append(f"{header}\n{sentence.text}")
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def render_blocks(
    blocks: list[Block],
    speaker_prefix: str,
    keep_time: bool,
    cleaned: bool,
    cleaning_config: TextCleaningConfig | None = None,
) -> str:
    if cleaned and cleaning_config is None:
        raise ValueError("cleaned=True 时必须提供 cleaning_config")

    rendered = []
    for block in blocks:
        text = clean_text(block.text, cleaning_config) if cleaned else block.text
        if cleaned and should_drop_cleaned_text(text, cleaning_config):
            continue

        speaker = normalize_speaker(block.spk, speaker_prefix)
        review_marker = f"【待回听：{'、'.join(block.review_flags)}】" if block.review_flags else ""
        if keep_time:
            header = f"[{format_time(block.start)} - {format_time(block.end)}] {speaker}{review_marker}："
        else:
            header = f"{speaker}{review_marker}："
        rendered.append(f"{header}\n{text}")

    return "\n\n".join(rendered) + ("\n" if rendered else "")


def split_transcript_blocks(transcript: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n{2,}", transcript.strip()) if block.strip()]


def parse_transcript_block(block: str, block_index: int) -> PolishBlock:
    lines = block.splitlines()
    if len(lines) < 2:
        raise ValueError(f"第 {block_index} 段格式错误，必须包含段落头和正文：{block[:80]}")

    header = lines[0].strip()
    text = "\n".join(line.strip() for line in lines[1:]).strip()
    if not re.match(r"^\[.+? - .+?\] .+：$", header):
        raise ValueError(f"第 {block_index} 段段落头格式错误：{header}")
    if not text:
        raise ValueError(f"第 {block_index} 段正文为空：{header}")
    return PolishBlock(header=header, text=text)


def parse_transcript_blocks(transcript: str) -> list[PolishBlock]:
    return [
        parse_transcript_block(block, index)
        for index, block in enumerate(split_transcript_blocks(transcript), start=1)
    ]


def chunk_blocks(blocks: list[PolishBlock], chunk_size: int) -> list[list[PolishBlock]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    return [blocks[index : index + chunk_size] for index in range(0, len(blocks), chunk_size)]


def render_polish_input(chunk: list[PolishBlock]) -> str:
    return "\n\n".join(
        f"<<<SEGMENT {index}>>>\n{block.text}"
        for index, block in enumerate(chunk, start=1)
    )


def build_polish_prompt(chunk_text: str, prompt_template: str) -> str:
    return prompt_template.replace("{{ chunk_text }}", chunk_text)


def parse_polished_texts(response: str, expected_count: int) -> list[str]:
    pattern = re.compile(r"(?m)^<<<SEGMENT (\d+)>>>\s*$")
    matches = list(pattern.finditer(response.strip()))
    if len(matches) != expected_count:
        raise ValueError(f"润色输出分段数量不一致：输入 {expected_count}，输出 {len(matches)}")

    polished_by_index: dict[int, str] = {}
    for position, match in enumerate(matches):
        segment_index = int(match.group(1))
        next_start = matches[position + 1].start() if position + 1 < len(matches) else len(response.strip())
        text = response.strip()[match.end():next_start].strip()
        if segment_index in polished_by_index:
            raise ValueError(f"润色输出分段编号重复：{segment_index}")
        if not text:
            raise ValueError(f"润色输出分段正文为空：{segment_index}")
        if re.match(r"^\[.+? - .+?\] .+：", text):
            raise ValueError(f"润色输出不应包含时间戳或说话人标签：{segment_index}")
        polished_by_index[segment_index] = text

    expected_indexes = list(range(1, expected_count + 1))
    actual_indexes = sorted(polished_by_index)
    if actual_indexes != expected_indexes:
        raise ValueError(f"润色输出分段编号不连续：期望 {expected_indexes}，实际 {actual_indexes}")
    return [polished_by_index[index] for index in expected_indexes]


def assemble_polished_chunk(chunk: list[PolishBlock], polished_texts: list[str]) -> str:
    if len(chunk) != len(polished_texts):
        raise ValueError(f"拼回段落数不一致：输入 {len(chunk)}，输出 {len(polished_texts)}")
    return "\n\n".join(
        f"{block.header}\n{polished_text}"
        for block, polished_text in zip(chunk, polished_texts)
    )


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str | None = None,
    enable_thinking: bool = False,
    messages: list[dict[str, str]] | None = None,
    response_format: dict[str, str] | None = None,
    timeout_seconds: int = 300,
) -> str:
    if messages is None:
        if prompt is None:
            raise ValueError("prompt 和 messages 不能同时为空")
        messages = [{"role": "user", "content": prompt}]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    payload["enable_thinking"] = enable_thinking

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")

    request = urllib.request.Request(
        normalize_base_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"大模型 API 请求失败：HTTP {error.code} {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"大模型 API 请求失败：{error}") from error

    choices = response_data.get("choices") or []
    if not choices:
        raise RuntimeError(f"大模型 API 未返回 choices：{response_data}")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content.strip():
        raise RuntimeError(f"大模型 API 返回内容为空：{response_data}")
    return content.strip()


def count_timestamp_blocks(text: str) -> int:
    return len(re.findall(r"(?m)^\[", text))


def polish_chunk(
    index: int,
    total: int,
    chunk: list[PolishBlock],
    base_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool,
    prompt_template: str,
    max_retries: int,
) -> tuple[int, str]:
    print(f"正在润色分块 {index}/{total}，段落数：{len(chunk)}", flush=True)
    prompt = build_polish_prompt(render_polish_input(chunk), prompt_template)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                enable_thinking=enable_thinking,
            )
            polished_texts = parse_polished_texts(response, len(chunk))
            return index, assemble_polished_chunk(chunk, polished_texts)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                print(f"分块 {index}/{total} 第 {attempt} 次润色失败，准备重试：{error}", flush=True)

    raise RuntimeError(f"分块 {index}/{total} 润色失败，已重试 {max_retries} 次：{last_error}") from last_error


def write_polished_transcript(
    source_path: Path,
    polished_path: Path,
    chunk_size: int,
    base_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool,
    max_workers: int,
    max_retries: int,
    prompt_template: str,
) -> int:
    transcript = source_path.read_text(encoding="utf-8")
    blocks = parse_transcript_blocks(transcript)
    if not blocks:
        raise ValueError(f"没有从润色输入中解析到段落：{source_path}")
    if max_workers <= 0:
        raise ValueError("max_workers 必须大于 0")
    if max_retries <= 0:
        raise ValueError("max_retries 必须大于 0")

    chunks = chunk_blocks(blocks, chunk_size)
    polished_chunks: dict[int, str] = {}
    workers = min(max_workers, len(chunks))
    if workers == 1:
        for index, chunk in enumerate(chunks, start=1):
            chunk_index, polished = polish_chunk(
                index, len(chunks), chunk, base_url, api_key, model, enable_thinking, prompt_template, max_retries
            )
            polished_chunks[chunk_index] = polished
    else:
        print(f"并行调用大模型，并发数：{workers}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    polish_chunk,
                    index,
                    len(chunks),
                    chunk,
                    base_url,
                    api_key,
                    model,
                    enable_thinking,
                    prompt_template,
                    max_retries,
                )
                for index, chunk in enumerate(chunks, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                chunk_index, polished = future.result()
                polished_chunks[chunk_index] = polished
                print(f"完成润色分块 {chunk_index}/{len(chunks)}", flush=True)

    polished_text = "\n\n".join(polished_chunks[index] for index in range(1, len(chunks) + 1))
    expected_total = len(blocks)
    actual_total = count_timestamp_blocks(polished_text)
    if actual_total != expected_total:
        raise RuntimeError(f"润色后总段落数不一致：输入 {expected_total}，输出 {actual_total}")

    polished_path.write_text(polished_text.strip() + "\n", encoding="utf-8")
    return len(chunks)


FINAL_DROP_REASONS = {"FILLER_ONLY", "NOISE_ONLY", "EXACT_REPETITION"}
FINAL_HARD_SUBJECTS = ("公司", "单位", "员工", "本单位", "人事", "领导")
FINAL_SOFT_SUBJECTS = ("我", "你", "我们", "你们", "他", "她", "咱们")
FINAL_CONDITIONS = ("如果", "除非", "只有", "否则")
FINAL_TOPICS = ("补偿", "赔偿", "工资", "薪资", "社保", "公积金", "年假", "欠薪")
FINAL_FACT_KINDS = (
    "money", "date", "n_plus_one", "number", "acceptance", "rejection",
    "commitment", "condition", "subject",
)
FINAL_JOIN_PUNCTUATION = frozenset("，,。！？?；：、")
FINAL_N_PLUS_ONE_PATTERN = re.compile(r"(?i)n\s*(?:\+|加)\s*1")
FINAL_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
FINAL_EXPLICIT_MONEY_PATTERN = re.compile(
    r"(?:[￥¥]\d+(?:\.\d+)?|\d+(?:\.\d+)?(?:元|块|万(?:元)?)|"
    r"[零〇一二两三四五六七八九十百千万幺]+(?:元|块|万(?:元)?))"
)
FINAL_CONTEXTUAL_MONEY_PATTERN = re.compile(
    r"(?:工资|薪资|补偿(?:金)?|赔偿(?:金)?|基数).{0,8}?"
    r"(?P<value>(?:\d{2,})|(?:[零〇一二两三四五六七八九幺]*"
    r"(?:十|百|千|万)[零〇一二两三四五六七八九十百千万幺]*))"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def span_reference(span: ReviewedSpan) -> dict[str, Any]:
    return {
        "source_id": span.source_id,
        "char_start": span.char_start,
        "char_end": span.char_end,
    }


def is_final_hard_boundary(spk: int | str) -> bool:
    return str(spk) in {"unknown", "overlap"}


def ends_complete_sentence(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in "。！？；"


def build_reading_units(spans: list[ReviewedSpan], max_gap_ms: int, max_chars: int) -> list[ReadingUnit]:
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    units: list[ReadingUnit] = []
    for span in spans:
        if not units:
            units.append(ReadingUnit(
                unit_id="u0001",
                start=span.start,
                end=span.end,
                spk=span.spk,
                texts=[span.text],
                source_spans=[span_reference(span)],
                review_flags=list(span.review_flags),
                review_required=span.review_required,
            ))
            continue
        current = units[-1]
        merged_text = join_texts(current.texts + [span.text])
        gap = span.start - current.end
        exceeds_hard_limit = len(merged_text) > max_chars * 2
        exceeds_soft_limit_at_sentence_end = len(merged_text) > max_chars and ends_complete_sentence(current.text)
        should_merge = (
            span.spk == current.spk
            and not is_final_hard_boundary(span.spk)
            and gap <= max_gap_ms
            and not exceeds_hard_limit
            and not exceeds_soft_limit_at_sentence_end
        )
        if should_merge:
            current.end = max(current.end, span.end)
            current.texts.append(span.text)
            current.source_spans.append(span_reference(span))
            current.review_required = current.review_required or span.review_required
            for flag in span.review_flags:
                if flag not in current.review_flags:
                    current.review_flags.append(flag)
            continue
        units.append(ReadingUnit(
            unit_id=f"u{len(units) + 1:04d}",
            start=span.start,
            end=span.end,
            spk=span.spk,
            texts=[span.text],
            source_spans=[span_reference(span)],
            review_flags=list(span.review_flags),
            review_required=span.review_required,
        ))
    return units


def clean_reading_unit(unit: ReadingUnit, config: TextCleaningConfig) -> tuple[str, list[dict[str, str]]]:
    original = unit.text
    text = re.sub(r"\s+", "", original)
    changes = []
    if text != original:
        changes.append({"reason_code": "REMOVE_WHITESPACE", "before": original, "after": text})
    filler_pattern = build_filler_pattern(config.filler_words)
    if filler_pattern:
        cleaned = re.sub(rf"^(?:{filler_pattern})+[，,、。\s]*", "", text)
        cleaned = re.sub(rf"[，,、\s]*(?:{filler_pattern})+[，,、。\s]*$", "", cleaned)
        if cleaned != text:
            changes.append({"reason_code": "EDGE_FILLER", "before": text, "after": cleaned})
            text = cleaned
    return text.strip(" ，,、"), changes


def is_droppable_final_text(text: str, config: TextCleaningConfig) -> bool:
    stripped = text.strip(" ，,、。！？；：")
    return stripped == "" or stripped in config.drop_words or stripped in config.filler_words


def punctuation_join_view(text: str) -> str:
    return "".join(
        character for character in text
        if not character.isspace() and character not in FINAL_JOIN_PUNCTUATION
    )


def raw_compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def normalize_lock_text(text: str) -> str:
    return punctuation_join_view(text)


def normalize_fact_clause(text: str) -> str:
    return punctuation_join_view(text)


def extract_protected_atoms(text: str) -> list[ProtectedAtom]:
    joined = punctuation_join_view(text)
    candidates: list[ProtectedAtom] = []

    def add_matches(kind: str, pattern: re.Pattern[str], group: str | int = 0) -> None:
        for match in pattern.finditer(joined):
            start, end = match.span(group)
            candidates.append(ProtectedAtom(kind, match.group(group), start, end))

    add_matches("money", FINAL_EXPLICIT_MONEY_PATTERN)
    add_matches("money", FINAL_CONTEXTUAL_MONEY_PATTERN, "value")
    add_matches("date", DATE_PATTERN)
    add_matches("n_plus_one", FINAL_N_PLUS_ONE_PATTERN)
    add_matches("number", FINAL_NUMBER_PATTERN)

    phrase_groups = (
        ("rejection", REJECTION_PHRASES),
        ("acceptance", ACCEPTANCE_PHRASES),
        ("commitment", COMMITMENT_PHRASES),
        ("condition", FINAL_CONDITIONS),
        ("subject", FINAL_HARD_SUBJECTS),
    )
    for kind, phrases in phrase_groups:
        for phrase in phrases:
            if phrase in {"可以", "愿意", "没问题"} and joined not in SHORT_ANSWER_WORDS:
                continue
            add_matches(kind, re.compile(re.escape(phrase)))

    atoms = []
    for atom in sorted(candidates, key=lambda item: (item.start, -(item.end - item.start), item.kind, item.value)):
        if not any(atom.start < kept.end and atom.end > kept.start for kept in atoms):
            atoms.append(atom)
    return atoms


def extract_fact_lock_occurrences(text: str) -> list[tuple[int, int, str]]:
    return [
        (atom.start, atom.end, atom.value)
        for atom in extract_protected_atoms(text)
    ]


def extract_fact_locks(text: str) -> list[str]:
    return list(dict.fromkeys(atom.value for atom in extract_protected_atoms(text)))


def protected_facts_payload(text: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for atom in extract_protected_atoms(text):
        grouped.setdefault(atom.kind, [])
        if atom.value not in grouped[atom.kind]:
            grouped[atom.kind].append(atom.value)
    return grouped


def split_protected_clauses(text: str) -> list[str]:
    return [clause for clause in re.split(r"[，,。！？?；：、]+|\n+", text) if clause.strip()]


def extract_protected_relations(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relations = []
    warnings = []
    for clause_index, clause in enumerate(split_protected_clauses(text)):
        atoms = extract_protected_atoms(clause)
        subjects = sorted({atom.value for atom in atoms if atom.kind == "subject"})
        core = [atom for atom in atoms if atom.kind in {
            "money", "date", "n_plus_one", "number", "acceptance", "rejection",
            "commitment", "condition",
        }]
        topics = sorted(topic for topic in FINAL_TOPICS if topic in punctuation_join_view(clause))
        relation_atoms = [
            atom for atom in core
            if atom.kind in {"acceptance", "rejection", "commitment"}
        ]
        if not subjects or not relation_atoms:
            continue
        values_by_kind = {
            kind: sorted({atom.value for atom in core if atom.kind == kind})
            for kind in FINAL_FACT_KINDS
            if kind != "subject" and any(atom.kind == kind for atom in core)
        }
        if len(subjects) > 1 or any(len(values) > 1 for values in values_by_kind.values()):
            warnings.append({
                "code": "LOW_CONFIDENCE_RELATION_NOT_ENFORCED",
                "clause_index": clause_index,
            })
            continue
        relations.append({
            "subject": subjects[0],
            "topics": topics,
            "facts": values_by_kind,
        })
    return relations, warnings


def relation_key(relation: dict[str, Any]) -> str:
    return json.dumps(relation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relation_satisfies(output_relation: dict[str, Any], source_relation: dict[str, Any]) -> bool:
    if output_relation["subject"] != source_relation["subject"]:
        return False
    if not set(source_relation["topics"]).issubset(output_relation["topics"]):
        return False
    for kind, source_values in source_relation["facts"].items():
        if not set(source_values).issubset(output_relation["facts"].get(kind, [])):
            return False
    return True


def soft_subject_warnings(source_text: str, output_text: str) -> list[dict[str, Any]]:
    source = punctuation_join_view(source_text)
    output = punctuation_join_view(output_text)
    return [
        {"code": "SOFT_SUBJECT_ADDED", "value": subject}
        for subject in FINAL_SOFT_SUBJECTS
        if len(subject) > 1 and subject in output and subject not in source
    ]


def validate_protected_content(source_text: str, output_text: str) -> list[dict[str, Any]]:
    source_atoms = extract_protected_atoms(source_text)
    output_atoms = extract_protected_atoms(output_text)
    source_by_kind = {
        kind: {atom.value for atom in source_atoms if atom.kind == kind}
        for kind in FINAL_FACT_KINDS
    }
    output_by_kind = {
        kind: {atom.value for atom in output_atoms if atom.kind == kind}
        for kind in FINAL_FACT_KINDS
    }
    missing = {
        kind: sorted(source_by_kind[kind] - output_by_kind[kind])
        for kind in FINAL_FACT_KINDS
        if source_by_kind[kind] - output_by_kind[kind]
    }
    if missing:
        raise FinalValidationError(
            "PROTECTED_FACT_MISSING",
            f"paragraph 丢失关键事实：{missing}",
            {"missing": missing},
        )
    added = {
        kind: sorted(output_by_kind[kind] - source_by_kind[kind])
        for kind in FINAL_FACT_KINDS
        if output_by_kind[kind] - source_by_kind[kind]
    }
    if added:
        raise FinalValidationError(
            "PROTECTED_FACT_ADDED",
            f"paragraph 新增关键事实：{added}",
            {"added": added},
        )

    source_relations, source_relation_warnings = extract_protected_relations(source_text)
    output_relations, output_relation_warnings = extract_protected_relations(output_text)
    missing_relations = [
        relation_key(source_relation)
        for source_relation in source_relations
        if not any(relation_satisfies(output_relation, source_relation) for output_relation in output_relations)
    ]
    if missing_relations:
        raise FinalValidationError(
            "PROTECTED_RELATION_CHANGED",
            "paragraph 改变关键事实关系",
            {"missing_relations": missing_relations},
        )

    raw_source = raw_compact_text(source_text)
    warnings = source_relation_warnings + output_relation_warnings + soft_subject_warnings(source_text, output_text)
    warnings.extend(
        {"code": "PROTECTED_RELATION_ADDED", "relation": relation_key(output_relation)}
        for output_relation in output_relations
        if not any(relation_satisfies(output_relation, source_relation) for source_relation in source_relations)
    )
    for atom in output_atoms:
        if atom.value in source_by_kind[atom.kind] and atom.value not in raw_source:
            warnings.append({
                "code": "PUNCTUATION_JOIN_USED",
                "kind": atom.kind,
                "value": atom.value,
            })
    return warnings


def prepare_reading_units(
    spans: list[ReviewedSpan],
    max_gap_ms: int,
    max_chars: int,
    cleaning_config: TextCleaningConfig,
) -> tuple[list[ReadingUnit], list[dict[str, Any]]]:
    units = build_reading_units(spans, max_gap_ms, max_chars)
    prepared = []
    cleanup_audit = []
    for unit in units:
        cleaned, changes = clean_reading_unit(unit, cleaning_config)
        if is_droppable_final_text(cleaned, cleaning_config):
            cleanup_audit.append({
                "unit_id": unit.unit_id,
                "source_spans": unit.source_spans,
                "reason_code": "FILLER_ONLY",
                "before": unit.text,
                "after": cleaned,
            })
            continue
        unit.texts = [cleaned]
        if changes:
            cleanup_audit.append({
                "unit_id": unit.unit_id,
                "source_spans": unit.source_spans,
                "reason_code": "DETERMINISTIC_CLEANING",
                "changes": changes,
            })
        prepared.append(unit)
    return prepared, cleanup_audit


def serialize_reading_unit(unit: ReadingUnit, speaker_prefix: str) -> dict[str, Any]:
    protected_facts = protected_facts_payload(unit.text)
    protected_relations, _ = extract_protected_relations(unit.text)
    payload = {
        "unit_id": unit.unit_id,
        "speaker": normalize_speaker(unit.spk, speaker_prefix),
        "text": unit.text,
        "review_required": unit.review_required,
        "review_flags": unit.review_flags,
    }
    if protected_facts:
        payload["protected_facts"] = protected_facts
    if protected_relations:
        payload["protected_relations"] = protected_relations
    return payload


def load_final_prompt_template(prompt_dir: Path) -> str:
    path = prompt_dir / "polish_prompt_template.txt"
    template = path.read_text(encoding="utf-8")
    if "{{ final_input }}" not in template:
        raise ValueError(f"最终阅读提示词模板必须包含占位符 {{{{ final_input }}}}：{path}")
    return template


def build_final_prompt(units: list[ReadingUnit], speaker_prefix: str, template: str) -> str:
    payload = {"units": [serialize_reading_unit(unit, speaker_prefix) for unit in units]}
    return template.replace("{{ final_input }}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def build_final_retry_prompt(prompt: str, error: FinalValidationError) -> str:
    feedback = json.dumps(
        {"code": error.code, "details": error.details},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"{prompt}\n\n"
        "上一次输出未通过程序校验。请重新生成完整 JSON，不要解释、不要沿用错误输出；"
        "必须保留输入中对应的受保护事实和关系，且不得新增以下校验反馈未支持的内容："
        f"{feedback}"
    )


def parse_final_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise FinalValidationError("RESPONSE_JSON_INVALID", "最终阅读整理未返回合法 JSON") from error
    if not isinstance(payload, dict):
        raise FinalValidationError("SCHEMA_INVALID", "最终阅读整理必须返回 JSON 对象")
    return payload


def validate_final_response(
    payload: dict[str, Any],
    units: list[ReadingUnit],
    speaker_prefix: str,
    cleaning_config: TextCleaningConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paragraphs = payload.get("paragraphs")
    dropped_units = payload.get("dropped_units")
    if not isinstance(paragraphs, list) or not isinstance(dropped_units, list):
        raise FinalValidationError("SCHEMA_INVALID", "最终阅读整理必须包含 paragraphs 和 dropped_units 数组")
    unit_by_id = {unit.unit_id: unit for unit in units}
    unit_index = {unit.unit_id: index for index, unit in enumerate(units)}
    expected_ids = list(unit_by_id)
    seen_ids: set[str] = set()
    normalized_paragraphs = []
    warnings = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            raise FinalValidationError("SCHEMA_INVALID", "paragraph 必须是对象")
        source_unit_ids = paragraph.get("source_unit_ids")
        speaker = paragraph.get("speaker")
        text = paragraph.get("text")
        if not isinstance(source_unit_ids, list) or not source_unit_ids or not isinstance(text, str) or not text.strip():
            raise FinalValidationError("SCHEMA_INVALID", "paragraph 的 source_unit_ids 或 text 非法")
        if len(set(source_unit_ids)) != len(source_unit_ids) or any(unit_id not in unit_by_id for unit_id in source_unit_ids):
            raise FinalValidationError("UNIT_REFERENCE_INVALID", "paragraph 引用了非法或重复的 unit")
        indexes = [unit_index[unit_id] for unit_id in source_unit_ids]
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise FinalValidationError("UNIT_NOT_CONTIGUOUS", "paragraph 的 source_unit_ids 必须连续且有序")
        source_units = [unit_by_id[unit_id] for unit_id in source_unit_ids]
        expected_speaker = normalize_speaker(source_units[0].spk, speaker_prefix)
        if speaker != expected_speaker or any(normalize_speaker(unit.spk, speaker_prefix) != speaker for unit in source_units):
            raise FinalValidationError("SPEAKER_CHANGED", "paragraph 不得跨 speaker 或修改 speaker")
        if re.match(r"^\[.+? - .+?\] .+：", text.strip()):
            raise FinalValidationError("TEXT_HEADER_FORBIDDEN", "paragraph 正文不得包含时间戳或 speaker 标签")
        source_text = "\n".join(unit.text for unit in source_units)
        paragraph_warnings = validate_protected_content(source_text, text)
        for warning in paragraph_warnings:
            warnings.append({"paragraph_index": paragraph_index, **warning})
        duplicate_ids = seen_ids & set(source_unit_ids)
        if duplicate_ids:
            raise FinalValidationError("UNIT_DUPLICATED", f"unit 被重复使用：{sorted(duplicate_ids)}")
        seen_ids.update(source_unit_ids)
        normalized_paragraphs.append({
            "speaker": speaker,
            "source_unit_ids": source_unit_ids,
            "text": text.strip(),
            "start": source_units[0].start,
            "end": source_units[-1].end,
            "review_required": any(unit.review_required for unit in source_units),
        })
    normalized_dropped = []
    for dropped in dropped_units:
        if not isinstance(dropped, dict):
            raise FinalValidationError("DROPPED_UNIT_INVALID", "dropped_unit 必须是对象")
        unit_id = dropped.get("source_unit_id")
        reason_code = dropped.get("reason_code")
        if unit_id not in unit_by_id or reason_code not in FINAL_DROP_REASONS:
            raise FinalValidationError("DROPPED_UNIT_INVALID", "dropped_unit 非法")
        unit = unit_by_id[unit_id]
        if extract_fact_locks(unit.text) or not is_droppable_final_text(unit.text, cleaning_config):
            raise FinalValidationError("SUBSTANTIVE_UNIT_DROPPED", "含实质内容或关键事实的 unit 不得删除")
        if unit_id in seen_ids:
            raise FinalValidationError("UNIT_DUPLICATED", "unit 不得同时输出和删除")
        seen_ids.add(unit_id)
        normalized_dropped.append({"source_unit_id": unit_id, "reason_code": reason_code})
    if seen_ids != set(expected_ids):
        raise FinalValidationError(
            "UNIT_COVERAGE_MISSING",
            f"最终阅读整理未完整覆盖 unit：缺失 {sorted(set(expected_ids) - seen_ids)}",
        )
    paragraph_indexes = [unit_index[paragraph["source_unit_ids"][0]] for paragraph in normalized_paragraphs]
    if paragraph_indexes != sorted(paragraph_indexes):
        raise FinalValidationError("PARAGRAPH_ORDER_CHANGED", "paragraph 顺序必须与输入 unit 一致")
    return normalized_paragraphs, normalized_dropped, {
        "hard_checks": {
            "structure": True,
            "protected_facts": True,
            "protected_relations": True,
        },
        "warnings": warnings,
    }


def preserve_fallback_facts(original: str, cleaned: str) -> bool:
    try:
        validate_protected_content(original, cleaned)
    except FinalValidationError:
        return False
    return True


def clean_fallback_text(text: str, config: TextCleaningConfig) -> str:
    cleaned = clean_text(text, config)
    cleaned = re.sub(r"([我你他她这那的了是为嗯啊呃])\1+", r"\1", cleaned)
    return cleaned if cleaned and preserve_fallback_facts(text, cleaned) else text


def fallback_final_paragraphs(
    units: list[ReadingUnit],
    speaker_prefix: str,
    cleaning_config: TextCleaningConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paragraphs = []
    cleanup = []
    for unit in units:
        text = clean_fallback_text(unit.text, cleaning_config)
        if text != unit.text:
            cleanup.append({
                "unit_id": unit.unit_id,
                "reason_code": "DETERMINISTIC_FALLBACK_CLEANING",
                "before": unit.text,
                "after": text,
            })
        paragraphs.append({
            "speaker": normalize_speaker(unit.spk, speaker_prefix),
            "source_unit_ids": [unit.unit_id],
            "text": text,
            "start": unit.start,
            "end": unit.end,
            "review_required": unit.review_required,
        })
    return paragraphs, cleanup


def render_final_transcript(paragraphs: list[dict[str, Any]], keep_time: bool) -> str:
    rendered = []
    for paragraph in paragraphs:
        marker = "【待回听】" if paragraph["review_required"] else ""
        header = (
            f"[{format_time(paragraph['start'])} - {format_time(paragraph['end'])}] {paragraph['speaker']}{marker}："
            if keep_time else f"{paragraph['speaker']}{marker}："
        )
        rendered.append(f"{header}\n{paragraph['text']}")
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def write_final_transcript(
    spans: list[ReviewedSpan],
    final_path: Path,
    final_audit_path: Path,
    max_gap_ms: int,
    max_chars: int,
    speaker_prefix: str,
    keep_time: bool,
    prompt_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool,
    chunk_size: int,
    max_retries: int,
    source_json_sha256: str,
) -> dict[str, Any]:
    if chunk_size <= 0 or max_retries <= 0:
        raise ValueError("chunk_size 和 max_retries 必须大于 0")
    cleaning_config = load_cleaning_config(prompt_dir)
    units, cleanup_audit = prepare_reading_units(spans, max_gap_ms, max_chars, cleaning_config)
    if not units:
        raise ValueError("没有可供生成最终阅读版的内容")
    template = load_final_prompt_template(prompt_dir)
    chunks = [units[index:index + chunk_size] for index in range(0, len(units), chunk_size)]
    def process_chunk(
        chunk: list[ReadingUnit],
        chunk_index: int | str,
        repartitioned_from: int | str | None = None,
        inherited_failures: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        base_prompt = build_final_prompt(chunk, speaker_prefix, template)
        prompt_sha256 = sha256_text(base_prompt)
        failure_history = list(inherited_failures or [])
        request_attempts = []
        last_error: Exception | None = None
        retry_error: FinalValidationError | None = None
        failure_category: str | None = None
        for attempt in range(1, max_retries + 1):
            prompt = (
                build_final_retry_prompt(base_prompt, retry_error)
                if retry_error is not None else base_prompt
            )
            request_attempt = {
                "attempt": attempt,
                "prompt_sha256": sha256_text(prompt),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "retrying_validation_code": retry_error.code if retry_error is not None else None,
            }
            request_attempts.append(request_attempt)
            try:
                response = call_openai_compatible_chat(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                )
            except Exception as error:
                last_error = error
                failure_category = "transport"
                failure_history.append({
                    "attempt": attempt,
                    "category": "transport",
                    "code": type(error).__name__,
                    "message": str(error),
                    "prompt_sha256": request_attempt["prompt_sha256"],
                })
                continue
            try:
                paragraphs, dropped, validation = validate_final_response(
                    parse_final_response(response), chunk, speaker_prefix, cleaning_config
                )
            except FinalValidationError as error:
                last_error = error
                retry_error = error
                failure_category = "hard_validation"
                failure_history.append({
                    "attempt": attempt,
                    "category": "hard_validation",
                    "code": error.code,
                    "message": str(error),
                    "details": error.details,
                    "prompt_sha256": request_attempt["prompt_sha256"],
                })
                continue
            audit = {
                "chunk_index": chunk_index,
                "status": "ok_with_warnings" if validation["warnings"] else "ok",
                "unit_ids": [unit.unit_id for unit in chunk],
                "prompt_sha256": prompt_sha256,
                "prompt_bytes": len(base_prompt.encode("utf-8")),
                "request_attempts": request_attempts,
                "attempts": attempt,
                "paragraphs": paragraphs,
                "dropped_units": dropped,
                "validation": validation,
                "failure_history": failure_history,
            }
            if repartitioned_from is not None:
                audit["repartitioned_from"] = repartitioned_from
            return paragraphs, [audit]
        if failure_category == "hard_validation" and len(chunk) > 2:
            midpoint = len(chunk) // 2
            left_paragraphs, left_audits = process_chunk(
                chunk[:midpoint], f"{chunk_index}.1", chunk_index, failure_history
            )
            right_paragraphs, right_audits = process_chunk(
                chunk[midpoint:], f"{chunk_index}.2", chunk_index, failure_history
            )
            return left_paragraphs + right_paragraphs, left_audits + right_audits
        fallback, fallback_cleanup = fallback_final_paragraphs(
            chunk, speaker_prefix, cleaning_config
        )
        error_code = last_error.code if isinstance(last_error, FinalValidationError) else type(last_error).__name__
        audit = {
            "chunk_index": chunk_index,
            "status": "fallback",
            "unit_ids": [unit.unit_id for unit in chunk],
            "prompt_sha256": prompt_sha256,
            "prompt_bytes": len(base_prompt.encode("utf-8")),
            "request_attempts": request_attempts,
            "attempts": max_retries,
            "failure_category": failure_category,
            "failure_code": error_code,
            "error": str(last_error),
            "failure_history": failure_history,
            "paragraphs": fallback,
            "dropped_units": [],
            "deterministic_cleanup": fallback_cleanup,
        }
        if repartitioned_from is not None:
            audit["repartitioned_from"] = repartitioned_from
        return fallback, [audit]

    final_paragraphs: list[dict[str, Any]] = []
    chunk_audits = []
    for index, chunk in enumerate(chunks, start=1):
        paragraphs, audits = process_chunk(chunk, index)
        final_paragraphs.extend(paragraphs)
        chunk_audits.extend(audits)
    final_text = render_final_transcript(final_paragraphs, keep_time)
    final_path.write_text(final_text, encoding="utf-8")
    final_audit = {
        "schema_version": 2,
        "run": {
            "source_json_sha256": source_json_sha256,
            "reviewed_span_count": len(spans),
            "reading_unit_count": len(units),
            "final_paragraph_count": len(final_paragraphs),
            "model": model,
            "prompt_template_sha256": sha256_text(template),
        },
        "reading_units": [
            {
                **serialize_reading_unit(unit, speaker_prefix),
                "start": unit.start,
                "end": unit.end,
                "source_spans": unit.source_spans,
            }
            for unit in units
        ],
        "deterministic_cleanup": cleanup_audit,
        "chunks": chunk_audits,
        "integrity": {
            "unit_coverage_passed": True,
            "unit_order_passed": True,
            "speaker_consistency_passed": True,
            "protected_facts_passed": True,
            "protected_relations_passed": True,
            "warning_chunk_count": sum(chunk["status"] == "ok_with_warnings" for chunk in chunk_audits),
            "warning_count": sum(len(chunk.get("validation", {}).get("warnings", [])) for chunk in chunk_audits),
            "fallback_chunk_count": sum(chunk["status"] == "fallback" for chunk in chunk_audits),
            "hard_validation_fallback_count": sum(
                chunk.get("failure_category") == "hard_validation" for chunk in chunk_audits
            ),
            "transport_fallback_count": sum(
                chunk.get("failure_category") == "transport" for chunk in chunk_audits
            ),
            "final_sha256": sha256_text(final_text),
        },
    }
    final_audit_path.write_text(json.dumps(final_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_audit


def reviewed_spans_from_audit(audit: dict[str, Any]) -> list[ReviewedSpan]:
    raw_spans = audit.get("reviewed_spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("speaker review 审计缺少 reviewed_spans")
    try:
        return [ReviewedSpan(**span) for span in raw_spans]
    except TypeError as error:
        raise ValueError("speaker review 审计中的 reviewed_spans 格式错误") from error
