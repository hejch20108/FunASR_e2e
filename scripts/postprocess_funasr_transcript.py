#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sentence:
    start: int
    end: int
    spk: int | str
    text: str


@dataclass
class Block:
    start: int
    end: int
    spk: int | str
    texts: list[str]

    @property
    def text(self) -> str:
        return join_texts(self.texts)


@dataclass
class TextCleaningConfig:
    repeated_words: list[str]
    drop_words: set[str]
    filler_words: list[str]


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


def load_sentences(json_path: Path) -> list[Sentence]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sentences = []

    if isinstance(data, dict):
        items = data.get("sentence_info") or data.get("segments") or []
    elif isinstance(data, list):
        items = []
        for result in data:
            if isinstance(result, dict):
                items.extend(result.get("sentence_info") or [])
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("sentence") or "").strip()
        if not text:
            continue
        start = item.get("start")
        end = item.get("end")
        spk = item.get("spk", item.get("speaker", "unknown"))
        if start is None or end is None:
            continue
        sentences.append(Sentence(start=int(float(start)), end=int(float(end)), spk=spk, text=text))

    return sorted(sentences, key=lambda sentence: (sentence.start, sentence.end))


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


def merge_sentences(sentences: list[Sentence], max_gap_ms: int, max_chars: int) -> list[Block]:
    blocks: list[Block] = []

    for sentence in sentences:
        if not blocks:
            blocks.append(Block(sentence.start, sentence.end, sentence.spk, [sentence.text]))
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
        else:
            blocks.append(Block(sentence.start, sentence.end, sentence.spk, [sentence.text]))

    return blocks


def format_time(ms: int) -> str:
    total_seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def normalize_speaker(spk: int | str, speaker_prefix: str) -> str:
    if isinstance(spk, str) and spk.startswith("SPEAKER_"):
        return spk
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
        if keep_time:
            header = f"[{format_time(block.start)} - {format_time(block.end)}] {speaker}："
        else:
            header = f"{speaker}："
        rendered.append(f"{header}\n{text}")

    return "\n\n".join(rendered) + ("\n" if rendered else "")


def split_transcript_blocks(transcript: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n{2,}", transcript.strip()) if block.strip()]


def chunk_blocks(blocks: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    return [blocks[index : index + chunk_size] for index in range(0, len(blocks), chunk_size)]


def build_polish_prompt(chunk_text: str, prompt_template: str) -> str:
    return prompt_template.replace("{{ chunk_text }}", chunk_text)


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
    prompt: str,
    enable_thinking: bool,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if enable_thinking:
        payload["enable_thinking"] = True

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
        with urllib.request.urlopen(request, timeout=300) as response:
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
    chunk: list[str],
    base_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool,
    prompt_template: str,
) -> tuple[int, str]:
    print(f"正在润色分块 {index}/{total}，段落数：{len(chunk)}", flush=True)
    prompt = build_polish_prompt("\n\n".join(chunk), prompt_template)
    polished = call_openai_compatible_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        enable_thinking=enable_thinking,
    )
    expected_blocks = len(chunk)
    actual_blocks = count_timestamp_blocks(polished)
    if actual_blocks != expected_blocks:
        raise RuntimeError(
            f"分块 {index}/{total} 润色后段落数不一致：输入 {expected_blocks}，输出 {actual_blocks}"
        )
    return index, polished


def write_polished_transcript(
    source_path: Path,
    polished_path: Path,
    chunk_size: int,
    base_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool,
    max_workers: int,
    prompt_template: str,
) -> int:
    transcript = source_path.read_text(encoding="utf-8")
    blocks = split_transcript_blocks(transcript)
    if not blocks:
        raise ValueError(f"没有从润色输入中解析到段落：{source_path}")
    if max_workers <= 0:
        raise ValueError("max_workers 必须大于 0")

    chunks = chunk_blocks(blocks, chunk_size)
    polished_chunks: dict[int, str] = {}
    workers = min(max_workers, len(chunks))
    if workers == 1:
        for index, chunk in enumerate(chunks, start=1):
            chunk_index, polished = polish_chunk(
                index, len(chunks), chunk, base_url, api_key, model, enable_thinking, prompt_template
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
