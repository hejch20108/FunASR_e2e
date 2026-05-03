#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FUNASR_ROOT = PROJECT_DIR.parent

for path in (SCRIPT_DIR, PROJECT_DIR, FUNASR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from funasr import AutoModel
from postprocess_funasr_transcript import (
    load_cleaning_config,
    load_env_file,
    load_prompt_template,
    load_sentences,
    merge_sentences,
    render_blocks,
    write_polished_transcript,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FunASR_e2e 音频转写、说话人识别和大模型润色端到端流水线。")
    parser.add_argument("--settings", default="settings.yaml", help="配置文件路径，默认 settings.yaml")
    return parser


def default_settings() -> dict[str, Any]:
    return {
        "paths": {
            "input_audio_dir": "input_audio",
            "output_dir": "output",
            "env_file": ".env",
            "prompt_dir": "prompt",
        },
        "audio": {
            "mode": "batch",
            "input_audio_file": None,
            "supported_extensions": [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"],
        },
        "funasr": {
            "model": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "punc_model": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            "spk_model": "iic/speech_campplus_sv_zh-cn_16k-common",
            "device": "cuda",
            "batch_size_s": 300,
            "batch_size_threshold_s": 60,
            "max_single_segment_time": 60000,
        },
        "postprocess": {
            "max_gap_ms": 2000,
            "max_chars": 400,
            "speaker_prefix": "说话人",
            "keep_time": True,
        },
        "llm": {
            "skip_polish": False,
            "provider": "dashscope",
            "model": None,
            "chunk_size": 20,
            "max_workers": 8,
            "max_retries": 3,
            "enable_thinking": False,
            "api_key_env": "API_KEY",
            "base_url_env": "BASE_URL",
            "model_name_env": "MODEL_NAME",
            "polished_suffix": "_polished",
        },
        "output": {
            "per_audio_subdir": True,
            "overwrite": True,
        },
    }


def deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(settings_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("缺少 PyYAML，请在 FunASR_e2e 虚拟环境中执行：uv pip install pyyaml") from error

    if not settings_path.exists():
        raise SystemExit(f"配置文件不存在：{settings_path}")

    loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"配置文件格式错误，顶层必须是 YAML 对象：{settings_path}")
    return deep_merge(default_settings(), loaded)


def resolve_project_path(project_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def discover_audio_files(settings: dict[str, Any], project_dir: Path) -> list[Path]:
    paths = settings["paths"]
    audio = settings["audio"]
    mode = str(audio.get("mode", "batch")).lower()
    if mode not in {"single", "batch"}:
        raise SystemExit(f"audio.mode 只支持 single 或 batch，当前值：{mode}")

    input_audio_file = resolve_project_path(project_dir, audio.get("input_audio_file"))
    if input_audio_file is not None:
        if not input_audio_file.exists():
            raise SystemExit(f"指定音频文件不存在：{input_audio_file}")
        return [input_audio_file]

    if mode == "single":
        raise SystemExit("audio.mode=single 时必须配置 audio.input_audio_file")

    input_audio_dir = resolve_project_path(project_dir, paths["input_audio_dir"])
    if input_audio_dir is None or not input_audio_dir.exists():
        raise SystemExit(f"音频目录不存在：{input_audio_dir}")

    supported_extensions = {str(ext).lower() for ext in audio.get("supported_extensions", [])}
    audio_files = sorted(
        path for path in input_audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    if not audio_files:
        raise SystemExit(f"未在 {input_audio_dir} 中找到支持的音频文件，请先上传音频。")
    return audio_files


def build_audio_output_dir(base_output_dir: Path, audio_path: Path, per_audio_subdir: bool) -> Path:
    return base_output_dir / audio_path.stem if per_audio_subdir else base_output_dir


def dump_json_output(result: list[dict], json_path: Path, overwrite: bool) -> None:
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在且 overwrite=false：{json_path}")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_cleaned_transcript(
    json_path: Path,
    cleaned_path: Path,
    max_gap_ms: int,
    max_chars: int,
    speaker_prefix: str,
    keep_time: bool,
    prompt_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    if cleaned_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在且 overwrite=false：{cleaned_path}")

    sentences = load_sentences(json_path)
    if not sentences:
        raise SystemExit(f"没有从 FunASR JSON 中解析到 sentence_info：{json_path}")

    blocks = merge_sentences(sentences, max_gap_ms, max_chars)
    cleaning_config = load_cleaning_config(prompt_dir)
    cleaned_path.write_text(
        render_blocks(blocks, speaker_prefix, keep_time, cleaned=True, cleaning_config=cleaning_config),
        encoding="utf-8",
    )
    return len(sentences), len(blocks)


def build_model(settings: dict[str, Any]) -> AutoModel:
    funasr_config = settings["funasr"]
    return AutoModel(
        model=funasr_config["model"],
        vad_model=funasr_config["vad_model"],
        vad_kwargs={"max_single_segment_time": funasr_config["max_single_segment_time"]},
        punc_model=funasr_config["punc_model"],
        spk_model=funasr_config["spk_model"],
        device=funasr_config["device"],
    )


def process_one_audio(
    audio_path: Path,
    model: AutoModel,
    settings: dict[str, Any],
    project_dir: Path,
) -> dict[str, str | None]:
    paths = settings["paths"]
    funasr_config = settings["funasr"]
    postprocess_config = settings["postprocess"]
    llm_config = settings["llm"]
    output_config = settings["output"]

    base_output_dir = resolve_project_path(project_dir, paths["output_dir"])
    prompt_dir = resolve_project_path(project_dir, paths["prompt_dir"])
    if base_output_dir is None or prompt_dir is None:
        raise SystemExit("paths.output_dir 和 paths.prompt_dir 不能为空")

    audio_output_dir = build_audio_output_dir(base_output_dir, audio_path, output_config["per_audio_subdir"])
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    json_path = audio_output_dir / f"{stem}.json"
    cleaned_path = audio_output_dir / f"{stem}_cleaned.txt"
    polished_path = audio_output_dir / f"{stem}{llm_config['polished_suffix']}.txt"

    print(f"正在转写音频：{audio_path}", flush=True)
    result = model.generate(
        input=str(audio_path),
        batch_size_s=funasr_config["batch_size_s"],
        batch_size_threshold_s=funasr_config["batch_size_threshold_s"],
    )
    dump_json_output(result, json_path, output_config["overwrite"])
    print(f"FunASR JSON 输出：{json_path}", flush=True)

    sentence_count, block_count = write_cleaned_transcript(
        json_path=json_path,
        cleaned_path=cleaned_path,
        max_gap_ms=postprocess_config["max_gap_ms"],
        max_chars=postprocess_config["max_chars"],
        speaker_prefix=postprocess_config["speaker_prefix"],
        keep_time=postprocess_config["keep_time"],
        prompt_dir=prompt_dir,
        overwrite=output_config["overwrite"],
    )
    print(f"读取句段：{sentence_count}", flush=True)
    print(f"合并段落：{block_count}", flush=True)
    print(f"cleaned 输出：{cleaned_path}", flush=True)

    if llm_config["skip_polish"]:
        return {
            "audio": str(audio_path),
            "json": str(json_path),
            "cleaned": str(cleaned_path),
            "polished": None,
        }

    env_path = resolve_project_path(project_dir, paths["env_file"])
    if env_path is None:
        raise SystemExit("paths.env_file 不能为空")
    load_env_file(env_path)

    api_key = os.getenv(llm_config["api_key_env"]) or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(llm_config["base_url_env"])
    llm_model = llm_config["model"] or os.getenv(llm_config["model_name_env"]) or "qwen3-max"
    if not api_key:
        raise SystemExit(f"未找到 API key，请在 {env_path} 中配置 {llm_config['api_key_env']} 或 DASHSCOPE_API_KEY")
    if not base_url:
        raise SystemExit(f"未找到 BASE_URL，请在 {env_path} 中配置 {llm_config['base_url_env']}")
    if polished_path.exists() and not output_config["overwrite"]:
        raise FileExistsError(f"输出文件已存在且 overwrite=false：{polished_path}")

    prompt_template = load_prompt_template(prompt_dir)
    chunk_count = write_polished_transcript(
        source_path=cleaned_path,
        polished_path=polished_path,
        chunk_size=llm_config["chunk_size"],
        base_url=base_url,
        api_key=api_key,
        model=llm_model,
        enable_thinking=llm_config["enable_thinking"],
        max_workers=llm_config["max_workers"],
        max_retries=llm_config["max_retries"],
        prompt_template=prompt_template,
    )
    print(f"LLM provider：{llm_config['provider']}", flush=True)
    print(f"LLM model：{llm_model}", flush=True)
    print(f"润色分块数量：{chunk_count}", flush=True)
    print(f"polished 输出：{polished_path}", flush=True)

    return {
        "audio": str(audio_path),
        "json": str(json_path),
        "cleaned": str(cleaned_path),
        "polished": str(polished_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    settings_path = Path(args.settings).expanduser()
    if not settings_path.is_absolute():
        settings_path = PROJECT_DIR / settings_path
    settings = load_settings(settings_path)
    audio_files = discover_audio_files(settings, PROJECT_DIR)

    print(f"读取配置：{settings_path}", flush=True)
    print(f"待处理音频数量：{len(audio_files)}", flush=True)
    print("正在加载 FunASR 模型...", flush=True)
    model = build_model(settings)

    results = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"\n===== 处理音频 {index}/{len(audio_files)}：{audio_path.name} =====", flush=True)
        results.append(process_one_audio(audio_path, model, settings, PROJECT_DIR))

    print("\n===== 处理完成 =====", flush=True)
    for result in results:
        print(f"音频：{result['audio']}", flush=True)
        print(f"JSON：{result['json']}", flush=True)
        print(f"cleaned：{result['cleaned']}", flush=True)
        if result["polished"]:
            print(f"polished：{result['polished']}", flush=True)


if __name__ == "__main__":
    main()
