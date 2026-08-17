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
    ReviewedSpan,
    Sentence,
    load_cleaning_config,
    load_env_file,
    load_sentences,
    merge_sentences,
    render_blocks,
    render_evidence_transcript,
    review_sentences,
    reviewed_spans_from_audit,
    write_final_transcript,
)
from review_funasr_speakers import run_speaker_review, sha256_file, write_speaker_review_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FunASR_e2e 音频转写、说话人识别和大模型润色端到端流水线。")
    parser.add_argument("--settings", default="settings.yaml", help="配置文件路径，默认 settings.yaml")
    parser.add_argument("--reuse-json", action="store_true", help="复用既有 FunASR JSON，只重跑后处理和 LLM 阶段")
    parser.add_argument("--skip-polish", action="store_true", help="本次仅生成 evidence、reviewed 和 cleaned，不生成最终阅读版")
    parser.add_argument("--polish-only", action="store_true", help="仅从已通过复核的 reviewed spans 生成最终阅读版")
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
            "preset_spk_num": None,
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
            "final_suffix": "_final",
            "final_audit_suffix": "_final_audit",
        },
        "speaker_review": {
            "enabled": True,
            "enable_thinking": False,
            "model": None,
            "context_size": 4,
            "max_risk_core_sentences": 12,
            "max_boundary_candidates": 16,
            "max_workers": 8,
            "max_retries": 3,
            "request_timeout_s": 90,
            "auto_apply_confidence": 0.90,
            "allowed_speakers": None,
            "unknown_label": "unknown",
            "overlap_label": "overlap",
            "failure_policy": "keep_original",
            "api_key_env": "API_KEY",
            "base_url_env": "BASE_URL",
            "model_name_env": "MODEL_NAME",
            "review_json_suffix": "_speaker_review",
            "reviewed_suffix": "_reviewed",
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


def get_preset_spk_num(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit("funasr.preset_spk_num 必须为 null 或正整数")
    return value


def load_hotwords(prompt_dir: Path) -> list[str]:
    hotword_path = prompt_dir / "hotwords.txt"
    if not hotword_path.exists():
        raise FileNotFoundError(f"热词文件不存在：{hotword_path}")

    hotwords = []
    seen = set()
    for line in hotword_path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if not word or word.startswith("#") or word in seen:
            continue
        seen.add(word)
        hotwords.append(word)
    return hotwords


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


def write_evidence_transcript(
    sentences: list[Sentence],
    evidence_path: Path,
    speaker_prefix: str,
    keep_time: bool,
    overwrite: bool,
) -> None:
    if evidence_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在且 overwrite=false：{evidence_path}")
    evidence_path.write_text(
        render_evidence_transcript(sentences, speaker_prefix, keep_time),
        encoding="utf-8",
    )


def write_cleaned_transcript(
    items: list[Sentence | ReviewedSpan],
    cleaned_path: Path,
    max_gap_ms: int,
    max_chars: int,
    speaker_prefix: str,
    keep_time: bool,
    prompt_dir: Path,
    overwrite: bool,
) -> int:
    if cleaned_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在且 overwrite=false：{cleaned_path}")
    blocks = merge_sentences(items, max_gap_ms, max_chars)
    cleaning_config = load_cleaning_config(prompt_dir)
    cleaned_path.write_text(
        render_blocks(blocks, speaker_prefix, keep_time, cleaned=True, cleaning_config=cleaning_config),
        encoding="utf-8",
    )
    return len(blocks)


def read_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        raise SystemExit(f"环境配置文件不存在：{env_path}")
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_api_credentials(
    project_dir: Path,
    paths: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str, str]:
    if config.get("model") is not None:
        raise SystemExit("llm.model 和 speaker_review.model 必须为 null；模型只能从 .env 的 MODEL_NAME 读取")
    env_path = resolve_project_path(project_dir, paths["env_file"])
    if env_path is None:
        raise SystemExit("paths.env_file 不能为空")
    env_values = read_env_values(env_path)
    load_env_file(env_path)

    api_key_env = config["api_key_env"]
    base_url_env = config["base_url_env"]
    model_name_env = config["model_name_env"]
    api_key = env_values.get(api_key_env) or env_values.get("DASHSCOPE_API_KEY") or os.getenv(api_key_env) or os.getenv("DASHSCOPE_API_KEY")
    base_url = env_values.get(base_url_env) or os.getenv(base_url_env)
    model = env_values.get(model_name_env)
    if not api_key:
        raise SystemExit(f"未找到 API key，请在 {env_path} 中配置 {api_key_env} 或 DASHSCOPE_API_KEY")
    if not base_url:
        raise SystemExit(f"未找到 BASE_URL，请在 {env_path} 中配置 {base_url_env}")
    if not model:
        raise SystemExit(f"未找到模型名，请在 {env_path} 中配置 {model_name_env}")
    return api_key, base_url, model


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
    model: AutoModel | None,
    settings: dict[str, Any],
    project_dir: Path,
    reuse_json: bool,
) -> dict[str, str | None]:
    paths = settings["paths"]
    funasr_config = settings["funasr"]
    postprocess_config = settings["postprocess"]
    llm_config = settings["llm"]
    speaker_review_config = settings["speaker_review"]
    output_config = settings["output"]

    base_output_dir = resolve_project_path(project_dir, paths["output_dir"])
    prompt_dir = resolve_project_path(project_dir, paths["prompt_dir"])
    if base_output_dir is None or prompt_dir is None:
        raise SystemExit("paths.output_dir 和 paths.prompt_dir 不能为空")

    audio_output_dir = build_audio_output_dir(base_output_dir, audio_path, output_config["per_audio_subdir"])
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    json_path = audio_output_dir / f"{stem}.json"
    evidence_path = audio_output_dir / f"{stem}_evidence.txt"
    review_json_path = audio_output_dir / f"{stem}{speaker_review_config['review_json_suffix']}.json"
    reviewed_path = audio_output_dir / f"{stem}{speaker_review_config['reviewed_suffix']}.txt"
    cleaned_path = audio_output_dir / f"{stem}_cleaned.txt"
    final_path = audio_output_dir / f"{stem}{llm_config['final_suffix']}.txt"
    final_audit_path = audio_output_dir / f"{stem}{llm_config['final_audit_suffix']}.json"

    preset_spk_num = get_preset_spk_num(funasr_config.get("preset_spk_num"))
    hotwords = load_hotwords(prompt_dir)
    generate_kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "batch_size_s": funasr_config["batch_size_s"],
        "batch_size_threshold_s": funasr_config["batch_size_threshold_s"],
    }
    if preset_spk_num is not None:
        generate_kwargs["preset_spk_num"] = preset_spk_num
    if hotwords:
        generate_kwargs["hotword"] = " ".join(hotwords)

    if reuse_json:
        if not json_path.exists():
            raise FileNotFoundError(f"--reuse-json 指定的 FunASR JSON 不存在：{json_path}")
        print(f"复用 FunASR JSON：{json_path}", flush=True)
    else:
        if model is None:
            raise RuntimeError("未提供 FunASR 模型")
        print(f"正在转写音频：{audio_path}", flush=True)
        if preset_spk_num is None:
            print("说话人数：自动聚类", flush=True)
        else:
            print(f"说话人数：指定 {preset_spk_num} 人", flush=True)
        print(f"有效热词数量：{len(hotwords)}", flush=True)
        result = model.generate(**generate_kwargs)
        dump_json_output(result, json_path, output_config["overwrite"])
        print(f"FunASR JSON 输出：{json_path}", flush=True)

    sentences = load_sentences(json_path)
    if not sentences:
        raise SystemExit(f"没有从 FunASR JSON 中解析到 sentence_info：{json_path}")
    review_sentences(sentences)
    write_evidence_transcript(
        sentences,
        evidence_path,
        postprocess_config["speaker_prefix"],
        postprocess_config["keep_time"],
        output_config["overwrite"],
    )
    print(f"读取句段：{len(sentences)}", flush=True)
    print(f"证据稿输出：{evidence_path}", flush=True)

    review_result = None
    transcript_items: list[Sentence | ReviewedSpan] = sentences
    review_api_key = None
    review_base_url = None
    review_model = None
    if speaker_review_config["enabled"]:
        if review_json_path.exists() and not output_config["overwrite"]:
            raise FileExistsError(f"输出文件已存在且 overwrite=false：{review_json_path}")
        if reviewed_path.exists() and not output_config["overwrite"]:
            raise FileExistsError(f"输出文件已存在且 overwrite=false：{reviewed_path}")
        review_api_key, review_base_url, review_model = load_api_credentials(project_dir, paths, speaker_review_config)
        print("正在进行语义说话人复核...", flush=True)
        review_result = run_speaker_review(
            json_path=json_path,
            sentences=sentences,
            prompt_dir=prompt_dir,
            config=speaker_review_config,
            base_url=review_base_url,
            api_key=review_api_key,
            default_model=review_model,
            speaker_prefix=postprocess_config["speaker_prefix"],
            keep_time=postprocess_config["keep_time"],
        )
        write_speaker_review_outputs(review_result, review_json_path, reviewed_path)
        transcript_items = review_result.spans
        integrity = review_result.audit["integrity"]
        print(f"speaker review 输出：{review_json_path}", flush=True)
        print(f"reviewed 输出：{reviewed_path}", flush=True)
        print(
            "复核统计："
            f"KEEP={integrity['operation_counts']['KEEP']}，"
            f"REASSIGN={integrity['operation_counts']['REASSIGN']}，"
            f"SPLIT={integrity['operation_counts']['SPLIT']}，"
            f"unknown={integrity['unknown_count']}，"
            f"overlap={integrity['overlap_count']}，"
            f"待回听={integrity['review_queue_count']}，"
            f"请求数={integrity.get('total_request_count', 0)}，"
            f"提示词字节数={integrity.get('total_prompt_bytes', 0)}",
            flush=True,
        )

    block_count = write_cleaned_transcript(
        transcript_items,
        cleaned_path,
        postprocess_config["max_gap_ms"],
        postprocess_config["max_chars"],
        postprocess_config["speaker_prefix"],
        postprocess_config["keep_time"],
        prompt_dir,
        output_config["overwrite"],
    )
    print(f"合并段落：{block_count}", flush=True)
    print(f"cleaned 输出：{cleaned_path}", flush=True)

    if llm_config["skip_polish"]:
        return {
            "audio": str(audio_path),
            "json": str(json_path),
            "evidence": str(evidence_path),
            "speaker_review": str(review_json_path) if review_result else None,
            "reviewed": str(reviewed_path) if review_result else None,
            "cleaned": str(cleaned_path),
            "final": None,
            "final_audit": None,
        }
    if review_result is None:
        raise RuntimeError("生成最终阅读版需要启用 speaker_review")
    if (final_path.exists() or final_audit_path.exists()) and not output_config["overwrite"]:
        raise FileExistsError(f"最终阅读版输出已存在且 overwrite=false：{final_path}")

    api_key, base_url, llm_model = load_api_credentials(project_dir, paths, llm_config)
    final_audit = write_final_transcript(
        spans=review_result.spans,
        final_path=final_path,
        final_audit_path=final_audit_path,
        max_gap_ms=postprocess_config["max_gap_ms"],
        max_chars=postprocess_config["max_chars"],
        speaker_prefix=postprocess_config["speaker_prefix"],
        keep_time=postprocess_config["keep_time"],
        prompt_dir=prompt_dir,
        base_url=base_url,
        api_key=api_key,
        model=llm_model,
        enable_thinking=llm_config["enable_thinking"],
        chunk_size=llm_config["chunk_size"],
        max_retries=llm_config["max_retries"],
        source_json_sha256=sha256_file(json_path),
    )
    print(f"LLM provider：{llm_config['provider']}", flush=True)
    print(f"LLM model：{llm_model}", flush=True)
    print(f"最终阅读分块数量：{len(final_audit['chunks'])}", flush=True)
    print(f"最终阅读版：{final_path}", flush=True)

    return {
        "audio": str(audio_path),
        "json": str(json_path),
        "evidence": str(evidence_path),
        "speaker_review": str(review_json_path),
        "reviewed": str(reviewed_path),
        "cleaned": str(cleaned_path),
        "final": str(final_path),
        "final_audit": str(final_audit_path),
    }


def polish_one_audio(audio_path: Path, settings: dict[str, Any], project_dir: Path) -> dict[str, str | None]:
    paths = settings["paths"]
    llm_config = settings["llm"]
    speaker_review_config = settings["speaker_review"]
    output_config = settings["output"]
    base_output_dir = resolve_project_path(project_dir, paths["output_dir"])
    prompt_dir = resolve_project_path(project_dir, paths["prompt_dir"])
    if base_output_dir is None or prompt_dir is None:
        raise SystemExit("paths.output_dir 和 paths.prompt_dir 不能为空")
    audio_output_dir = build_audio_output_dir(base_output_dir, audio_path, output_config["per_audio_subdir"])
    stem = audio_path.stem
    json_path = audio_output_dir / f"{stem}.json"
    evidence_path = audio_output_dir / f"{stem}_evidence.txt"
    review_json_path = audio_output_dir / f"{stem}{speaker_review_config['review_json_suffix']}.json"
    reviewed_path = audio_output_dir / f"{stem}{speaker_review_config['reviewed_suffix']}.txt"
    cleaned_path = audio_output_dir / f"{stem}_cleaned.txt"
    final_path = audio_output_dir / f"{stem}{llm_config['final_suffix']}.txt"
    final_audit_path = audio_output_dir / f"{stem}{llm_config['final_audit_suffix']}.json"
    if not json_path.exists() or not review_json_path.exists() or not reviewed_path.exists():
        raise FileNotFoundError("--polish-only 需要已有 JSON、speaker_review 和 reviewed 输出")
    audit = json.loads(review_json_path.read_text(encoding="utf-8"))
    integrity = audit.get("integrity")
    required_integrity = (
        "source_hash_verified",
        "per_source_reconstruction_passed",
        "global_reconstruction_passed",
        "order_passed",
        "coverage_passed",
        "allowed_speaker_passed",
        "unknown_from_explicit_review_passed",
    )
    if audit.get("schema_version") != 3 or not isinstance(integrity, dict) or not all(integrity.get(key) is True for key in required_integrity):
        raise RuntimeError("--polish-only 只接受 schema v3 且完整性校验全部通过的 speaker review 输出")
    if audit.get("run", {}).get("source_json_sha256") != sha256_file(json_path):
        raise RuntimeError("--polish-only 检测到原始 FunASR JSON 已变化")
    if (final_path.exists() or final_audit_path.exists()) and not output_config["overwrite"]:
        raise FileExistsError(f"最终阅读版输出已存在且 overwrite=false：{final_path}")
    api_key, base_url, llm_model = load_api_credentials(project_dir, paths, llm_config)
    final_audit = write_final_transcript(
        spans=reviewed_spans_from_audit(audit),
        final_path=final_path,
        final_audit_path=final_audit_path,
        max_gap_ms=settings["postprocess"]["max_gap_ms"],
        max_chars=settings["postprocess"]["max_chars"],
        speaker_prefix=settings["postprocess"]["speaker_prefix"],
        keep_time=settings["postprocess"]["keep_time"],
        prompt_dir=prompt_dir,
        base_url=base_url,
        api_key=api_key,
        model=llm_model,
        enable_thinking=llm_config["enable_thinking"],
        chunk_size=llm_config["chunk_size"],
        max_retries=llm_config["max_retries"],
        source_json_sha256=sha256_file(json_path),
    )
    print(f"LLM provider：{llm_config['provider']}", flush=True)
    print(f"LLM model：{llm_model}", flush=True)
    print(f"最终阅读分块数量：{len(final_audit['chunks'])}", flush=True)
    print(f"最终阅读版：{final_path}", flush=True)
    return {
        "audio": str(audio_path),
        "json": str(json_path),
        "evidence": str(evidence_path),
        "speaker_review": str(review_json_path),
        "reviewed": str(reviewed_path),
        "cleaned": str(cleaned_path) if cleaned_path.exists() else None,
        "final": str(final_path),
        "final_audit": str(final_audit_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    settings_path = Path(args.settings).expanduser()
    if not settings_path.is_absolute():
        settings_path = PROJECT_DIR / settings_path
    settings = load_settings(settings_path)
    if args.polish_only and args.reuse_json:
        raise SystemExit("--polish-only 不可与 --reuse-json 同时使用")
    if args.skip_polish:
        settings["llm"]["skip_polish"] = True
    audio_files = discover_audio_files(settings, PROJECT_DIR)

    print(f"读取配置：{settings_path}", flush=True)
    print(f"待处理音频数量：{len(audio_files)}", flush=True)
    if args.polish_only:
        results = []
        for index, audio_path in enumerate(audio_files, start=1):
            print(f"\n===== 仅生成最终阅读版 {index}/{len(audio_files)}：{audio_path.name} =====", flush=True)
            results.append(polish_one_audio(audio_path, settings, PROJECT_DIR))
    else:
        model = None
        if not args.reuse_json:
            print("正在加载 FunASR 模型...", flush=True)
            model = build_model(settings)
        results = []
        for index, audio_path in enumerate(audio_files, start=1):
            print(f"\n===== 处理音频 {index}/{len(audio_files)}：{audio_path.name} =====", flush=True)
            results.append(process_one_audio(audio_path, model, settings, PROJECT_DIR, args.reuse_json))

    print("\n===== 处理完成 =====", flush=True)
    for result in results:
        print(f"音频：{result['audio']}", flush=True)
        print(f"JSON：{result['json']}", flush=True)
        print(f"evidence：{result['evidence']}", flush=True)
        if result["speaker_review"]:
            print(f"speaker review：{result['speaker_review']}", flush=True)
            print(f"reviewed：{result['reviewed']}", flush=True)
        if result["cleaned"]:
            print(f"cleaned：{result['cleaned']}", flush=True)
        if result["final"]:
            print(f"最终阅读版：{result['final']}", flush=True)
            print(f"final audit：{result['final_audit']}", flush=True)


if __name__ == "__main__":
    main()
