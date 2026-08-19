import json
import tempfile
import unittest
from pathlib import Path

from funasr_e2e.pipeline.service import (
    build_speaker_summaries,
    generate_evidence_stage,
    run_funasr_stage,
)


class FakeFunASRModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [{
            "sentence_info": [
                {"text": "第一句", "start": 0, "end": 800, "spk": 0},
                {"text": "第二句", "start": 900, "end": 1500, "spk": 1},
                {"text": "第三句内容更长", "start": 1600, "end": 2500, "spk": 0},
            ]
        }]


class PipelineServiceTest(unittest.TestCase):
    def test_funasr_and_evidence_stages_share_validated_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_dir = root / "prompt"
            prompt_dir.mkdir()
            (prompt_dir / "hotwords.txt").write_text("术语\n# 注释\n术语\n", encoding="utf-8")
            model = FakeFunASRModel()
            events = []
            raw_json_path = root / "raw.json"
            result = run_funasr_stage(
                audio_path=root / "audio.wav",
                raw_json_path=raw_json_path,
                model=model,
                funasr_config={
                    "batch_size_s": 300,
                    "batch_size_threshold_s": 60,
                    "preset_spk_num": 2,
                },
                prompt_dir=prompt_dir,
                progress_callback=events.append,
            )
            evidence_path = root / "evidence.txt"
            evidence_sentences = generate_evidence_stage(
                raw_json_path=raw_json_path,
                evidence_path=evidence_path,
                speaker_prefix="说话人",
                keep_time=True,
                progress_callback=events.append,
            )

            self.assertEqual(len(result.sentences), 3)
            self.assertEqual([sentence.text for sentence in evidence_sentences], ["第一句", "第二句", "第三句内容更长"])
            self.assertEqual(model.calls, [{
                "input": str(root / "audio.wav"),
                "batch_size_s": 300,
                "batch_size_threshold_s": 60,
                "preset_spk_num": 2,
                "hotword": "术语",
            }])
            self.assertEqual(json.loads(raw_json_path.read_text(encoding="utf-8"))[0]["sentence_info"][0]["text"], "第一句")
            self.assertIn("说话人0", evidence_path.read_text(encoding="utf-8"))
            self.assertEqual([(event.stage, event.event) for event in events], [
                ("funasr", "started"),
                ("funasr", "completed"),
                ("evidence", "started"),
                ("evidence", "completed"),
            ])

    def test_speaker_summaries_remain_anonymous_and_deterministic(self) -> None:
        model = FakeFunASRModel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_dir = root / "prompt"
            prompt_dir.mkdir()
            (prompt_dir / "hotwords.txt").write_text("", encoding="utf-8")
            result = run_funasr_stage(
                audio_path=root / "audio.wav",
                raw_json_path=root / "raw.json",
                model=model,
                funasr_config={"batch_size_s": 1, "batch_size_threshold_s": 1, "preset_spk_num": None},
                prompt_dir=prompt_dir,
            )

        summaries = build_speaker_summaries(result.sentences)
        self.assertEqual([summary.anonymous_label for summary in summaries], ["SPEAKER_0", "SPEAKER_1"])
        self.assertEqual(summaries[0].occurrence_count, 2)
        self.assertEqual([excerpt.text for excerpt in summaries[0].excerpts], ["第一句", "第三句内容更长"])


if __name__ == "__main__":
    unittest.main()
