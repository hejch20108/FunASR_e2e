import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from postprocess_funasr_transcript import (
    ReadingUnit,
    ReviewedSpan,
    TextCleaningConfig,
    build_reading_units,
    clean_fallback_text,
    extract_fact_locks,
    prepare_reading_units,
    validate_final_response,
    write_final_transcript,
)


class FinalTranscriptTest(unittest.TestCase):
    def span(self, text, index, spk=0, start=None, end=None, review_required=False, flags=None):
        start = index * 1000 if start is None else start
        end = start + 800 if end is None else end
        return ReviewedSpan(
            source_id=f"r000.s{index:06d}",
            source_order=index,
            char_start=0,
            char_end=len(text),
            start=start,
            end=end,
            spk=spk,
            original_spk=spk,
            text=text,
            operation="KEEP",
            confidence=1.0,
            reason_codes=["SOURCE_SPK_PRIOR"],
            review_required=review_required,
            review_flags=flags or [],
            time_method="source_range",
        )

    def cleaning_config(self):
        return TextCleaningConfig(
            repeated_words=[],
            drop_words={"嗯", "啊", "呃"},
            filler_words=["嗯", "啊", "呃"],
        )

    def test_extract_fact_locks_ignores_ordinary_one_after_compensation(self):
        self.assertEqual(extract_fact_locks("赔偿金的，这一个算法需要重新核算。"), [])

    def test_build_reading_units_merges_same_speaker_without_crossing_turn(self):
        spans = [
            self.span("我想", 0, 0, 0, 300),
            self.span("了解补偿。", 1, 0, 350, 900),
            self.span("可以。", 2, 1, 950, 1200),
        ]
        units = build_reading_units(spans, max_gap_ms=1000, max_chars=100)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].text, "我想了解补偿。")
        self.assertEqual([item["source_id"] for item in units[0].source_spans], [
            "r000.s000000", "r000.s000001",
        ])
        self.assertEqual(units[1].spk, 1)

    def test_prepare_reading_units_drops_only_pure_filler(self):
        spans = [self.span("嗯", 0), self.span("对", 1, 1)]
        units, cleanup = prepare_reading_units(spans, 1000, 100, self.cleaning_config())
        self.assertEqual([unit.text for unit in units], ["对"])
        self.assertEqual(cleanup[0]["reason_code"], "FILLER_ONLY")

    def test_clean_fallback_text_removes_only_safe_filler_and_repetition(self):
        config = self.cleaning_config()
        config.repeated_words = ["就是"]
        self.assertEqual(
            clean_fallback_text(
                "嗯，就是就是公司不同意赔偿一万元。嗯",
                config,
            ),
            "就是公司不同意赔偿一万元。",
        )
        self.assertEqual(
            clean_fallback_text("我我不同意赔偿一万元。", config),
            "我不同意赔偿一万元。",
        )

    def test_validate_final_response_accepts_same_speaker_merge_with_fact_locks(self):
        units = [
            ReadingUnit("u0001", 0, 500, 0, ["公司补偿一万元"], [], [], False),
            ReadingUnit("u0002", 550, 900, 0, ["我可以接受"], [], [], False),
        ]
        paragraphs, dropped, validation = validate_final_response(
            {
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001", "u0002"],
                    "text": "公司补偿一万元，我可以接受。",
                }],
                "dropped_units": [],
            },
            units,
            "说话人",
            self.cleaning_config(),
        )
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(dropped, [])
        self.assertEqual(validation["warnings"], [])

    def test_validate_final_response_allows_punctuation_joined_facts(self):
        units = [
            ReadingUnit("u0001", 0, 500, 0, ["公司正在降低员，工成本。"], [], [], False),
            ReadingUnit("u0002", 550, 900, 0, ["我。们接？受。"], [], [], False),
        ]
        paragraphs, _, validation = validate_final_response(
            {
                "paragraphs": [
                    {
                        "speaker": "说话人0",
                        "source_unit_ids": ["u0001", "u0002"],
                        "text": "公司正在降低员工成本。我们接受。",
                    },
                ],
                "dropped_units": [],
            },
            units,
            "说话人",
            self.cleaning_config(),
        )
        self.assertEqual(len(paragraphs), 1)
        self.assertIn(
            {"paragraph_index": 0, "code": "PUNCTUATION_JOIN_USED", "kind": "subject", "value": "员工"},
            validation["warnings"],
        )
        self.assertIn(
            {"paragraph_index": 0, "code": "PUNCTUATION_JOIN_USED", "kind": "acceptance", "value": "接受"},
            validation["warnings"],
        )

    def test_validate_final_response_rejects_synonym_and_new_amount(self):
        units = [ReadingUnit("u0001", 0, 500, 0, ["公司接受补偿方案。"], [], [], False)]
        with self.assertRaisesRegex(ValueError, "关键事实"):
            validate_final_response(
                {
                    "paragraphs": [{
                        "speaker": "说话人0",
                        "source_unit_ids": ["u0001"],
                        "text": "公司认可补偿方案。",
                    }],
                    "dropped_units": [],
                },
                units,
                "说话人",
                self.cleaning_config(),
            )
        with self.assertRaisesRegex(ValueError, "关键事实"):
            validate_final_response(
                {
                    "paragraphs": [{
                        "speaker": "说话人0",
                        "source_unit_ids": ["u0001"],
                        "text": "公司接受两千块补偿方案。",
                    }],
                    "dropped_units": [],
                },
                units,
                "说话人",
                self.cleaning_config(),
            )

    def test_validate_final_response_allows_relation_reordering(self):
        units = [ReadingUnit("u0001", 0, 500, 0, ["公司不同意补偿一万元。"], [], [], False)]
        paragraphs, _, _ = validate_final_response(
            {
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001"],
                    "text": "公司对一万元补偿不同意。",
                }],
                "dropped_units": [],
            },
            units,
            "说话人",
            self.cleaning_config(),
        )
        self.assertEqual(paragraphs[0]["text"], "公司对一万元补偿不同意。")

    def test_validate_final_response_allows_merged_relation_superset(self):
        units = [
            ReadingUnit("u0001", 0, 500, 0, ["公司不同意。"], [], [], False),
            ReadingUnit("u0002", 550, 900, 0, ["补偿一万元。"], [], [], False),
        ]
        paragraphs, _, _ = validate_final_response(
            {
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001", "u0002"],
                    "text": "公司不同意一万元补偿。",
                }],
                "dropped_units": [],
            },
            units,
            "说话人",
            self.cleaning_config(),
        )
        self.assertEqual(len(paragraphs), 1)

    def test_validate_final_response_rejects_changed_amount_or_negation(self):
        units = [
            ReadingUnit("u0001", 0, 500, 0, ["公司不同意补偿一万元"], [], [], False),
        ]
        with self.assertRaisesRegex(ValueError, "关键事实"):
            validate_final_response(
                {
                    "paragraphs": [{
                        "speaker": "说话人0",
                        "source_unit_ids": ["u0001"],
                        "text": "公司同意补偿一万二。",
                    }],
                    "dropped_units": [],
                },
                units,
                "说话人",
                self.cleaning_config(),
            )

    def test_validate_final_response_rejects_changed_fact_relationship_with_same_locks(self):
        units = [
            ReadingUnit("u0001", 0, 500, 0, ["公司不同意给我补偿一万元。"], [], [], False),
        ]
        with self.assertRaisesRegex(ValueError, "关键事实关系"):
            validate_final_response(
                {
                    "paragraphs": [{
                        "speaker": "说话人0",
                        "source_unit_ids": ["u0001"],
                        "text": "公司不同意，我补偿一万元。",
                    }],
                    "dropped_units": [],
                },
                units,
                "说话人",
                self.cleaning_config(),
            )

    def test_write_final_transcript_accepts_soft_warning_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = json.dumps({
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001"],
                    "text": "公司正在降低员工成本。",
                }],
                "dropped_units": [],
            }, ensure_ascii=False)
            with patch(
                "postprocess_funasr_transcript.call_openai_compatible_chat",
                return_value=response,
            ) as call:
                audit = write_final_transcript(
                    spans=[self.span("公司正在降低员，工成本。", 0)],
                    final_path=root / "final.txt",
                    final_audit_path=root / "final_audit.json",
                    max_gap_ms=1000,
                    max_chars=100,
                    speaker_prefix="说话人",
                    keep_time=True,
                    prompt_dir=PROJECT_DIR / "prompt",
                    base_url="https://example.invalid",
                    api_key="secret",
                    model="test-model",
                    enable_thinking=False,
                    chunk_size=1,
                    max_retries=3,
                    source_json_sha256="source-hash",
                )
            self.assertEqual(call.call_count, 1)
            self.assertEqual(audit["chunks"][0]["status"], "ok_with_warnings")
            self.assertEqual(audit["integrity"]["warning_chunk_count"], 1)

    def test_write_final_transcript_retries_with_validation_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_response = json.dumps({
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001"],
                    "text": "公司认可补偿方案。",
                }],
                "dropped_units": [],
            }, ensure_ascii=False)
            valid_response = json.dumps({
                "paragraphs": [{
                    "speaker": "说话人0",
                    "source_unit_ids": ["u0001"],
                    "text": "公司接受补偿方案。",
                }],
                "dropped_units": [],
            }, ensure_ascii=False)
            with patch(
                "postprocess_funasr_transcript.call_openai_compatible_chat",
                side_effect=[invalid_response, valid_response],
            ) as call:
                audit = write_final_transcript(
                    spans=[self.span("公司接受补偿方案。", 0)],
                    final_path=root / "final.txt",
                    final_audit_path=root / "final_audit.json",
                    max_gap_ms=1000,
                    max_chars=100,
                    speaker_prefix="说话人",
                    keep_time=True,
                    prompt_dir=PROJECT_DIR / "prompt",
                    base_url="https://example.invalid",
                    api_key="secret",
                    model="test-model",
                    enable_thinking=False,
                    chunk_size=1,
                    max_retries=3,
                    source_json_sha256="source-hash",
                )
            chunk = audit["chunks"][0]
            self.assertEqual(call.call_count, 2)
            self.assertEqual(chunk["status"], "ok")
            self.assertEqual(chunk["attempts"], 2)
            self.assertEqual(
                chunk["request_attempts"][1]["retrying_validation_code"],
                "PROTECTED_FACT_MISSING",
            )
            self.assertIn(
                "PROTECTED_FACT_MISSING",
                call.call_args_list[1].kwargs["prompt"],
            )

    def test_write_final_transcript_uses_deterministic_fallback_per_failed_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "final.txt"
            audit_path = root / "final_audit.json"
            spans = [self.span("嗯，我想了解补偿。", 0)]
            with patch(
                "postprocess_funasr_transcript.call_openai_compatible_chat",
                side_effect=RuntimeError("模型不可用"),
            ):
                audit = write_final_transcript(
                    spans=spans,
                    final_path=final_path,
                    final_audit_path=audit_path,
                    max_gap_ms=1000,
                    max_chars=100,
                    speaker_prefix="说话人",
                    keep_time=True,
                    prompt_dir=PROJECT_DIR / "prompt",
                    base_url="https://example.invalid",
                    api_key="secret",
                    model="test-model",
                    enable_thinking=False,
                    chunk_size=1,
                    max_retries=1,
                    source_json_sha256="source-hash",
                )
                self.assertTrue(final_path.exists())
                self.assertTrue(audit_path.exists())
                self.assertIn("我想了解补偿。", final_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["chunks"][0]["status"], "fallback")


if __name__ == "__main__":
    unittest.main()
