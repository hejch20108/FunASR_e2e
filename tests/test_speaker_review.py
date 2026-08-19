import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from postprocess_funasr_transcript import Sentence, load_sentences, review_sentences
from review_funasr_speakers import (
    BoundaryCandidate,
    RiskSignal,
    build_boundary_candidates,
    build_full_baseline,
    build_risk_segment_input,
    build_risk_segments,
    build_segment_candidates,
    collect_risk_signals,
    apply_unreviewed_high_impact_keep_guard,
    component_partition_passed,
    conservative_keep_guard_signals,
    call_json_with_retries,
    decision_to_spans,
    fallback_decision,
    normalize_review_config,
    normalize_segment_decision,
    run_risk_segment_pass,
    run_speaker_review,
    sha256_file,
    validate_decision_response,
    validate_full_review_response,
    validate_risk_segment_response,
    validate_spans,
)
from funasr_e2e.pipeline.control import PipelineCancelled
from run_funasr_full_pipeline import default_settings, load_api_credentials, polish_one_audio


class SpeakerReviewTest(unittest.TestCase):
    def make_sentence(self, text="甲，乙N+1", index=0, spk=0, start=100, end=1000, flags=None, timestamps=None):
        return Sentence(
            index=index,
            source_id=f"r000.s{index:06d}",
            result_index=0,
            sentence_index=index,
            start=start,
            end=end,
            spk=spk,
            text=text,
            timestamps=timestamps or [],
            review_flags=flags or [],
        )

    def candidates_for(self, sentence):
        return {
            sentence.source_id: {
                f"{sentence.source_id}.b{offset:03d}": BoundaryCandidate(
                    boundary_id=f"{sentence.source_id}.b{offset:03d}",
                    source_id=sentence.source_id,
                    char_offset=offset,
                    estimated_time_ms=None,
                    time_method="unavailable",
                )
                for offset in range(len(sentence.text) + 1)
            }
        }

    def decision(self, sentence, operation="KEEP", **extra):
        value = {
            "source_id": sentence.source_id,
            "operation": operation,
            "confidence": 0.95,
            "reason_codes": ["ROLE_CONSISTENCY"],
        }
        value.update(extra)
        return {"decisions": [value]}

    def baseline(self, sentence, speaker=None):
        return {
            "source_id": sentence.source_id,
            "operation": "KEEP",
            "target_speaker": speaker or f"SPEAKER_{sentence.spk}",
            "parts": [],
            "confidence": 1.0,
            "reason_codes": ["SOURCE_SPK_PRIOR"],
            "review_required": False,
            "source": "implicit_keep",
        }

    def validate(self, payload, sentence):
        return validate_decision_response(
            payload,
            [sentence],
            self.candidates_for(sentence),
            {"SPEAKER_0", "SPEAKER_1"},
            "unknown",
            "overlap",
        )[sentence.source_id]

    def full_sentences(self):
        return [
            self.make_sentence("甲", 0, 0),
            self.make_sentence("乙", 1, 0),
            self.make_sentence("丙", 2, 1),
            self.make_sentence("丁", 3, 1),
        ]

    def full_payload(self, **extra):
        payload = {
            "speaker_registry": {
                "speakers": [
                    {
                        "speaker_id": "SPEAKER_0",
                        "role_summary": "匿名角色一",
                        "evidence_source_ids": ["r000.s000000", "r000.s000001"],
                        "confidence": 0.95,
                        "reason_codes": ["ROLE_CONSISTENCY"],
                    },
                    {
                        "speaker_id": "SPEAKER_1",
                        "role_summary": "匿名角色二",
                        "evidence_source_ids": ["r000.s000002", "r000.s000003"],
                        "confidence": 0.95,
                        "reason_codes": ["ROLE_CONSISTENCY"],
                    },
                ]
            },
            "overrides": [],
            "risk_items": [],
        }
        payload.update(extra)
        return payload

    def test_load_sentences_preserves_json_order_and_text(self):
        source = [{"sentence_info": [
            {"text": "  第一段  ", "start": 500, "end": 600, "spk": 1, "timestamp": [[500, 600]]},
            {"text": "第二段", "start": 100, "end": 200, "spk": 0},
        ]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            sentences = load_sentences(path)
        self.assertEqual([sentence.source_id for sentence in sentences], ["r000.s000000", "r000.s000001"])
        self.assertEqual(sentences[0].text, "  第一段  ")
        self.assertEqual(sentences[1].start, 100)

    def test_keep_and_reassign_preserve_exact_text(self):
        sentence = self.make_sentence()
        keep = self.validate(self.decision(sentence), sentence)
        keep_spans = decision_to_spans(sentence, keep)
        self.assertEqual(keep_spans[0].spk, "SPEAKER_0")
        self.assertEqual(keep_spans[0].text, sentence.text)
        reassigned = self.validate(self.decision(sentence, "REASSIGN", target_speaker="SPEAKER_1"), sentence)
        self.assertEqual(decision_to_spans(sentence, reassigned)[0].spk, "SPEAKER_1")

    def test_split_reconstructs_original_text(self):
        sentence = self.make_sentence()
        split = self.validate(self.decision(sentence, "SPLIT", parts=[
            {"start_boundary_id": f"{sentence.source_id}.b000", "end_boundary_id": f"{sentence.source_id}.b002", "speaker": "SPEAKER_0"},
            {"start_boundary_id": f"{sentence.source_id}.b002", "end_boundary_id": f"{sentence.source_id}.b{len(sentence.text):03d}", "speaker": "SPEAKER_1"},
        ]), sentence)
        spans = decision_to_spans(sentence, split)
        self.assertEqual("".join(span.text for span in spans), "甲，乙N+1")
        integrity = validate_spans([sentence], spans, {"SPEAKER_0", "SPEAKER_1", "unknown", "overlap"})
        self.assertTrue(integrity["global_reconstruction_passed"])

    def test_invalid_split_is_rejected(self):
        sentence = self.make_sentence()
        payload = self.decision(sentence, "SPLIT", parts=[
            {"start_boundary_id": f"{sentence.source_id}.b000", "end_boundary_id": f"{sentence.source_id}.b003", "speaker": "SPEAKER_0"},
            {"start_boundary_id": f"{sentence.source_id}.b002", "end_boundary_id": f"{sentence.source_id}.b{len(sentence.text):03d}", "speaker": "SPEAKER_1"},
        ])
        with self.assertRaisesRegex(ValueError, "连续"):
            self.validate(payload, sentence)

    def test_overlap_keeps_one_copy_of_text(self):
        sentence = self.make_sentence()
        decision = self.validate(self.decision(sentence, "OVERLAP"), sentence)
        spans = decision_to_spans(sentence, decision)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].spk, "overlap")
        self.assertEqual(spans[0].text, sentence.text)

    def test_reconstruction_rejects_text_change_and_reordering(self):
        sentence = self.make_sentence("甲乙")
        decision = self.validate(self.decision(sentence), sentence)
        spans = decision_to_spans(sentence, decision)
        broken = [replace(spans[0], text="甲丙")]
        with self.assertRaisesRegex(RuntimeError, "文本不匹配"):
            validate_spans([sentence], broken, {"SPEAKER_0"})

    def test_hash_detects_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("甲", encoding="utf-8")
            first_hash = sha256_file(path)
            path.write_text("乙", encoding="utf-8")
            self.assertNotEqual(first_hash, sha256_file(path))

    def test_retry_after_invalid_json(self):
        responses = iter(["not-json", '{"valid": true}'])
        requests = []
        with patch(
            "review_funasr_speakers.call_openai_compatible_chat",
            side_effect=lambda **kwargs: requests.append(kwargs) or next(responses),
        ):
            result = call_json_with_retries(
                base_url="https://example.invalid",
                api_key="secret",
                model="test",
                prompt="test",
                max_retries=2,
                timeout_seconds=90,
                enable_thinking=True,
                validator=lambda payload: payload["valid"],
            )
        self.assertTrue(result)
        self.assertTrue(all(request["enable_thinking"] for request in requests))

    def test_cancellation_is_not_converted_to_retry_or_fallback(self):
        def cancel() -> None:
            raise PipelineCancelled()

        with patch("review_funasr_speakers.call_openai_compatible_chat") as call:
            with self.assertRaises(PipelineCancelled):
                call_json_with_retries(
                    base_url="https://example.invalid",
                    api_key="secret",
                    model="test",
                    prompt="test",
                    max_retries=2,
                    timeout_seconds=90,
                    validator=lambda payload: payload,
                    cancel_check=cancel,
                )
        call.assert_not_called()

    def test_run_speaker_review_propagates_cancellation(self):
        sentence = self.make_sentence("甲", 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "source.json"
            json_path.write_text(json.dumps([{"sentence_info": [{
                "text": "甲", "start": 100, "end": 1000, "spk": 0,
            }]}], ensure_ascii=False), encoding="utf-8")
            with patch("review_funasr_speakers.run_full_review", side_effect=PipelineCancelled()):
                with self.assertRaises(PipelineCancelled):
                    run_speaker_review(
                        json_path=json_path,
                        sentences=[sentence],
                        prompt_dir=PROJECT_DIR / "prompt",
                        config={},
                        base_url="https://example.invalid",
                        api_key="secret",
                        default_model="test-model",
                        speaker_prefix="说话人",
                        keep_time=True,
                    )

    def test_failure_policy_fallback(self):
        sentence = self.make_sentence()
        unknown = fallback_decision(sentence, "mark_unknown", "unknown")
        original = fallback_decision(sentence, "keep_original", "unknown")
        self.assertEqual(unknown["target_speaker"], "unknown")
        self.assertEqual(original["target_speaker"], "SPEAKER_0")

    def test_full_review_sparse_payload_defaults_to_keep(self):
        sentences = self.full_sentences()
        result = validate_full_review_response(
            self.full_payload(), sentences, {"SPEAKER_0", "SPEAKER_1"}, "unknown", "overlap", 12
        )
        baseline, forced = build_full_baseline(sentences, result["overrides"], normalize_review_config({}, sentences))
        self.assertFalse(forced)
        self.assertTrue(all(item["operation"] == "KEEP" for item in baseline.values()))

    def test_full_review_rejects_keep_override_and_low_confidence_reassign(self):
        sentences = self.full_sentences()
        invalid = self.full_payload(overrides=[{
            "source_id": "r000.s000000", "operation": "KEEP", "confidence": 0.9, "reason_codes": ["SOURCE_SPK_PRIOR"],
        }])
        with self.assertRaisesRegex(ValueError, "只能使用"):
            validate_full_review_response(invalid, sentences, {"SPEAKER_0", "SPEAKER_1"}, "unknown", "overlap", 12)
        payload = self.full_payload(overrides=[{
            "source_id": "r000.s000000", "operation": "REASSIGN", "target_speaker": "SPEAKER_1", "confidence": 0.5, "reason_codes": ["ROLE_CONSISTENCY"],
        }])
        result = validate_full_review_response(payload, sentences, {"SPEAKER_0", "SPEAKER_1"}, "unknown", "overlap", 12)
        baseline, forced = build_full_baseline(sentences, result["overrides"], normalize_review_config({}, sentences))
        self.assertEqual(baseline["r000.s000000"]["target_speaker"], "SPEAKER_0")
        self.assertEqual(baseline["r000.s000000"]["source"], "low_confidence_full_override")
        self.assertEqual(forced, {0})

    def test_risk_keep_preserves_full_reassign_baseline(self):
        sentence = self.make_sentence("补偿一万元", 0, 0, flags=["涉及金额"])
        baseline = self.baseline(sentence, "SPEAKER_1")
        baseline["operation"] = "REASSIGN"
        baseline["source"] = "full_review"
        normalized = normalize_segment_decision(
            self.decision(sentence)["decisions"][0],
            baseline,
            sentence,
            normalize_review_config({}, [sentence]),
            [],
        )
        self.assertEqual(normalized["operation"], "REASSIGN")
        self.assertEqual(normalized["target_speaker"], "SPEAKER_1")

    def test_low_confidence_risk_reassign_preserves_baseline_and_marks_high_impact(self):
        sentence = self.make_sentence("补偿一万元", 0, 0, flags=["涉及金额"])
        decision = self.decision(sentence, "REASSIGN", target_speaker="SPEAKER_1")["decisions"][0]
        decision["confidence"] = 0.5
        normalized = normalize_segment_decision(
            decision,
            self.baseline(sentence),
            sentence,
            normalize_review_config({}, [sentence]),
            [],
        )
        self.assertEqual(normalized["target_speaker"], "SPEAKER_0")
        self.assertEqual(normalized["source"], "low_confidence_segment")
        self.assertTrue(normalized["review_required"])

    def test_full_review_discards_invalid_optional_risk_item(self):
        sentences = self.full_sentences()
        payload = self.full_payload(risk_items=[{
            "source_ids": [sentence.source_id for sentence in sentences],
            "risk_type": "AMBIGUOUS_TURN",
            "confidence": 0.95,
            "reason_codes": ["TURN_TAKING"],
        }])
        result = validate_full_review_response(
            payload,
            sentences,
            {"SPEAKER_0", "SPEAKER_1"},
            "unknown",
            "overlap",
            2,
        )
        self.assertEqual(result["risk_items"], [])
        self.assertEqual(result["discarded_risk_items"][0]["source_count"], 4)

    def test_risk_segment_groups_only_structural_risks(self):
        sentences = [
            self.make_sentence("一万", 0, 0, 0, 1000, ["涉及金额"]),
            self.make_sentence("甲" * 20, 1, 1, 1200, 4500, ["快速换人"]),
            self.make_sentence("乙", 2, 0, 4600, 5000),
        ]
        signals = [
            RiskSignal((0,), "涉及金额", 5, False),
            RiskSignal((1,), "LONG_FAST_TURN", 80, True),
        ]
        config = normalize_review_config({}, sentences)
        segments = build_risk_segments(sentences, signals, config)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].core_indexes, (1,))
        self.assertNotIn(0, segments[0].core_indexes)

    def test_risk_segment_keeps_aba_together_and_context_separate(self):
        sentences = [
            self.make_sentence("左", 0, 0, 0, 300),
            self.make_sentence("对", 1, 1, 400, 600, ["关键短答"]),
            self.make_sentence("右", 2, 0, 700, 1000),
        ]
        segments = build_risk_segments(
            sentences,
            [RiskSignal((0, 1, 2), "ABA_SHORT_TURN", 80, True)],
            normalize_review_config({}, sentences),
        )
        self.assertEqual(segments[0].core_indexes, (0, 1, 2))
        self.assertFalse(set(segments[0].core_indexes) & set(segments[0].context_indexes))

    def test_semantic_flags_keep_rejection_distinct_from_acceptance(self):
        sentence = self.make_sentence("我不同意这个安排", 0)
        review_sentences([sentence])
        self.assertIn("拒绝立场", sentence.review_flags)
        self.assertIn("否定或限制", sentence.review_flags)
        self.assertNotIn("接受立场", sentence.review_flags)

    def test_high_impact_terms_alone_do_not_create_segment(self):
        sentences = [
            self.make_sentence("补偿一万元", 0, 0, 0, 100),
            self.make_sentence("2026年8月5日", 1, 0, 2000, 2100),
            self.make_sentence("社保需要补缴", 2, 0, 4000, 4100),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        self.assertFalse(any(signal.primary for signal in signals))
        self.assertEqual(build_risk_segments(sentences, signals, normalize_review_config({}, sentences)), [])

    def test_same_label_high_impact_question_answer_creates_segment(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 400, 800),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        self.assertIn("SAME_LABEL_HIGH_IMPACT_QA", {signal.signal_type for signal in signals})
        segments = build_risk_segments(sentences, signals, normalize_review_config({}, sentences))
        self.assertEqual(segments[0].core_indexes, (0, 1))

    def test_semantic_risk_keep_guard_preserves_baseline(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 400, 800),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        config = normalize_review_config({}, sentences)
        keep_decision = self.decision(sentences[1])["decisions"][0]
        guard_signals = conservative_keep_guard_signals(
            sentences[1],
            1,
            sentences,
            (0, 1),
            {sentences[0].source_id: self.decision(sentences[0])["decisions"][0], sentences[1].source_id: keep_decision},
            signals,
        )
        normalized = normalize_segment_decision(
            keep_decision, self.baseline(sentences[1]), sentences[1], config, guard_signals
        )
        self.assertEqual(guard_signals, ["SAME_LABEL_HIGH_IMPACT_QA"])
        self.assertEqual(normalized["operation"], "KEEP")
        self.assertEqual(normalized["target_speaker"], "SPEAKER_0")
        self.assertEqual(normalized["source"], "risk_segment")

    def test_semantic_risk_keep_guard_preserves_reassign_and_isolated_amount(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 400, 800),
            self.make_sentence("补偿一万元", 2, 1, 3000, 3300),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        config = normalize_review_config({}, sentences)
        reassigned_decision = self.decision(sentences[1], "REASSIGN", target_speaker="SPEAKER_1")["decisions"][0]
        isolated_decision = self.decision(sentences[2])["decisions"][0]
        raw_decisions = {
            sentences[0].source_id: self.decision(sentences[0])["decisions"][0],
            sentences[1].source_id: reassigned_decision,
            sentences[2].source_id: isolated_decision,
        }
        reassigned = normalize_segment_decision(
            reassigned_decision,
            self.baseline(sentences[1]),
            sentences[1],
            config,
            conservative_keep_guard_signals(sentences[1], 1, sentences, (0, 1, 2), raw_decisions, signals),
        )
        isolated = normalize_segment_decision(
            isolated_decision,
            self.baseline(sentences[2]),
            sentences[2],
            config,
            conservative_keep_guard_signals(sentences[2], 2, sentences, (0, 1, 2), raw_decisions, signals),
        )
        self.assertEqual(reassigned["operation"], "REASSIGN")
        self.assertEqual(isolated["operation"], "KEEP")

    def test_semantic_risk_keep_guard_propagates_ambiguous_same_speaker_cluster(self):
        sentences = [
            self.make_sentence("我先重新说明", 0, 0, 0, 200),
            self.make_sentence("补偿怎么算？", 1, 0, 300, 600),
            self.make_sentence("一万五我可以接受", 2, 0, 700, 1100),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        raw_decisions = {
            sentences[0].source_id: self.decision(sentences[0])["decisions"][0],
            sentences[1].source_id: self.decision(sentences[1])["decisions"][0],
            sentences[2].source_id: self.decision(sentences[2], "REVIEW_REQUIRED")["decisions"][0],
        }
        guard_signals = conservative_keep_guard_signals(
            sentences[0],
            0,
            sentences,
            (0, 1, 2),
            raw_decisions,
            signals,
        )
        normalized = normalize_segment_decision(
            raw_decisions[sentences[0].source_id],
            self.baseline(sentences[0]),
            sentences[0],
            normalize_review_config({}, sentences),
            guard_signals,
        )
        self.assertEqual(guard_signals, ["SAME_LABEL_HIGH_IMPACT_QA"])
        self.assertEqual(normalized["operation"], "KEEP")

    def test_unreviewed_high_impact_keep_guard_only_marks_implicit_keep(self):
        sentences = [
            self.make_sentence("补偿一万元", 0, 0, flags=["涉及金额"]),
            self.make_sentence("补偿一万元", 1, 0, flags=["涉及金额"]),
            self.make_sentence("普通内容", 2, 0),
        ]
        decisions = {
            sentences[0].source_id: {"operation": "KEEP", "source": "implicit_keep"},
            sentences[1].source_id: {"operation": "KEEP", "source": "risk_segment"},
            sentences[2].source_id: {"operation": "KEEP", "source": "implicit_keep"},
        }
        guarded = apply_unreviewed_high_impact_keep_guard(
            sentences,
            decisions,
            normalize_review_config({}, sentences),
        )
        self.assertEqual(guarded, [{"source_id": sentences[0].source_id, "review_flags": ["涉及金额"]}])
        self.assertEqual(decisions[sentences[0].source_id]["operation"], "KEEP")
        self.assertEqual(decisions[sentences[0].source_id]["source"], "implicit_keep")
        self.assertEqual(decisions[sentences[1].source_id]["operation"], "KEEP")
        self.assertEqual(decisions[sentences[2].source_id]["operation"], "KEEP")

    def test_risk_segment_audit_records_conservative_keep_guard(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 400, 800),
        ]
        review_sentences(sentences)
        config = normalize_review_config({}, sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        segments = build_risk_segments(sentences, signals, config)
        baseline = {
            sentence.source_id: {
                "operation": "KEEP",
                "target_speaker": "SPEAKER_0",
                "confidence": 1.0,
            }
            for sentence in sentences
        }
        raw_decisions = {
            sentence.source_id: self.decision(sentence)["decisions"][0]
            for sentence in sentences
        }
        with patch("review_funasr_speakers.call_json_with_retries", return_value=raw_decisions):
            results, audits, _ = run_risk_segment_pass(
                segments,
                sentences,
                signals,
                {"speakers": []},
                baseline,
                config,
                "{{ review_input }}",
                "https://example.invalid",
                "secret",
                "test-model",
            )
        self.assertEqual(results[sentences[1].source_id]["operation"], "KEEP")
        self.assertEqual(results[sentences[1].source_id]["source"], "risk_segment")
        self.assertEqual(
            audits[0]["conservative_keep_guards"],
            [
                {
                    "source_id": sentences[0].source_id,
                    "signal_types": ["SAME_LABEL_HIGH_IMPACT_QA"],
                    "review_flags": ["劳动争议内容"],
                },
                {
                    "source_id": sentences[1].source_id,
                    "signal_types": ["SAME_LABEL_HIGH_IMPACT_QA"],
                    "review_flags": ["涉及金额", "接受立场"],
                },
            ],
        )

    def test_risk_segment_failure_preserves_baseline(self):
        sentences = [
            self.make_sentence("甲", 0, 0, 0, 300),
            self.make_sentence("乙", 1, 1, 400, 800),
        ]
        segments = build_risk_segments(
            sentences,
            [RiskSignal((0, 1), "ABA_SHORT_TURN", 80, True)],
            normalize_review_config({}, sentences),
        )
        baseline = {
            sentences[0].source_id: self.baseline(sentences[0], "SPEAKER_1"),
            sentences[1].source_id: self.baseline(sentences[1]),
        }
        with patch("review_funasr_speakers.call_json_with_retries", side_effect=RuntimeError("请求失败")):
            results, audits, _ = run_risk_segment_pass(
                segments,
                sentences,
                [],
                {"speakers": []},
                baseline,
                normalize_review_config({}, sentences),
                "{{ review_input }}",
                "https://example.invalid",
                "secret",
                "test-model",
            )
        self.assertEqual(audits[0]["status"], "failed")
        self.assertEqual(results[sentences[0].source_id]["target_speaker"], "SPEAKER_1")
        self.assertEqual(results[sentences[0].source_id]["source"], "risk_segment_failure_baseline")

    def test_same_label_question_answer_includes_limited_follow_through(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 100),
            self.make_sentence("一万五我可以接受", 1, 0, 200, 300),
            *[
                self.make_sentence(f"后续{index}", index, 0, index * 200, index * 200 + 100)
                for index in range(2, 10)
            ],
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        qa_signal = next(signal for signal in signals if signal.signal_type == "SAME_LABEL_HIGH_IMPACT_QA")
        self.assertEqual(qa_signal.source_indexes, tuple(range(10)))

    def test_local_label_turbulence_with_high_impact_content_creates_segment(self):
        sentences = [
            self.make_sentence("前一句", 0, 0, 0, 100),
            self.make_sentence("第二句", 1, 1, 150, 250),
            self.make_sentence("第三句", 2, 0, 300, 400),
            self.make_sentence("停顿后的句子", 3, 1, 2500, 2600),
            self.make_sentence("第五句", 4, 0, 2650, 2750),
            self.make_sentence("公积金需要补缴", 5, 1, 2800, 3050),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        turbulence = next(
            signal for signal in signals
            if signal.signal_type == "LOCAL_LABEL_TURBULENCE_HIGH_IMPACT"
        )
        self.assertEqual(turbulence.source_indexes, (0, 1, 2, 3, 4, 5))
        self.assertEqual(
            build_risk_segments(sentences, signals, normalize_review_config({}, sentences))[0].core_indexes,
            (0, 1, 2, 3, 4, 5),
        )

    def test_same_label_question_answer_burst_includes_lead_in(self):
        sentences = [
            self.make_sentence(f"前导{index}", index, 0, index * 300, index * 300 + 100)
            for index in range(5)
        ]
        sentences.extend([
            self.make_sentence("补偿基数是多少？", 5, 0, 1500, 1600),
            self.make_sentence("我再说明一下", 6, 0, 1700, 1800),
            self.make_sentence("14600这个金额算", 7, 0, 1900, 2000),
        ])
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        qa_signal = next(signal for signal in signals if signal.signal_type == "SAME_LABEL_HIGH_IMPACT_QA")
        self.assertEqual(qa_signal.source_indexes, tuple(range(8)))
        segments = build_risk_segments(sentences, signals, normalize_review_config({}, sentences))
        self.assertEqual(segments[0].core_indexes, tuple(range(8)))

    def test_structure_adjacent_high_impact_content_extends_segment(self):
        sentences = [
            self.make_sentence("甲" * 20, 0, 0, 0, 3500),
            self.make_sentence("一万元", 1, 1, 3600, 3900),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        self.assertIn("STRUCTURE_HIGH_IMPACT", {signal.signal_type for signal in signals})
        segments = build_risk_segments(sentences, signals, normalize_review_config({}, sentences))
        self.assertEqual(segments[0].core_indexes, (0, 1))

    def test_same_label_stance_and_position_conflicts_create_segments(self):
        stance_sentences = [
            self.make_sentence("我接受这个方案", 0, 0, 0, 300),
            self.make_sentence("我不同意这个方案", 1, 0, 400, 800),
        ]
        review_sentences(stance_sentences)
        stance_signals = collect_risk_signals(stance_sentences, {"risk_items": []}, set())
        self.assertIn("SAME_LABEL_STANCE_CONFLICT", {signal.signal_type for signal in stance_signals})

        position_sentences = [
            self.make_sentence("公司只愿意补一万元", 0, 0, 0, 300),
            self.make_sentence("我要求补偿", 1, 0, 400, 800),
        ]
        review_sentences(position_sentences)
        position_signals = collect_risk_signals(position_sentences, {"risk_items": []}, set())
        self.assertIn("SAME_LABEL_POSITION_CONFLICT", {signal.signal_type for signal in position_signals})

    def test_distant_same_label_question_answer_stays_weak(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 2000, 2400),
        ]
        review_sentences(sentences)
        signals = collect_risk_signals(sentences, {"risk_items": []}, set())
        self.assertNotIn("SAME_LABEL_HIGH_IMPACT_QA", {signal.signal_type for signal in signals})
        self.assertFalse(any(signal.primary for signal in signals))

    def test_oversize_component_partitions_without_overlap(self):
        sentences = [
            self.make_sentence(f"句{index}", index, 0, index * 1000, index * 1000 + 100)
            for index in range(25)
        ]
        signals = [RiskSignal(tuple(range(25)), "ROLE_DISCONTINUITY", 100, True)]
        config = normalize_review_config({}, sentences)
        segments = build_risk_segments(sentences, signals, config)
        self.assertEqual([len(segment.core_indexes) for segment in segments], [12, 12, 1])
        self.assertTrue(all(segment.forced_cut for segment in segments[:-1]))
        self.assertTrue(component_partition_passed(segments, config["max_risk_core_sentences"]))
        self.assertEqual(
            [index for segment in segments for index in segment.core_indexes],
            list(range(25)),
        )

    def test_component_partition_prefers_safe_boundary(self):
        sentences = [
            self.make_sentence(f"句{index}", index, index % 2, index * 1000, index * 1000 + 100)
            for index in range(25)
        ]
        sentences[6] = replace(sentences[6], start=10000, end=10100)
        signals = [RiskSignal((index,), "ROLE_DISCONTINUITY", 100, True) for index in range(25)]
        segments = build_risk_segments(sentences, signals, normalize_review_config({}, sentences))
        self.assertEqual(segments[0].core_indexes[-1], 5)
        self.assertEqual(segments[0].cut_rule, "safe_boundary")
        self.assertFalse(segments[0].forced_cut)

    def test_semantic_only_segment_never_allows_split(self):
        sentences = [
            self.make_sentence("补偿怎么算？", 0, 0, 0, 300),
            self.make_sentence("一万五我可以接受", 1, 0, 400, 800),
        ]
        review_sentences(sentences)
        signals = [RiskSignal((0, 1), "SAME_LABEL_HIGH_IMPACT_QA", 90, True)]
        config = normalize_review_config({}, sentences)
        segment = build_risk_segments(sentences, signals, config)[0]
        candidates, split_source_ids, eligibility = build_segment_candidates(sentences, segment, signals, config)
        baseline = {
            sentence.source_id: {
                "operation": "KEEP",
                "target_speaker": "SPEAKER_0",
                "confidence": 1.0,
            }
            for sentence in sentences
        }
        payload = build_risk_segment_input(segment, sentences, {"speakers": []}, baseline, candidates, split_source_ids, config)
        self.assertEqual(split_source_ids, set())
        self.assertEqual(set(eligibility.values()), {"no_split_signal"})
        self.assertTrue(all("SPLIT" not in item["allowed_operations"] for item in payload["core"]))

    def test_unreliable_boundary_does_not_allow_split(self):
        sentence = self.make_sentence("甲，乙", 0, 0, 0, 300)
        signals = [RiskSignal((0,), "LONG_FAST_TURN", 80, True)]
        config = normalize_review_config({}, [sentence])
        segment = build_risk_segments([sentence], signals, config)[0]
        _, split_source_ids, eligibility = build_segment_candidates([sentence], segment, signals, config)
        self.assertEqual(split_source_ids, set())
        self.assertEqual(eligibility[sentence.source_id], "no_reliable_internal_boundary")

    def test_boundary_candidates_preserve_endpoints_with_small_limit(self):
        sentence = self.make_sentence("甲，乙但是丙。", timestamps=[[0, 10]] * 6)
        offsets = [candidate.char_offset for candidate in build_boundary_candidates(sentence, 1)]
        self.assertEqual(offsets, [0, len(sentence.text)])

    def test_boundary_candidates_are_sparse_and_stable(self):
        sentence = self.make_sentence("甲，乙但是丙。", timestamps=[[0, 10], [20, 30], [400, 410], [420, 430], [440, 450], [460, 470], [480, 490]])
        candidates = build_boundary_candidates(sentence, 4)
        offsets = [candidate.char_offset for candidate in candidates]
        self.assertIn(0, offsets)
        self.assertIn(len(sentence.text), offsets)
        self.assertLessEqual(len(candidates), 4)
        self.assertEqual([candidate.boundary_id for candidate in candidates], [f"{sentence.source_id}.o{offset:03d}" for offset in offsets])

    def test_allowed_speakers_derive_from_source_labels(self):
        sentence = self.make_sentence()
        config = normalize_review_config({}, [sentence])
        self.assertEqual(config["allowed_speakers"], ["SPEAKER_0"])

    def test_review_config_rejects_fixed_model(self):
        with self.assertRaisesRegex(ValueError, "必须为 null"):
            normalize_review_config({"model": "other-model"}, [self.make_sentence()])

    def test_risk_segment_rejects_unauthorized_and_unreliable_split(self):
        sentence = self.make_sentence()
        payload = self.decision(sentence, "SPLIT", parts=[
            {"start_boundary_id": f"{sentence.source_id}.b000", "end_boundary_id": f"{sentence.source_id}.b002", "speaker": "SPEAKER_0"},
            {"start_boundary_id": f"{sentence.source_id}.b002", "end_boundary_id": f"{sentence.source_id}.b{len(sentence.text):03d}", "speaker": "SPEAKER_1"},
        ])
        candidates = self.candidates_for(sentence)
        with self.assertRaisesRegex(ValueError, "未提供可拆分"):
            validate_risk_segment_response(
                payload,
                [sentence],
                candidates,
                set(),
                {"SPEAKER_0", "SPEAKER_1"},
                "unknown",
                "overlap",
            )
        boundary_id = f"{sentence.source_id}.b002"
        candidates[sentence.source_id][boundary_id] = replace(
            candidates[sentence.source_id][boundary_id],
            reliable=False,
        )
        with self.assertRaisesRegex(ValueError, "内部边界必须可靠"):
            validate_risk_segment_response(
                payload,
                [sentence],
                candidates,
                {sentence.source_id},
                {"SPEAKER_0", "SPEAKER_1"},
                "unknown",
                "overlap",
            )

    def test_api_credentials_read_model_only_from_project_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            env_path.write_text(
                "API_KEY=test-key\nBASE_URL=https://example.invalid\nMODEL_NAME=test-model\n",
                encoding="utf-8",
            )
            paths = {"env_file": str(env_path)}
            config = {
                "model": None,
                "api_key_env": "API_KEY",
                "base_url_env": "BASE_URL",
                "model_name_env": "MODEL_NAME",
            }
            self.assertEqual(
                load_api_credentials(root, paths, config),
                ("test-key", "https://example.invalid", "test-model"),
            )
            config["model"] = "fixed-model"
            with self.assertRaisesRegex(SystemExit, "必须为 null"):
                load_api_credentials(root, paths, config)

    def test_polish_only_rejects_invalid_audit_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"audio")
            settings = default_settings()
            settings["paths"].update({
                "output_dir": str(root / "output"),
                "env_file": str(root / ".env"),
                "prompt_dir": str(PROJECT_DIR / "prompt"),
            })
            output_dir = root / "output" / audio_path.stem
            output_dir.mkdir(parents=True)
            (output_dir / "sample.json").write_text("{}", encoding="utf-8")
            (output_dir / "sample_speaker_review.json").write_text(
                json.dumps({"schema_version": 1, "integrity": {}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "sample_reviewed.txt").write_text("reviewed", encoding="utf-8")
            (output_dir / "sample_cleaned.txt").write_text("cleaned", encoding="utf-8")
            with patch("run_funasr_full_pipeline.generate_final_stage") as write_final:
                with self.assertRaisesRegex(RuntimeError, "schema v3"):
                    polish_one_audio(audio_path, settings, root)
            write_final.assert_not_called()

    def test_polish_only_keeps_existing_final_when_request_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"audio")
            settings = default_settings()
            settings["paths"].update({
                "output_dir": str(root / "output"),
                "env_file": str(root / ".env"),
                "prompt_dir": str(PROJECT_DIR / "prompt"),
            })
            (root / ".env").write_text(
                "API_KEY=test-key\nBASE_URL=https://example.invalid\nMODEL_NAME=test-model\n",
                encoding="utf-8",
            )
            output_dir = root / "output" / audio_path.stem
            output_dir.mkdir(parents=True)
            json_path = output_dir / "sample.json"
            json_path.write_text("{}", encoding="utf-8")
            audit = {
                "schema_version": 3,
                "run": {"source_json_sha256": sha256_file(json_path)},
                "integrity": {
                    "source_hash_verified": True,
                    "per_source_reconstruction_passed": True,
                    "global_reconstruction_passed": True,
                    "order_passed": True,
                    "coverage_passed": True,
                    "allowed_speaker_passed": True,
                    "unknown_from_explicit_review_passed": True,
                },
                "reviewed_spans": [{
                    "source_id": "r000.s000000",
                    "source_order": 0,
                    "char_start": 0,
                    "char_end": 2,
                    "start": 0,
                    "end": 100,
                    "spk": "SPEAKER_0",
                    "original_spk": 0,
                    "text": "内容",
                    "operation": "KEEP",
                    "confidence": 1.0,
                    "reason_codes": ["SOURCE_SPK_PRIOR"],
                    "review_required": False,
                    "review_flags": [],
                    "time_method": "source_range",
                }],
            }
            (output_dir / "sample_speaker_review.json").write_text(json.dumps(audit), encoding="utf-8")
            (output_dir / "sample_reviewed.txt").write_text("reviewed", encoding="utf-8")
            stale_path = output_dir / "sample_final.txt"
            stale_path.write_text("stale", encoding="utf-8")
            with patch(
                "run_funasr_full_pipeline.generate_final_stage",
                side_effect=RuntimeError("请求失败"),
            ):
                with self.assertRaisesRegex(RuntimeError, "请求失败"):
                    polish_one_audio(audio_path, settings, root)
            self.assertTrue(stale_path.exists())


if __name__ == "__main__":
    unittest.main()
