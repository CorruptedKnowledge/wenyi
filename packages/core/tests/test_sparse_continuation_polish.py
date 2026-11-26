"""Tests for continuation-aware sparse polish and minimal escaped-newline hygiene."""

from __future__ import annotations

import json

from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.sparse_editor import SparseSemanticEditor, _apply_validated_patches
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.pipeline.translate.post_normalize import repair_escaped_newlines
from wenyi_core.pipeline.translate.translation_batch import BatchPlan, TranslationBatchExecutor

from tests.fake_llm import FakeClient


def _zh_en_config() -> Config:
    return Config.from_dict(
        {
            "language": {"source": "zh", "target": "en"},
            "llm": {"preset": "fake"},
            "pipeline": {"polish": True, "polish_mode": "sparse"},
        }
    )


def test_sparse_polish_continues_translation_conversation_for_kv_cache():
    calls: list[dict] = []

    def handler(messages, tier, json_mode):
        calls.append({"operation": tier, "messages": messages})
        if tier == "translation.body":
            return json.dumps({"translations": ["Draft zero.", "Draft one literal calque."]})
        if tier == "polish.body":
            return json.dumps(
                {"patches": [{"index": 1, "text": "Clean natural English.", "reason": "calque"}]}
            )
        raise AssertionError(f"unexpected tier {tier}")

    cfg = _zh_en_config()
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(
        Translator(client, cfg),
        Polisher(client, cfg),
        SparseSemanticEditor(client, cfg),
    )
    plan = BatchPlan(
        chapter=0,
        start_index=0,
        segment_indices=(0, 1),
        sources=("第一句。", "第二句。"),
        terms=(GlossaryTerm(source="林尘", target="Lin Chen", type="person"),),
        context="",
        style="Punchy xianxia.",
        book_synopsis="A young cultivator rises.",
        chapter_digest="Chapter 1.",
        annotation_contexts=[[], []],
        next_source="",
    )
    res = executor.execute(plan, polish=True, polish_mode="sparse")
    assert not res.polish_failed
    assert res.targets == ("Draft zero.", "Clean natural English.")
    assert res.before_polish == ("Draft zero.", "Draft one literal calque.")
    assert len(calls) == 2
    translate_msgs = calls[0]["messages"]
    polish_msgs = calls[1]["messages"]
    # Continuation turn preserves the exact [system, user, assistant] prefix for DeepSeek KV cache.
    assert len(polish_msgs) == 4
    assert polish_msgs[:2] == translate_msgs
    assert polish_msgs[2]["role"] == "assistant"
    assert polish_msgs[3]["role"] == "user"


def test_sparse_polish_falls_back_to_standalone_cached_batch_when_continuation_fails():
    calls: list[dict] = []

    def handler(messages, tier, json_mode):
        calls.append({"operation": tier, "messages": messages})
        if tier == "translation.body":
            return json.dumps({"translations": ["Draft zero.", "Draft one."]})
        if tier == "polish.body" and len(messages) == 4:
            return "not json at all"
        if tier == "polish.body" and len(messages) == 2:
            return json.dumps(
                {"patches": [{"index": 0, "text": "Patched zero.", "reason": "omission"}]}
            )
        raise AssertionError(f"unexpected call {tier}")

    cfg = _zh_en_config()
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(
        Translator(client, cfg),
        Polisher(client, cfg),
        SparseSemanticEditor(client, cfg),
    )
    plan = BatchPlan(
        chapter=0,
        start_index=0,
        segment_indices=(0, 1),
        sources=("原文一。", "原文二。"),
        terms=(GlossaryTerm(source="青云宗", target="Azure Cloud Sect", type="organization"),),
        context="",
        style="Crisp tone.",
        book_synopsis="",
        chapter_digest="",
        annotation_contexts=[[], []],
        next_source="",
    )
    res = executor.execute(plan, polish=True, polish_mode="sparse")
    assert not res.polish_failed
    assert res.targets == ("Patched zero.", "Draft one.")
    assert len(calls) == 3
    standalone_system = calls[2]["messages"][0]["content"]
    assert "<style_data>" in standalone_system
    assert "<glossary_data>" in standalone_system
    assert "青云宗 → Azure Cloud Sect" in standalone_system


def test_sparse_patch_validation_is_strict_and_never_drops_markers_or_duplicates():
    drafts = [
        "Hello 〘object:1〙 world.",
        "Keep non-empty draft.",
        "Original two.",
    ]
    res = _apply_validated_patches(
        drafts,
        [
            {"index": 0, "text": "Hello world.", "reason": "dropped marker"},
            {"index": 1, "text": "   ", "reason": "blanked"},
            {"index": 2, "text": "First fix.", "reason": "omission"},
            {"index": 2, "text": "Second overwrite.", "reason": "duplicate index"},
            {"index": True, "text": "Bool index.", "reason": "invalid"},
            {"index": 99, "text": "Out of bounds.", "reason": "invalid"},
        ],
    )
    assert not res.failed
    assert res.targets == ("Hello 〘object:1〙 world.", "Keep non-empty draft.", "First fix.")
    assert res.patched_indices == (2,)


def test_repair_escaped_newlines_preserves_source_literals_code_urls_and_markers():
    assert repair_escaped_newlines("第一行\n第二行", "Line 1\\nLine 2") == "Line 1\nLine 2"
    # Source already has literal \n -> untouched.
    assert repair_escaped_newlines(r"literal \n here", r"literal \n here") == r"literal \n here"
    # Protected spans untouched.
    assert (
        repair_escaped_newlines(
            "原文", "Before `a\\nb` https://ex.com/\\n 〘object:a\\nb〙 end\\nnext"
        )
        == "Before `a\\nb` https://ex.com/\\n 〘object:a\\nb〙 end\nnext"
    )
