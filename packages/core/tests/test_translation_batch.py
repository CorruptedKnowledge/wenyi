"""Integration of token budgets and translation conversations across batch boundaries."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.i18n.resources import read_text
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.review.review_chunks import ReviewChunkService
from wenyi_core.pipeline.translate.translation import TranslationService
from wenyi_core.pipeline.translate.translation_batch import (
    TranslationBatchExecutor,
    resume_batches,
)
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler
from tests.pipeline_fixtures import batch_plan as _plan
from tests.pipeline_fixtures import fake_pipeline_config as _config


def test_resume_and_review_pack_by_tokens_and_preserve_completed_blank():
    segments = [
        Segment(index=10, source="hello world", target=""),
        Segment(index=11, source="foo bar"),
        Segment(index=12, source="x"),
    ]
    # Balanced batching splits [2, 2, 1] at budget 4 into [10] and [11, 12];
    # Review keeps greedy front-packing [10, 11] and [12].
    assert [[s.index for s in b] for b in resume_batches(segments, 4)] == [[10], [11, 12]]
    assert [[s.index for s in b] for b in resume_batches(segments, 5)] == [[10], [11, 12]]
    assert [[s.index for s in b] for b in ReviewChunkService.pack_contiguous(segments, 4)] == [
        [10, 11],
        [12],
    ]


def test_serial_resume_skips_glossary_extract_when_legacy_checkpoints_cover_rebalanced_batch(
    tmp_path,
):
    """Legacy greedy checkpoints (0:2) cover rebalanced completed sub-batches (0:1, 1:1)."""
    from wenyi_core.storage.run_store import STATUS_PENDING

    config = _config(str(tmp_path / "state"))
    config.segment.max_tokens_per_batch = 4
    config.pipeline.polish = False
    config.pipeline.review = False
    config.pipeline.annotation_alignment = False
    store = FileStorage(str(tmp_path / "state" / "demo" / "targets" / "zh"))
    store.save_manifest(
        {
            "title": "demo",
            "source_file": "demo.txt",
            "source_sha256": "a" * 64,
            "fmt": "text",
            "source_lang": "en",
            "target_lang": "zh",
            "chapters": [{"index": 0, "title": "Ch1", "status": STATUS_PENDING}],
        }
    )
    # Segment 0 is a saved blank (""), segment 1 is saved ("T1"), segment 2 is pending.
    # Balanced batching splits [2, 2, 1] @ 4 into [[0], [1, 2]], and resume_batches splits
    # the second raw batch on completion state into [[0], [1], [2]].
    chapter = Chapter(
        index=0,
        title="Ch1",
        segments=[
            Segment(index=0, source="hello world", target=""),
            Segment(index=1, source="foo bar", target="T1"),
            Segment(index=2, source="x", target=None),
        ],
    )
    store.save_chapter(chapter)
    # Legacy greedy run checkpointed segments [0, 1] as a single batch "0:2".
    store.log_event("batch_glossary_extracted", chapter=0, start_index=0, count=2)

    def handler(messages, tier, json_mode):
        if "literary translator" in messages[0]["content"]:
            return json.dumps({"translations": ["T2"]})
        if "terminology" in messages[0]["content"] and "extractor" in messages[0]["content"]:
            return json.dumps({"terms": []})
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    Orchestrator(config, client)._translation.run(store, book_synopsis="")

    saved = store.load_chapter(0)
    assert [s.target for s in saved.text_segments] == ["", "T1", "T2"]
    trans_calls = [c for c in client.calls if c["operation"] == "translation.body"]
    extract_calls = [c for c in client.calls if c["operation"] == "glossary.extract"]
    assert len(trans_calls) == 1
    # Completed sub-batches [0] and [1] are covered by legacy checkpoint "0:2", so only
    # newly translated batch [2] plus the normal end-of-chapter pass run extraction.
    assert len(extract_calls) == 2
    assert "T2" in extract_calls[0]["messages"][-1]["content"]
    assert "T1" not in extract_calls[0]["messages"][-1]["content"]

    with open(store.event_log_path, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    skipped = [e for e in events if e.get("event") == "batch_skipped"]
    assert [
        (e["start_index"], e["count"], e["glossary_extraction"]["skipped"]) for e in skipped
    ] == [
        (0, 1, 1),
        (1, 1, 1),
    ]


def test_serial_resume_extracts_only_uncovered_span_on_partial_checkpoint_overlap(tmp_path):
    """When all targets exist and legacy checkpoint 0:2 partially overlaps rebalanced [1, 2],
    only uncovered span (2, 1) runs extraction and writes checkpoint 2:1.
    """
    from wenyi_core.storage.run_store import STATUS_PENDING

    config = _config(str(tmp_path / "state"))
    config.segment.max_tokens_per_batch = 4
    config.pipeline.polish = False
    config.pipeline.review = False
    config.pipeline.annotation_alignment = False
    store = FileStorage(str(tmp_path / "state" / "demo" / "targets" / "zh"))
    store.save_manifest(
        {
            "title": "demo",
            "source_file": "demo.txt",
            "source_sha256": "b" * 64,
            "fmt": "text",
            "source_lang": "en",
            "target_lang": "zh",
            "chapters": [{"index": 0, "title": "Ch1", "status": STATUS_PENDING}],
        }
    )
    chapter = Chapter(
        index=0,
        title="Ch1",
        segments=[
            Segment(index=0, source="hello world", target="T0"),
            Segment(index=1, source="foo bar", target="T1"),
            Segment(index=2, source="x", target="T2"),
        ],
    )
    store.save_chapter(chapter)
    store.log_event("batch_glossary_extracted", chapter=0, start_index=0, count=2)

    def handler(messages, tier, json_mode):
        if "terminology" in messages[0]["content"] and "extractor" in messages[0]["content"]:
            return json.dumps({"terms": []})
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    Orchestrator(config, client)._translation.run(store, book_synopsis="")

    assert not [c for c in client.calls if c["operation"] == "translation.body"]
    extract_calls = [c for c in client.calls if c["operation"] == "glossary.extract"]
    # First call is uncovered span (2, 1) only; second is normal end-of-chapter extraction.
    assert len(extract_calls) == 2
    assert "T2" in extract_calls[0]["messages"][-1]["content"]
    assert "T1" not in extract_calls[0]["messages"][-1]["content"]
    with open(store.event_log_path, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    extracted_keys = [
        (e["start_index"], e["count"])
        for e in events
        if e.get("event") == "batch_glossary_extracted"
    ]
    assert extracted_keys == [(0, 2), (2, 1)]


def test_review_chunk_budget_caps_default_multiplier_and_preserves_overrides(tmp_path):
    """Review chunk budget caps 3x multiplier at 9600 while preserving small/large overrides."""
    from wenyi_core.agents.reviewer import Reviewer
    from wenyi_core.pipeline.review.review_workflow import ReviewService
    from wenyi_core.pipeline.run.runtime import PipelineRuntime

    cases = ((1, 3), (1800, 5400), (4800, 9600), (100_000, 100_000))
    captured_budgets: list[int] = []

    def spy_pack(segments, max_tokens):
        captured_budgets.append(max_tokens)
        return [segments]

    snapshots: list[dict] = []
    for configured, _ in cases:
        cfg = _config(str(tmp_path / f"state-{configured}"))
        cfg.segment.max_tokens_per_batch = configured
        client = FakeClient(handler=routing_handler)
        snapshots.append(ReviewService(PipelineRuntime(cfg, client))._review_config_snapshot())
        service = ReviewChunkService(cfg, client, Reviewer(client, cfg))
        chapter = Chapter(
            index=0,
            title="Ch1",
            segments=[Segment(index=0, source="alpha", target="甲")],
        )
        with patch.object(service, "pack_contiguous", side_effect=spy_pack):
            service.review_chapters([chapter], terms=[])

    expected_budgets = [expected for _, expected in cases]
    assert captured_budgets == expected_budgets
    assert [snap["review_chunk_budget"] for snap in snapshots] == expected_budgets
    # Different effective chunk budgets produce distinct reuse fingerprints so stale
    # chunk artifacts from an older layout cannot mix into a resumed report.
    assert snapshots[1] != snapshots[2]


@pytest.mark.parametrize("genre", ["literary", "academic"])
@pytest.mark.parametrize("invalid_continuation", [False, True])
def test_batch_polish_preserves_filtered_positions_and_fallback(invalid_continuation, genre):
    def handler(messages, tier, json_mode):
        if len(messages) == 4:
            return json.dumps(
                {"polished": ["wrong count"] if invalid_continuation else ["ONE", "TWO"]}
            )
        if f"{genre} translator" in messages[0]["content"]:
            return json.dumps({"translations": ["one", "two"]})
        return json.dumps({"polished": ["ONE", "123", "TWO"]})

    config = Config.from_dict({"llm": {"preset": "fake"}})
    config.set_prompt_genre(genre)
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    plan = _plan(["alpha", "123", "beta"])
    result = executor.execute(plan, polish=True)

    assert result.targets == ("ONE", "123", "TWO")
    assert result.before_polish == ("one", "123", "two")
    assert plan.segment_indices == (10, 11, 12)
    assert [c["operation"] for c in client.calls] == ["translation.body", "polish.body"] + (
        ["polish.body"] if invalid_continuation else []
    )
    assert client.calls[1]["messages"][:2] == client.calls[0]["messages"]
    assert client.calls[1]["messages"][2]["role"] == "assistant"
    rules = dict(config.language_policy().prompt_values)
    for call in client.calls:
        # Continuation repeats the complete rules in its own user turn.
        instruction = call["messages"][-1 if len(call["messages"]) == 4 else 0]["content"]
        assert rules["target_guidance"] in instruction
        assert read_text("shared/target_language.txt").strip() in instruction
        assert rules["punct_rule"] in instruction


def test_batch_passes_mineru_blank_allowance_to_translator():
    config = Config.from_dict({"llm": {"preset": "fake"}})
    client = FakeClient(handler=lambda *_: '{"translations": [""]}')
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    result = executor.execute(_plan(["OCR artifact"], allow_empty=True), polish=False)
    assert result.targets == ("",)
    assert result.before_polish == (None,)
    assert len(client.calls) == 1


@pytest.mark.parametrize("backend", ["mineru", "babeldoc", "text"])
def test_chapter_service_limits_blank_allowance_and_resumes_without_retranslation(
    tmp_path, backend
):
    from wenyi_core.agents.translator import AlignmentError
    from wenyi_core.pipeline.orchestrator import Orchestrator
    from wenyi_core.storage.run_store import STATUS_PENDING

    from tests.fake_llm import routing_handler
    from tests.review_fixtures import autofix_config as _config
    from tests.review_fixtures import autofix_store as _store

    store = _store(str(tmp_path))
    manifest = store.load_manifest()
    manifest["fmt"] = "text" if backend == "text" else "pdf"
    manifest["meta"] = {
        "text": {},
        "babeldoc": {"babeldoc": True},
        "mineru": {"mineru_epub_sha256": "a" * 64},
    }[backend]
    manifest["chapters"][0]["status"] = STATUS_PENDING
    store.save_manifest(manifest)
    chapter = store.load_chapter(0)
    chapter.text_segments[0].target = None
    store.save_chapter(chapter)
    config = _config(str(tmp_path / "state"))
    config.pipeline.polish = False
    config.pipeline.annotation_alignment = False
    config.pipeline.align_retry_limit = 0

    def handler(messages, tier, json_mode):
        if "literary translator" in messages[0]["content"]:
            return '{"translations": [""]}'
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    service = Orchestrator(config, client)._translation
    with pytest.raises(AlignmentError):
        service.run(store, book_synopsis="")
    chapter = store.load_chapter(0)
    assert chapter.text_segments[0].target is None
    chapter.text_segments[0].target = ""
    store.save_chapter(chapter)
    resumed_client = FakeClient(handler=handler)
    Orchestrator(config, resumed_client)._translation.run(store, book_synopsis="")
    assert store.load_chapter(0).text_segments[0].target == ""
    assert store.pending_chapters() == []
    assert not [c for c in resumed_client.calls if c["operation"] == "translation.body"]


class TestTranslationService(unittest.TestCase):
    def test_progress_label_prefers_real_title(self):
        self.assertEqual(TranslationService.chapter_progress_label("引言", 0), "引言")
        self.assertEqual(TranslationService.chapter_progress_label("第一章", 1), "第一章")
        self.assertEqual(
            TranslationService.chapter_progress_label("", 1), "progress.stage.chapter:2"
        )

    def test_resume_batch_from_continuation_receives_absolute_annotation_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(os.path.join(directory, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.annotation_alignment = False
            cfg.segment.max_tokens_per_batch = 6
            chapter = Chapter(
                index=0,
                segments=[
                    Segment(
                        index=0,
                        source="aa",
                        target="既译",
                        anchor="tn0_0",
                        meta={
                            "epub_annotations": {
                                "version": 1,
                                "source_length": 6,
                                "items": [
                                    {
                                        "id": "second-piece",
                                        "mode": "point",
                                        "source_start": 3,
                                        "source_end": 3,
                                        "target_key": "notes.xhtml#n2",
                                        "relation": "noteref",
                                    },
                                    {
                                        "id": "third-piece",
                                        "mode": "point",
                                        "source_start": 5,
                                        "source_end": 5,
                                        "target_key": "notes.xhtml#n3",
                                        "relation": "noteref",
                                    },
                                ],
                            }
                        },
                    ),
                    Segment(index=1, source="bb", cont=True),
                    Segment(index=2, source="cc", cont=True),
                ],
            )
            registry = {
                "version": 1,
                "contexts": {
                    "notes.xhtml#n2": {"source_blocks": ["Note two"]},
                    "notes.xhtml#n3": {"source_blocks": ["Note three"]},
                },
            }
            store = FileStorage(os.path.join(directory, "state", "book"))
            store.save_chapter(chapter)
            store.save_manifest(
                {
                    "title": "Book",
                    "fmt": "epub",
                    "source_sha256": "a" * 64,
                    "source_lang": "en",
                    "target_lang": "zh",
                    "chapters": [{"index": 0, "title": "", "status": "pending"}],
                }
            )
            glossary = GlossaryStore(store.glossary_path)
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            captured: list[list[list[dict[str, str]]]] = []

            def process(plan, **kwargs):
                from wenyi_core.pipeline.translate.translation_batch import BatchResult

                captured.append(plan.annotation_contexts)
                return BatchResult(
                    tuple(f"译{source}" for source in plan.sources), (None,) * len(plan.sources)
                )

            try:
                with (
                    patch.object(orch._translation._batches, "execute", side_effect=process),
                    patch.object(
                        orch._translation,
                        "extract_batch_glossary",
                        return_value={
                            "inserted": 0,
                            "conflict": 0,
                            "unchanged": 0,
                            "updated": 0,
                        },
                    ),
                    patch.object(
                        orch._runtime.extractor,
                        "extract_and_store",
                        return_value={
                            "inserted": 0,
                            "conflict": 0,
                            "unchanged": 0,
                            "updated": 0,
                        },
                    ),
                ):
                    orch._translation.translate_chapter(
                        0,
                        store,
                        glossary,
                        "",
                        translation_history={},
                        source_corpus="aabbcc",
                        annotation_context_registry=registry,
                    )
            finally:
                glossary.close()

            self.assertEqual(
                captured,
                [
                    [
                        [{"target_key": "notes.xhtml#n2", "source": "Note two"}],
                        [{"target_key": "notes.xhtml#n3", "source": "Note three"}],
                    ]
                ],
            )
