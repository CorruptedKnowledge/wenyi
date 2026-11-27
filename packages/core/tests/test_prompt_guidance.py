"""Regression tests for generalized prompt guidance, genre composition, and agent roles."""

from __future__ import annotations

import hashlib
import json
import re

import pytest
from wenyi_core.agents.cjk_seed import CjkGlossarySeeder
from wenyi_core.agents.sparse_editor import SparseSemanticEditor
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.i18n import languages
from wenyi_core.i18n.policy.models import PolicyContext
from wenyi_core.i18n.policy.resolver import resolve_policy
from wenyi_core.i18n.prompts import render
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.routing import inference_snapshot
from wenyi_core.pipeline.run.runtime import PipelineRuntime
from wenyi_core.pipeline.translate.seed_service import _SEED_SCHEMA_VERSION, CjkGlossarySeedService
from wenyi_core.storage.file import FileStorage

_WEB_NOVEL_OVERFIT_RE = re.compile(
    r"\b(xianxia|wuxia|cultivation|daoist|punchy|tanka|haiku)\b"
    r"|serial storytelling|game-like|web[- ]novel|Senior Brother|Fellow Daoist|Sect Master",
    re.IGNORECASE,
)


def test_honorific_rule_composes_genre_slots(monkeypatch):
    real_read_json = languages.read_json

    def patched_read_json(path: str):
        if path == "pairs/zh__en.json":
            return {"honorific": "Match ${genre_prose_kind} conventions."}
        return real_read_json(path)

    monkeypatch.setattr(languages, "read_json", patched_read_json)

    literary = languages.honorific_rule("zh", "en", genre="literary")
    academic = languages.honorific_rule("zh", "en", genre="academic")
    assert literary == "Match fiction prose conventions."
    assert academic == "Match academic prose conventions."

    academic_plan = resolve_policy(PolicyContext("zh", "en", genre="academic"))
    rules = dict(academic_plan.prompt_values)
    assert "Match academic prose conventions." in rules["lang_guidance"]
    assert "${genre_" not in rules["lang_guidance"]


@pytest.mark.parametrize(
    "source,target", [("zh", "en"), ("zh", "en-US"), ("zh", "en-GB"), ("ja", "en")]
)
@pytest.mark.parametrize("genre", ["literary", "academic"])
@pytest.mark.parametrize(
    "task,phase",
    [
        ("translator_system", "translation"),
        ("polisher_system", "translation"),
        ("polisher_continue_user", "translation"),
        ("precision_generation_system", "translation"),
        ("review_fixer_system", "review"),
    ],
)
def test_rendered_prompts_have_no_unexpanded_slots_or_subgenre_overfit(
    source, target, genre, task, phase
):
    cfg = Config.from_dict(
        {
            "language": {"source": source, "target": target},
            "pipeline": {
                "translation_mode": (
                    "three_draft_synthesis" if task == "precision_generation_system" else "standard"
                ),
            },
        }
    )
    cfg.set_prompt_genre(genre)
    prompt = render(
        task,
        src=source,
        tgt=target,
        plan=cfg.language_policy(phase),
        n=2,
        next_source="",
    )
    assert "${genre_" not in prompt
    assert "$lang_guidance" not in prompt
    assert "$target_guidance" not in prompt
    assert "$punct_rule" not in prompt
    assert not _WEB_NOVEL_OVERFIT_RE.search(prompt)
    if genre == "academic":
        assert not re.search(
            r"\b(novels?|fiction|fictional|literary|plot|foreshadowing)\b",
            prompt,
            re.IGNORECASE,
        )


@pytest.mark.parametrize("target", ["en", "en-US", "en-GB"])
def test_english_dialogue_cues_stay_scoped_to_english_literary_targets(target):
    zh_en_lit = render("translator_system", src="zh", tgt=target)
    zh_en_acad = render(
        "translator_system",
        src="zh",
        tgt=target,
        plan=resolve_policy(PolicyContext("zh", target, genre="academic")),
    )
    en_zh_lit = render("translator_system", src=target, tgt="zh")

    assert "contractions in informal speech" in zh_en_lit
    assert "'said'/'asked'" in zh_en_lit
    assert "contractions" not in zh_en_acad
    assert "'said'" not in zh_en_acad
    assert "logical connectors" in zh_en_acad
    assert "contractions" not in en_zh_lit
    assert "'said'" not in en_zh_lit
    assert "'asked'" not in en_zh_lit


def test_translator_system_zh_en_structure_and_craft_contracts():
    prompt = render("translator_system", src="zh", tgt="en")
    # Intra-paragraph syntactic freedom alongside strict meaning/paragraph boundaries.
    assert "Fidelity means meaning, subtext, and logic—not source word order" in prompt
    assert "reorder clauses and split or join sentences" in prompt
    # Deliberate source shifts in viewpoint, register, tense, and pacing are preserved.
    assert "shifts viewpoint, register, tense, or pacing" in prompt
    assert "unless the source deliberately shifts tense" in prompt
    # Glossary governs how terms render, not mechanical name repetition.
    assert "fixes how recurring names and terms render, not how often they appear" in prompt
    # Continuity context, annotations, and batch-boundary lookahead stay clear and grounded.
    assert "Context, annotations & batch boundary:" in prompt
    assert "unclosed quote, dangling clause, or cutoff" in prompt
    assert "continues into the following source paragraph" in prompt
    assert "do not assume continuity without source evidence" in prompt
    assert "never translate, copy, borrow from, or emit an output item for it" in prompt
    # Chinese->English literary craft: syntax, hedges, brackets, dialogue colons, no silly scar tissue.
    assert "的-chains" in prompt
    assert (
        "Absorb recurring Chinese hedges, intensifiers, and formulaic speech/thought tags" in prompt
    )
    assert "preferring natural 'I' with matching register" in prompt
    assert "single quotes ('...') for direct first-person inner monologue" in prompt
    assert "Render 【...】 as [...]" in prompt
    assert "without CJK brackets or Markdown asterisks" in prompt
    assert "convert Chinese dialogue colons" in prompt
    assert "the lone one" not in prompt
    assert "the sir" not in prompt
    # Third-language parenthetical rule reconciled with no-gloss rule for ordinary source text.
    assert "a third language neither source nor target" in prompt
    assert "ordinary source terms or idioms must not receive parenthetical glosses" in prompt
    # Punctuation harmonizes English conversion with structure preservation.
    assert "convert foreign full-width punctuation and brackets to English equivalents" in prompt
    # JSON-escaped \n instruction lives in the output format rule, not as raw (\n) in pair text.
    assert "JSON-escaped \\n within the same string" in prompt
    assert "(\\n)" not in prompt


@pytest.mark.parametrize(
    "genre,expected_kind,expected_heading",
    [
        ("literary", "fiction", "Characters / Style guide"),
        ("academic", "academic writing", "Academic style guide"),
    ],
)
def test_sparse_editor_batch_prompt_uses_genre_labels_guidance_and_punct_rule(
    genre, expected_kind, expected_heading
):
    captured: list[dict] = []

    def handler(messages, tier, json_mode):
        captured.append({"messages": messages, "tier": tier})
        return json.dumps({"patches": []})

    cfg = Config.from_dict(
        {"language": {"source": "zh", "target": "en"}, "llm": {"preset": "fake"}}
    )
    cfg.set_prompt_genre(genre)
    editor = SparseSemanticEditor(FakeClient(handler=handler), cfg)
    res = editor.edit_batch(
        ["原文。"],
        ["Translated."],
        style="Measured tone.",
        next_source="下一段。",
    )
    assert not res.failed
    assert len(captured) == 1
    system = captured[0]["messages"][0]["content"]
    assert f"Simplified Chinese→English {expected_kind} translations" in system
    assert "web-novel" not in system
    assert "Language guidance:" in system
    assert "Chinese→English guidance:" in system
    assert "Use standard English punctuation and spacing" in system
    assert f"[{expected_heading}:" in system
    assert "register mismatch with the source" in system
    assert "elevate the register above the source" in system
    assert "preserve unfinished syntax or open dialogue" in system
    if genre == "academic":
        assert "fiction" not in system
        assert "微微" not in system
        assert "embedded verse" not in system
    else:
        assert "微微" in system


@pytest.mark.parametrize(
    "genre,expected_role",
    [("literary", "literary translator"), ("academic", "academic translator")],
)
def test_plain_text_fallback_uses_genre_role_labels_and_carries_next_source(genre, expected_role):
    captured: list[dict] = []

    def handler(messages, tier, json_mode):
        captured.append({"messages": messages, "json_mode": json_mode})
        if json_mode:
            return '{"translations": [""]}'
        return "Recovered translation."

    cfg = Config.from_dict(
        {"language": {"source": "zh", "target": "en"}, "llm": {"preset": "fake"}}
    )
    cfg.set_prompt_genre(genre)
    translator = Translator(FakeClient(handler=handler), cfg)
    result = translator.translate_batch(["“你好"], next_source="世界。”")
    assert result == ["Recovered translation."]
    plain_call = next(call for call in captured if call["json_mode"] is False)
    system = plain_call["messages"][0]["content"]
    user = plain_call["messages"][1]["content"]
    assert f"experienced {expected_role} translating Simplified Chinese" in system
    assert (
        "Even if the paragraph begins with an open quotation mark or looks like a fragment"
        in system
    )
    assert "real line breaks" in system
    assert "[Simplified Chinese paragraph to translate]\n“你好" in user
    assert "[Following source paragraph]" in user
    assert "世界。”" in user
    if genre == "academic":
        assert "literary translator" not in system


def test_cjk_seeder_prompt_and_seed_fingerprint_follow_genre_without_rebilling_literary(tmp_path):
    literary_cfg = Config.from_dict(
        {"language": {"source": "zh", "target": "en"}, "llm": {"preset": "fake"}}
    )
    academic_cfg = Config.from_dict(
        {"language": {"source": "zh", "target": "en"}, "llm": {"preset": "fake"}}
    )
    academic_cfg.set_prompt_genre("academic")

    lit_prompt = CjkGlossarySeeder(FakeClient(), literary_cfg)._build_system_prompt(
        "", "Calm voice.", []
    )
    acad_prompt = CjkGlossarySeeder(FakeClient(), academic_cfg)._build_system_prompt(
        "", "Formal analysis.", []
    )

    assert "Simplified Chinese novel into English" in lit_prompt
    assert "[Characters / Style guide:" in lit_prompt
    assert "Simplified Chinese scholarly book into English" in acad_prompt
    assert "[Academic style guide:" in acad_prompt
    assert "discipline-specific concepts" in acad_prompt
    assert not _WEB_NOVEL_OVERFIT_RE.search(lit_prompt)
    assert not _WEB_NOVEL_OVERFIT_RE.search(acad_prompt)

    store = FileStorage(str(tmp_path / "run"))
    store.save_manifest({"source_sha256": "abc123", "source_lang": "zh", "target_lang": "en"})
    lit_svc = CjkGlossarySeedService(PipelineRuntime(literary_cfg, client=FakeClient()))
    acad_svc = CjkGlossarySeedService(PipelineRuntime(academic_cfg, client=FakeClient()))

    # Literary seed fingerprint stays identical to the pre-genre payload so existing runs don't re-bill.
    legacy_payload = {
        "v": _SEED_SCHEMA_VERSION,
        "source_sha256": "abc123",
        "src": "zh",
        "tgt": "en",
        "digests": [],
        "synopsis": "",
        "style": "",
        "routing": inference_snapshot(literary_cfg.llm, ("glossary.extract",)),
        "user_terms": [],
    }
    expected_legacy = hashlib.sha256(
        json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    lit_fp = lit_svc._fingerprint(store, [], "", "", [])
    acad_fp = acad_svc._fingerprint(store, [], "", "", [])
    assert lit_fp == expected_legacy
    assert acad_fp != lit_fp
