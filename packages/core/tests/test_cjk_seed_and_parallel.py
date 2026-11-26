"""Tests for lean CJK upfront glossary seeding and parallel chapter execution."""

from __future__ import annotations

import json
import os
import threading
import time

from wenyi_core.agents.cjk_seed import CjkGlossarySeeder, SeedCandidate, mine_cjk_candidates
from wenyi_core.config import Config
from wenyi_core.glossary.prompt_scope import estimate_term_tokens, split_glossary_tiers
from wenyi_core.models.glossary import TYPE_NAVIGATION, GlossaryTerm
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.run_store import STATUS_DONE

from tests.fake_llm import FakeClient, MeteredFakeClient, routing_handler
from tests.pipeline_fixtures import fake_pipeline_config, require_file_storage


def test_mine_cjk_candidates_filters_noise_and_falls_back_without_digests():
    corpus = (
        "青云宗的林尘修炼《太虚剑诀》，拜见大人。林尘再次前往青云宗施展《太虚剑诀》。"
        "孤独自离开。偶尔路过一次。"
    )
    digests = ["本章林尘在青云宗修炼《太虚剑诀》，并与大人交谈，孤也出场。"]
    candidates = mine_cjk_candidates(
        digests,
        corpus,
        existing_sources={"青云宗"},
        context_dependent_sources={"大人", "孤"},
    )
    sources = [c.source for c in candidates]
    assert "太虚剑诀" in sources
    assert "林尘" in sources
    assert "青云宗" not in sources
    assert "大人" not in sources
    assert "孤" not in sources
    # Shadowed substring inside "太虚剑诀" should be suppressed when >=80% shadowed.
    assert "太虚" not in sources

    # Without digests, bounded corpus sampling still discovers recurring CJK entities.
    fallback = mine_cjk_candidates([], corpus, context_dependent_sources={"大人", "孤"})
    fallback_sources = [c.source for c in fallback]
    assert "青云宗" in fallback_sources
    assert "林尘" in fallback_sources


def test_cjk_seeder_preserves_person_named_grace_and_caches_system_prefix():
    systems: list[str] = []

    def handler(messages, tier, json_mode):
        systems.append(messages[0]["content"])
        return json.dumps(
            {
                "terms": [
                    {"source": "格蕾丝", "target": "Grace", "type": "person", "gender": "female"},
                    {"source": "长老称呼", "target": "Elder", "type": "term"},
                    {"source": "青云宗", "target": "Azure Cloud Sect", "type": "organization"},
                ]
            },
            ensure_ascii=False,
        )

    cfg = Config.from_dict(
        {"language": {"source": "zh", "target": "en"}, "llm": {"preset": "fake"}}
    )
    seeder = CjkGlossarySeeder(FakeClient(handler=handler), cfg)
    # 65 candidates forces 2 chunks so we verify byte-identical KV prefix across chunks.
    candidates = [
        SeedCandidate("格蕾丝", 5, "格蕾丝走进大厅"),
        SeedCandidate("长老称呼", 4, "长老称呼"),
        SeedCandidate("青云宗", 8, "青云宗大殿"),
        *[SeedCandidate(f"宗门{i:02d}", 2, "测试") for i in range(62)],
    ]
    terms = seeder.seed(candidates, book_synopsis="Synopsis.", style="Crisp.", max_workers=2)
    by_source = {t.source: t for t in terms}
    assert "格蕾丝" in by_source and by_source["格蕾丝"].target == "Grace"
    assert "青云宗" in by_source and by_source["青云宗"].target == "Azure Cloud Sect"
    assert "长老称呼" not in by_source  # Generic non-person word filtered out.
    assert len(systems) == 2
    assert systems[0] == systems[1]


def test_split_glossary_tiers_keeps_frozen_prefix_independent_of_batch_overflow():
    terms = [
        GlossaryTerm(source="林尘", target="Lin Chen", type="person"),
        GlossaryTerm(source="青云宗", target="Azure Cloud Sect", type="organization"),
        GlossaryTerm(source="太虚剑诀", target="Grand Void Sword Art", type="term"),
        GlossaryTerm(source="第一章 启程", target="Chapter 1: Departure", type=TYPE_NAVIGATION),
    ]
    budget = estimate_term_tokens(terms[0]) + estimate_term_tokens(terms[1])
    p1, o1 = split_glossary_tiers(terms, "林尘施展太虚剑诀，第一章 启程", prefix_budget=budget)
    p2, o2 = split_glossary_tiers(terms, "青云宗的大殿很安静", prefix_budget=budget)
    # Frozen prefix is byte-identical across batches regardless of what overflow terms match.
    assert [t.source for t in p1] == [t.source for t in p2] == ["林尘", "青云宗"]
    assert "太虚剑诀" in [t.source for t in o1]
    assert "第一章 启程" in [t.source for t in o1]
    assert o2 == []


def test_parallel_chapters_seeds_once_protects_user_terms_and_runs_concurrently(tmp_path):
    book = (
        "第一章 启程\n\n"
        "林尘离开青云宗，踏上修行之路。\n\n"
        "林尘带着太虚剑诀前往天城。\n\n"
        "第二章 试炼\n\n"
        "青云宗的林尘在天城遇到试炼。\n\n"
        "太虚剑诀震动了整个青云宗。\n\n"
        "第三章 突破\n\n"
        "林尘终于突破，青云宗欢呼。\n"
    )
    src_path = tmp_path / "webnovel.txt"
    src_path.write_text(book, encoding="utf-8")

    cfg = fake_pipeline_config(os.path.join(tmp_path, "state"))
    cfg.source_lang = "zh"
    cfg.target_lang = "en"
    cfg.pipeline.review = False
    cfg.pipeline.polish = False
    cfg.pipeline.book_understanding = False
    cfg.pipeline.chapter_concurrency = 3

    lock = threading.Lock()
    active = 0
    peak = 0
    seed_calls = 0

    def handler(messages, tier, json_mode):
        nonlocal active, peak, seed_calls
        system = messages[0]["content"]
        if "compile a consistent glossary" in system:
            seed_calls += 1
            return json.dumps(
                {
                    "terms": [
                        {
                            "source": "林尘",
                            "target": "Lin Chen",
                            "type": "person",
                            "gender": "male",
                        },
                        {"source": "青云宗", "target": "Wrong Sect", "type": "organization"},
                    ]
                },
                ensure_ascii=False,
            )
        if tier == "translation.body":
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                return routing_handler(messages, tier, json_mode)
            finally:
                with lock:
                    active -= 1
        return routing_handler(messages, tier, json_mode)

    client = MeteredFakeClient(handler=handler)
    orch = Orchestrator(cfg, client=client)
    store = require_file_storage(orch.prepare(str(src_path)))
    # Pre-seed a user-owned term (first_chapter=None): upfront seed must never overwrite it.
    store.upsert_term(GlossaryTerm(source="青云宗", target="Azure Cloud Sect", type="organization"))

    orch.run(str(src_path))
    assert peak >= 2
    assert seed_calls == 1
    by_source = {t.source: t for t in store.all_terms()}
    assert by_source["青云宗"].target == "Azure Cloud Sect"
    assert by_source["林尘"].target == "Lin Chen"
    manifest = store.load_manifest()
    assert all(row["status"] == STATUS_DONE for row in manifest["chapters"])

    # Resuming a completed run reuses the cached seed artifact with 0 extra seed calls.
    orch.run(str(src_path))
    assert seed_calls == 1
