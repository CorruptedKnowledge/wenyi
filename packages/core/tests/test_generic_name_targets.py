"""A name that is also an English title or common word ("Grace", "King") stays a name.

The target-language generic-word list exists to drop pronouns and forms of address, whose
rendering depends on who is speaking. It must not drop a character, place or organization
whose name happens to be one of those words, in any stage that uses the list.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from wenyi_core.agents.batch_risk import assess_batch_risk
from wenyi_core.agents.book_bible import BookBibleAgent
from wenyi_core.config import Config
from wenyi_core.glossary.targets import is_generic_target
from wenyi_core.i18n.languages import glossary_generic_target_words
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.models.glossary import GlossaryTerm

_EN = glossary_generic_target_words("en")


def test_one_capitalized_word_of_a_name_type_is_a_name():
    assert not is_generic_target("Grace", _EN, term_type="person")
    assert not is_generic_target("King", _EN, term_type="place")
    # Without a name type, or in another shape, the word is still a title or pronoun.
    assert is_generic_target("Grace", _EN)
    assert is_generic_target("Grace", _EN, term_type="term")
    assert is_generic_target("grace", _EN, term_type="person")
    assert is_generic_target("Your Grace", _EN, term_type="person")
    assert is_generic_target("I", _EN, term_type="person")
    assert not is_generic_target("Lord Jiang", _EN)
    assert not is_generic_target("Grace", frozenset(), term_type="term")


_GRACE = GlossaryTerm(source="格蕾丝", target="Grace", type="person")


def test_batch_prescreen_flags_a_missed_name_that_is_also_a_title_word():
    report = assess_batch_risk(
        ["格蕾丝推门而入。"],
        ["Grayce pushed the door open."],
        source_lang="zh",
        target_lang="en",
        glossary_terms=[_GRACE],
    )
    assert any(a.code == "glossary_miss" for a in report.alerts)


def test_batch_prescreen_still_skips_generic_title_terms():
    report = assess_batch_risk(
        ["国主下旨。"],
        ["The sovereign issued a decree."],
        source_lang="zh",
        target_lang="en",
        glossary_terms=[GlossaryTerm(source="国主", target="the king", type="term")],
    )
    assert not any(a.code == "glossary_miss" for a in report.alerts)


def test_consistency_sweep_patches_a_missed_name_that_is_also_a_title_word():
    from wenyi_core.pipeline.translate.consistency_sweep import ConsistencySweepService

    seen: list[list[str]] = []

    class _Editor:
        def edit_batch(self, sources, drafts, *, glossary_terms=None, style=""):
            seen.append(list(sources))
            return SimpleNamespace(failed=False, patched_indices=(), targets=tuple(drafts))

        def adjudicate_drift(self, clusters, *, style="", conventions=""):
            return {}

    chapters = {
        0: Chapter(
            index=0,
            title="第一章",
            segments=[Segment(index=0, source="格蕾丝点了点头。", target="Grayce nodded.")],
        ),
        1: Chapter(
            index=1,
            title="第二章",
            segments=[Segment(index=0, source="夜幕降临。", target="Night fell.")],
        ),
    }

    class _Store:
        def load_chapter(self, idx):
            return chapters[idx]

        def save_chapter(self, ch):
            chapters[ch.index] = ch

        def open_conflicts(self):
            return []

    ConsistencySweepService(_Editor(), source_lang="zh", target_lang="en").run(
        _Store(), chapter_indices=[0, 1], terms=[_GRACE]
    )
    assert seen == [["格蕾丝点了点头。"]]


def _compile(probes: dict[str, str], adjudicated: list[dict], corpus: str):
    cfg = Config(source_lang="zh", target_lang="en", llm=LLMConfig(preset="fake"))
    adjudicate_users: list[str] = []

    def handler(messages, operation, json_mode):
        sys_msg = messages[0]["content"]
        if "terminology prober" in sys_msg:
            return json.dumps({"renderings": probes})
        if "Book Bible" in sys_msg:
            adjudicate_users.append(messages[-1]["content"])
            return json.dumps({"terms": adjudicated, "conventions": []})
        return "{}"

    res = BookBibleAgent(FakeClient(handler), cfg).compile_bible(["见到了。"], corpus)
    return res, adjudicate_users


_GRACE_CORPUS = (
    "格蕾丝来了。格蕾丝笑了。格蕾丝点头。格蕾丝叹息。格蕾丝起身。格蕾丝坐下。格蕾丝离去。"
)


def test_book_bible_keeps_a_recurring_name_that_is_also_a_title_word():
    res, users = _compile(
        {"格蕾丝": "Grace"},
        [{"source": "格蕾丝", "target": "Grace", "type": "person", "gender": "female"}],
        _GRACE_CORPUS,
    )
    # Both probes agree on a name-shaped rendering, so the adjudicator decides what it is.
    assert users and "格蕾丝" in users[0]
    assert [(t.source, t.target, t.type) for t in res.terms] == [("格蕾丝", "Grace", "person")]
    assert res.variants_by_source.get("格蕾丝") == ("Grace",)


def test_book_bible_still_drops_a_title_word_the_adjudicator_calls_a_term():
    res, users = _compile(
        {"国主": "King"},
        [{"source": "国主", "target": "King", "type": "term"}],
        "国主下旨。国主点头。国主叹息。国主问政。国主大笑。国主起身。国主坐下。国主离去。",
    )
    assert users
    assert res.terms == ()
    # The recurring title is still watched for cross-chapter drift.
    assert dict(res.divergent_titles)["国主"] == ("King",)
