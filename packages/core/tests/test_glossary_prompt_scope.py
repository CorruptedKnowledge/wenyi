"""A long book's glossary must not grow every model request with it."""

import json
import re
from contextlib import closing

import pytest
from wenyi_core.config import Config
from wenyi_core.glossary.prompt_scope import prompt_terms
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.ingest.tokens import count_tokens
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Chapter, Document, Segment
from wenyi_core.models.glossary import TYPE_APPELLATION, GlossaryTerm
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler


def _term(source, **fields):
    return GlossaryTerm(source=source, target=f"<{source}>", **fields)


def _sources(terms):
    return [term.source for term in terms]


def _fillers(count):
    """Terms that occur in no test source, in a stable insertion order."""
    return [_term(f"Filler{index:03d}") for index in range(count)]


def _cost(terms):
    """The budget that exactly fits ``terms``, measured the way the selection measures."""
    low, high = 0, 10_000
    while low < high:
        middle = (low + high) // 2
        if len(prompt_terms([*terms, _term("Probe")], "", budget=middle)) >= len(terms):
            high = middle
        else:
            low = middle + 1
    return low


def test_a_glossary_within_the_budget_is_sent_whole_and_in_order():
    terms = [_term("Zebra"), _term("Apple"), _term("Mango")]
    assert prompt_terms(terms, "No term occurs here.", budget=10_000) == terms
    assert prompt_terms([], "Anything.", budget=0) == []


def test_a_larger_glossary_keeps_its_leading_run_and_the_terms_the_text_uses():
    fillers = _fillers(40)
    terms = [*fillers, _term("Citadel"), _term("Harbor"), _term("Alice")]
    budget = _cost(fillers[:10])
    selected = prompt_terms(terms, "Alice walked to the Citadel.", budget=budget)
    # Insertion order survives: the leading run first, then matches as the glossary lists them.
    assert _sources(selected) == [*_sources(fillers[:10]), "Citadel", "Alice"]


def test_the_budget_boundary_is_inclusive():
    fillers = _fillers(6)
    exact = _cost(fillers[:3])
    assert _sources(prompt_terms(fillers, "", budget=exact)) == _sources(fillers[:3])
    assert _sources(prompt_terms(fillers, "", budget=exact - 1)) == _sources(fillers[:2])


def test_a_zero_budget_sends_only_what_the_text_uses():
    terms = [*_fillers(3), _term("Alice")]
    assert _sources(prompt_terms(terms, "Alice smiled.", budget=0)) == ["Alice"]
    assert prompt_terms(terms, "Nobody smiled.", budget=0) == []


def test_later_terms_match_by_alias_but_addresses_only_by_their_own_source():
    terms = [
        *_fillers(3),
        _term("Elizabeth", aliases=["Lizzy"]),
        _term("Lizzy dear", type=TYPE_APPELLATION, aliases=["Lizzy"]),
    ]
    assert _sources(prompt_terms(terms, "Lizzy laughed.", budget=0)) == ["Elizabeth"]


def test_appended_terms_never_change_the_leading_run():
    """The provider's prefix cache keeps hitting while translation adds terms."""
    fillers = _fillers(30)
    budget = _cost(fillers[:8])
    before = prompt_terms(fillers, "", budget=budget)
    after = prompt_terms([*fillers, _term("Later"), _term("Latest")], "", budget=budget)
    assert _sources(before) == _sources(after) == _sources(fillers[:8])


def test_request_size_does_not_grow_with_the_glossary():
    text = "Alice walked to the Citadel."
    sizes = []
    for count in (100, 1_000, 5_000):
        terms = [*_fillers(count), _term("Citadel"), _term("Alice")]
        selected = prompt_terms(terms, text, budget=200)
        assert _sources(selected)[-2:] == ["Citadel", "Alice"]
        sizes.append(len(selected))
    assert len(set(sizes)) == 1


@pytest.fixture
def config(tmp_path):
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "paths": {"state_dir": str(tmp_path / "state")},
            "segment": {"max_tokens_per_batch": 1, "max_tokens_per_segment": 0},
            "pipeline": {
                "review": False,
                "polish": True,
                "book_understanding": False,
                "annotation_alignment": False,
            },
        }
    )


def _glossary_lines(user):
    """Mapping lines of the first glossary block in a prompt."""
    return re.findall(r"^- (\S+) → ", user, flags=re.MULTILINE)


def _handler(messages, tier, json_mode):
    system = messages[0]["content"]
    if "terminology" in system and "extractor" in system:
        return json.dumps({"terms": []})
    return routing_handler(messages, tier, json_mode)


def _translate(tmp_path, config, budget, *, book="book", text=None):
    """Translate two one-paragraph batches against a glossary far larger than the budget."""
    config.pipeline.glossary_prompt_tokens = budget
    source = tmp_path / f"{book}.txt"
    source.write_text(
        text or "Alice waited by the window.\n\nThe Citadel gate opened.", encoding="utf-8"
    )
    orchestrator = Orchestrator(config, client=FakeClient(handler=_handler))
    store = orchestrator.prepare(str(source))
    for term in [*_fillers(300), _term("Citadel"), _term("Alice")]:
        store.upsert_term(term)
    client = FakeClient(handler=_handler)
    Orchestrator(config, client=client).run(str(source))
    return client.calls


@pytest.mark.parametrize("operation", ["translation.body", "polish.body"])
def test_requests_carry_the_leading_run_and_only_their_own_later_terms(tmp_path, config, operation):
    budget = _cost(_fillers(5))
    calls = [
        call for call in _translate(tmp_path, config, budget) if call["operation"] == operation
    ]
    carried = [_glossary_lines(call["messages"][1]["content"]) for call in calls]
    assert carried == [
        [*_sources(_fillers(5)), "Alice"],
        [*_sources(_fillers(5)), "Citadel"],
    ]


def test_extraction_carries_only_the_recurring_terms_its_own_text_uses(tmp_path, config):
    """Extraction already drops terms the book uses once; the rest follow the budget too."""
    text = "Alice waited by the window.\n\nThe Citadel gate opened for Alice at the Citadel."
    calls = _translate(tmp_path, config, 0, text=text)
    carried = [
        _glossary_lines(call["messages"][-1]["content"])
        for call in calls
        if call["operation"] == "glossary.extract"
    ]
    # One request per batch, then one for the whole chapter.
    assert carried == [["Alice"], ["Citadel", "Alice"], ["Citadel", "Alice"]]


def test_the_budget_bounds_the_translation_prompt(tmp_path, config):
    def largest(calls):
        return max(
            count_tokens(call["messages"][-1]["content"])
            for call in calls
            if call["operation"] == "translation.body"
        )

    bounded = largest(_translate(tmp_path, config, 100, book="bounded"))
    whole = largest(_translate(tmp_path, config, 1_000_000, book="whole"))
    assert bounded * 5 < whole


def test_the_whole_glossary_is_still_sent_when_it_fits(tmp_path, config):
    """The default budget leaves an ordinary glossary untouched, absent terms included."""
    assert config.pipeline.glossary_prompt_tokens == 4000
    source = tmp_path / "book.txt"
    source.write_text("Alice waited by the window.", encoding="utf-8")
    store = Orchestrator(config, client=FakeClient(handler=_handler)).prepare(str(source))
    store.upsert_term(_term("Zebra"))
    store.upsert_term(_term("Apple"))
    client = FakeClient(handler=_handler)
    Orchestrator(config, client=client).run(str(source))
    for call in client.calls:
        if call["operation"] in {"translation.body", "polish.body"}:
            assert _glossary_lines(call["messages"][1]["content"]) == ["Zebra", "Apple"]


def test_reviewer_blocks_carry_only_their_own_terms_beyond_the_budget(config):
    config.pipeline.glossary_prompt_tokens = 0
    seen = []

    def handler(messages, tier, json_mode):
        seen.append(_glossary_lines(messages[-1]["content"]))
        return json.dumps({"issues": [], "reviewed_segments": 1, "complete": True})

    reviewer = Orchestrator(config, client=FakeClient(handler=handler))._runtime.reviewer
    terms = [_term("Ann"), _term("Bob"), _term("Unused")]
    reviewer.review_result(["Ann meets the council."], ["安会见议会。"], terms)
    reviewer.review_result(["Bob leaves at dawn."], ["鲍勃黎明离开。"], terms)
    assert seen == [["Ann"], ["Bob"]]


def test_title_batches_carry_only_their_own_terms_beyond_the_budget(tmp_path, config):
    config.pipeline.glossary_prompt_tokens = 0
    store = FileStorage(str(tmp_path / "state"))
    chapters = [
        Chapter(index=0, title="The Citadel", segments=[Segment(index=0, source="Text.")]),
        Chapter(index=1, title="Alice returns", segments=[Segment(index=0, source="Text.")]),
    ]
    store.save_manifest(
        store.stage_document(
            Document(
                title="Book", source_lang="en", target_lang="zh", fmt="txt", chapters=chapters
            ),
            source_hash="a" * 64,
        )
    )
    seen = []

    def handler(messages, tier, json_mode):
        user = messages[-1]["content"]
        seen.append(_glossary_lines(user))
        count = len(re.findall(r"^\[\d+\] ", user, flags=re.MULTILINE))
        return json.dumps({"titles": ["标题"] * count}, ensure_ascii=False)

    with closing(GlossaryStore(store.glossary_path)) as glossary:
        for term in [_term("Unused"), _term("Citadel"), _term("Alice")]:
            glossary.upsert_term(term)
        Orchestrator(config, FakeClient(handler=handler))._translation._titles.run(store, glossary)
    assert seen == [["Citadel", "Alice"]]
