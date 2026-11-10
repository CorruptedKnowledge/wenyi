"""Glossary tests."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from wenyi_core.glossary.store import (
    TYPE_APPELLATION,
    TYPE_PERSON,
    GlossaryStore,
    GlossaryTerm,
    conflict_needed,
    same_translation,
    source_matches_text,
)


class TestGlossary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = GlossaryStore(os.path.join(self.tmp.name, "g.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_insert_and_lookup(self):
        r = self.store.upsert_term(
            GlossaryTerm(
                source="綾小路",
                target="绫小路",
                type=TYPE_PERSON,
                gender="male",
                aliases=["綾小路くん"],
                reading="あやのこうじ",
            ),
            chapter=0,
        )
        self.assertEqual(r, "inserted")
        t = self.store.get_term("綾小路")
        assert t is not None
        self.assertEqual(t.target, "绫小路")
        self.assertEqual(t.gender, "male")

    def test_terms_in_matches_alias(self):
        self.store.upsert_term(
            GlossaryTerm(source="綾小路", target="绫小路", aliases=["綾小路くん"])
        )
        hits = self.store.terms_in(
            self.store.all_terms(), "「おはよう、綾小路くん」と堀北が言った。"
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "綾小路")

    def test_terms_in_normalizes_case_and_character_width(self):
        self.store.upsert_term(GlossaryTerm(source="OpenAI", target="开放人工智能"))
        self.store.upsert_term(GlossaryTerm(source="ＡＢＣ", target="ABC 组织"))

        hits = self.store.terms_in(self.store.all_terms(), "openai 与 ABC")

        self.assertEqual(
            {term.source for term in hits},
            {"OpenAI", "ＡＢＣ"},
        )

    def test_ascii_source_match_respects_word_boundaries(self):
        self.assertTrue(source_matches_text("Ann", "Ann opened the door."))
        self.assertTrue(source_matches_text("ANN", "ann opened the door."))
        self.assertFalse(source_matches_text("Ann", "Anna opened the door."))

    def test_cyrillic_source_match_respects_word_boundaries(self):
        self.assertTrue(source_matches_text("гад", "Этот гад снова пришёл."))
        self.assertFalse(source_matches_text("гад", "Этот гадкий человек снова пришёл."))

    def test_appellation_does_not_match_bare_name_alias(self):
        self.store.upsert_term(
            GlossaryTerm(
                source="夏帆ちゃん",
                target="小夏帆",
                type=TYPE_APPELLATION,
                aliases=["夏帆"],
            )
        )
        self.assertEqual(self.store.terms_in(self.store.all_terms(), "夏帆は窓の外を見た。"), [])
        hits = self.store.terms_in(self.store.all_terms(), "「夏帆ちゃん」と母親が言った。")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "夏帆ちゃん")

    def test_recurring_terms_require_two_full_text_occurrences(self):
        terms = [
            GlossaryTerm(source="唯一术语", target="Unique"),
            GlossaryTerm(source="重复术语", target="Repeated"),
            GlossaryTerm(
                source="AliasCanonical",
                target="Alias",
                aliases=["别名"],
            ),
            GlossaryTerm(
                source="夏帆ちゃん",
                target="小夏帆",
                type=TYPE_APPELLATION,
                aliases=["夏帆"],
            ),
        ]
        corpus = "唯一术语。重复术语再次成为重复术语。别名先来，别名再来。夏帆出现两次，夏帆。"

        recurring = GlossaryStore.recurring_terms(terms, corpus)

        self.assertEqual(
            {term.source for term in recurring},
            {"重复术语", "AliasCanonical"},
        )

        overlapping_alias = GlossaryTerm(
            source="夏帆",
            target="Kaho",
            aliases=["夏帆ちゃん"],
        )
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [overlapping_alias],
                "夏帆ちゃん只在全文出现一次。",
            ),
            [],
        )

        cyrillic = GlossaryTerm(source="гад", target="畜生")
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [cyrillic],
                "Один гад ушёл, но гадкий человек остался гадким.",
            ),
            [],
        )
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [cyrillic],
                "Один гад ушёл, затем другой гад пришёл.",
            ),
            [cyrillic],
        )

    def test_conflict_keeps_current_until_resolved(self):
        self.store.upsert_term(GlossaryTerm(source="堀北", target="堀北"), chapter=0)
        # An alternate translation preserves the established mapping and records a candidate.
        r = self.store.upsert_term(GlossaryTerm(source="堀北", target="掘北"), chapter=1)
        self.assertEqual(r, "conflict")
        term = self.store.get_term("堀北")
        assert term is not None
        self.assertEqual(term.target, "堀北")
        self.assertEqual(len(self.store.open_conflicts()), 1)

        self.assertTrue(self.store.resolve_term("堀北", "掘北"))
        self.store.mark_conflicts_resolved("堀北")
        term = self.store.get_term("堀北")
        assert term is not None
        self.assertEqual(term.target, "掘北")
        self.assertEqual(term.status, "ok")
        self.assertEqual(self.store.open_conflicts(), [])

    def test_single_spelling_shortcut_matches_a_full_count(self):
        from wenyi_core.glossary.store import (
            GlossaryOccurrenceMatcher,
            _match_text,
            _source_occurrence_spans,
        )

        corpus = "Ann met Anna. ANN left; ann? 甲甲甲乙甲乙。гад, гадкий, Гад. ＡＢＣ abc"
        sources = ["Ann", "Anna", "甲", "甲甲", "甲乙", "乙", "гад", "abc", "absent", "an"]
        matcher = GlossaryOccurrenceMatcher(corpus)
        for minimum in (1, 2, 3, 4):
            with self.subTest(minimum=minimum):
                expected = {
                    source
                    for source in sources
                    if len(_source_occurrence_spans(_match_text(source), matcher.normalized_text))
                    >= minimum
                }
                terms = [GlossaryTerm(source=source, target="t") for source in sources]
                found = matcher.recurring_terms(terms, min_occurrences=minimum)
                self.assertEqual({term.source for term in found}, expected)

    def test_cosmetic_variant_is_not_a_conflict(self):
        self.store.upsert_term(GlossaryTerm(source="市政厅", target="City Hall"), chapter=0)
        for variant in ("city hall", "City-Hall", "CITY  HALL.", "Ｃｉｔｙ Ｈａｌｌ"):
            with self.subTest(variant=variant):
                result = self.store.upsert_term(
                    GlossaryTerm(source="市政厅", target=variant, aliases=["厅"]), chapter=1
                )
                self.assertEqual(result, "unchanged")
        term = self.store.get_term("市政厅")
        assert term is not None
        # The stored spelling wins; the variant only contributes metadata.
        self.assertEqual((term.target, term.status, term.aliases), ("City Hall", "ok", ["厅"]))
        self.assertEqual(self.store.open_conflicts(), [])

    def test_repeated_alternative_is_recorded_once(self):
        self.store.upsert_term(GlossaryTerm(source="老板", target="boss"), chapter=0)
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="老板", target="Chief"), chapter=1),
            "conflict",
        )
        for chapter in (2, 3):
            self.assertEqual(
                self.store.upsert_term(GlossaryTerm(source="老板", target="chief"), chapter),
                "unchanged",
            )
        # A genuinely new alternative is still recorded.
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="老板", target="owner"), chapter=4),
            "conflict",
        )
        self.assertEqual(
            [row["proposed_target"] for row in self.store.open_conflicts()], ["Chief", "owner"]
        )

    def test_rejected_alternative_does_not_reopen_the_term(self):
        self.store.upsert_term(GlossaryTerm(source="老板", target="boss"), chapter=0)
        self.store.upsert_term(GlossaryTerm(source="老板", target="chief"), chapter=1)
        # Keep the current translation, as the conflict list's "keep current" does.
        self.store.resolve_term("老板", "boss")
        self.store.mark_conflicts_resolved("老板")

        result = self.store.upsert_term(GlossaryTerm(source="老板", target="chief"), chapter=9)

        self.assertEqual(result, "unchanged")
        self.assertEqual(self.store.open_conflicts(), [])
        term = self.store.get_term("老板")
        assert term is not None
        self.assertEqual((term.target, term.status), ("boss", "ok"))

    def test_translation_equivalence_ignores_only_case_width_spacing_and_punctuation(self):
        self.assertTrue(same_translation("Lance", "lance"))
        self.assertTrue(same_translation("约翰·史密斯", "约翰史密斯"))
        self.assertTrue(same_translation("Mr. White", "Mr White"))
        self.assertFalse(same_translation("the boss", "boss"))
        self.assertFalse(same_translation("criminal", "criminals"))
        self.assertFalse(same_translation("堀北", "掘北"))
        self.assertFalse(same_translation("カード", "カド"))
        # Punctuation-only translations are compared exactly.
        self.assertFalse(same_translation("……", "——"))
        self.assertTrue(conflict_needed("boss", "owner", ["chief"]))
        self.assertFalse(conflict_needed("boss", "Boss", []))
        self.assertFalse(conflict_needed("boss", "Chief", ["chief"]))

    def test_concurrent_upserts_make_one_atomic_conflict_decision(self):
        path = os.path.join(self.tmp.name, "concurrent.db")
        initial = GlossaryStore(path)
        initial.close()
        barrier = threading.Barrier(2)

        def write(target: str) -> str:
            store = GlossaryStore(path)
            try:
                barrier.wait()
                return store.upsert_term(GlossaryTerm(source="Name", target=target), chapter=1)
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(write, ["译名甲", "译名乙"]))

        check = GlossaryStore(path)
        try:
            self.assertCountEqual(results, ["inserted", "conflict"])
            self.assertEqual(len(check.all_terms()), 1)
            self.assertEqual(len(check.open_conflicts()), 1)
        finally:
            check.close()

    def test_stats(self):
        self.store.upsert_term(GlossaryTerm(source="A", target="甲"))
        s = self.store.stats()
        self.assertEqual(s, {"terms": 1, "open_conflicts": 0})

    def test_all_terms_preserves_insert_order_not_type_source_sort(self):
        """Insertion order controls all_terms so new entries cannot disrupt prompt prefix
        caching.
        """
        # Insert terms in an order that differs from type/source sorting deliberately.
        self.store.upsert_term(
            GlossaryTerm(source="乙", target="Yi", type="term"),
            chapter=0,
        )
        self.store.upsert_term(
            GlossaryTerm(source="甲", target="Jia", type=TYPE_PERSON),
            chapter=0,
        )
        self.assertEqual(
            [term.source for term in self.store.all_terms()],
            ["乙", "甲"],
        )
        # Human resolution or same-target field merging must not change insertion position.
        self.assertTrue(self.store.resolve_term("乙", "Yi-updated"))
        terms = self.store.all_terms()
        self.assertEqual([term.source for term in terms], ["乙", "甲"])
        self.assertEqual(terms[0].target, "Yi-updated")
        # New terms may only append to the end.
        self.store.upsert_term(
            GlossaryTerm(source="丙", target="Bing", type=TYPE_PERSON),
            chapter=1,
        )
        self.assertEqual(
            [term.source for term in self.store.all_terms()],
            ["乙", "甲", "丙"],
        )


if __name__ == "__main__":
    unittest.main()
