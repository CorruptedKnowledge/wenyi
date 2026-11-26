"""Lean CJK upfront glossary seeder for parallel ZH->EN (and future JA/KO->EN) web novels.

Mines recurring CJK proper-noun candidates from chapter digests (or bounded source samples
when digests are absent) and compiles a compact canonical entity glossary in parallel
chunks over the ``glossary.extract`` route so parallel chapter workers share consistent
names and a frozen KV-cacheable system prefix.

Pure agent module: zero pipeline or storage imports.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..glossary.store import GlossaryStore, GlossaryTerm
from ..i18n.languages import (
    glossary_context_dependent_sources,
    glossary_generic_target_words,
)
from ..llm.json_parser import JsonParseError
from ..llm.retrying import TruncatedResponseError, is_resumable_provider_interrupt
from . import prompts
from .base import Agent

_LOGGER = logging.getLogger(__name__)

# Contiguous CJK ideograph runs and katakana runs (for future JA support).
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_KATAKANA_RE = re.compile(r"[\u30a1-\u30fa\u30fc]{2,12}")
# Titles/techniques/factions explicitly marked in CJK brackets.
_BRACKETED_RE = re.compile(r"[《「『【]([^》」』】\n]{2,12})[》」』】]")
_NGRAM_MIN = 2
_NGRAM_MAX = 5
_VALID_TYPES = frozenset({"person", "place", "organization", "term"})
_GENERIC_SINGLE_WORD_RE = re.compile(r"[A-Za-z]+")
_SNIPPET_RADIUS = 36
_CHUNK_SIZE = 60


@dataclass(frozen=True)
class SeedCandidate:
    """One recurring CJK candidate term and a short raw context snippet."""

    source: str
    count: int
    snippet: str


def _first_snippet(corpus: str, term: str) -> str:
    pos = corpus.find(term)
    if pos < 0:
        return ""
    start = max(0, pos - _SNIPPET_RADIUS)
    end = min(len(corpus), pos + len(term) + _SNIPPET_RADIUS)
    return corpus[start:end].replace("\n", " ").strip()


def _bounded_sample_texts(
    source_corpus: str, *, max_slices: int = 16, slice_chars: int = 1500
) -> list[str]:
    """Sample evenly across ``source_corpus`` when chapter digests are unavailable."""
    if not source_corpus:
        return []
    total = len(source_corpus)
    if total <= max_slices * slice_chars:
        return [source_corpus]
    step = max(1, (total - slice_chars) // max(1, max_slices - 1))
    return [source_corpus[i * step : i * step + slice_chars] for i in range(max_slices)]


def mine_cjk_candidates(
    digests: Sequence[str],
    source_corpus: str,
    *,
    existing_sources: set[str] | frozenset[str] = frozenset(),
    context_dependent_sources: set[str] | frozenset[str] = frozenset(),
    max_candidates: int = 480,
) -> list[SeedCandidate]:
    """Deterministically mine recurring CJK proper-noun candidates for upfront seeding.

    Reads chapter digests first (high proper-noun density); if digests are absent, falls
    back to evenly spaced slices of ``source_corpus``. Requires >= 2 corpus occurrences,
    drops context-dependent forms and existing entries, and suppresses substrings shadowed
    >= 80% by a longer compound.
    """
    if not source_corpus:
        return []
    mining_texts = [d for d in digests if d and d.strip()]
    if not mining_texts:
        mining_texts = _bounded_sample_texts(source_corpus)

    raw_hits: Counter[str] = Counter()
    for text in mining_texts:
        for m in _BRACKETED_RE.finditer(text):
            token = m.group(1).strip()
            if _CJK_RUN_RE.search(token) or _KATAKANA_RE.search(token):
                raw_hits[token] += 2
        for m in _KATAKANA_RE.finditer(text):
            raw_hits[m.group(0)] += 1
        # Chinese has no spaces between words: mine 2..5 char sliding CJK n-grams inside
        # each hanzi run, then validate exact recurrence against the full corpus.
        for run_match in _CJK_RUN_RE.finditer(text):
            run = run_match.group(0)
            run_len = len(run)
            for size in range(_NGRAM_MIN, min(_NGRAM_MAX, run_len) + 1):
                for start in range(0, run_len - size + 1):
                    raw_hits[run[start : start + size]] += 1

    # Count exact occurrences in the full source corpus and filter noise.
    counts: dict[str, int] = {}
    for token in raw_hits:
        if token in existing_sources or token in context_dependent_sources:
            continue
        freq = source_corpus.count(token)
        if freq >= 2:
            counts[token] = freq

    if not counts:
        return []

    # Drop shorter substrings shadowed >= 80% by a longer mined compound, and drop any
    # longer compound that is merely a superstring of an existing term with identical count.
    by_length_desc = sorted(counts, key=lambda s: (-len(s), -counts[s], s))
    kept: list[str] = []
    for token in by_length_desc:
        freq = counts[token]
        shadowed = any(
            len(longer) > len(token) and token in longer and counts[longer] * 5 >= freq * 4
            for longer in kept
        )
        if not shadowed:
            kept.append(token)

    # Also drop substrings shadowed >= 80% by an already-existing glossary source.
    existing_counts = {
        src: source_corpus.count(src) for src in existing_sources if len(src) > _NGRAM_MIN
    }
    kept = [
        tok
        for tok in kept
        if not any(
            len(ex) > len(tok) and tok in ex and ex_cnt * 5 >= counts[tok] * 4
            for ex, ex_cnt in existing_counts.items()
        )
    ]

    # Rank by corpus frequency descending (tie-break deterministically by token).
    kept.sort(key=lambda s: (-counts[s], -len(s), s))
    top = kept[: max(1, max_candidates)]
    return [
        SeedCandidate(source=tok, count=counts[tok], snippet=_first_snippet(source_corpus, tok))
        for tok in top
    ]


class CjkGlossarySeeder(Agent):
    """Compile recurring CJK web-novel candidates into canonical glossary terms."""

    def seed(
        self,
        candidates: Sequence[SeedCandidate],
        *,
        book_synopsis: str = "",
        style: str = "",
        existing_terms: Sequence[GlossaryTerm] = (),
        max_workers: int = 4,
    ) -> list[GlossaryTerm]:
        """Adjudicate mined CJK candidates into canonical terms in parallel chunks."""
        if not candidates:
            return []

        system_prompt = self._build_system_prompt(book_synopsis, style, existing_terms)
        chunks = [
            list(candidates[i : i + _CHUNK_SIZE]) for i in range(0, len(candidates), _CHUNK_SIZE)
        ]

        workers = min(max(1, max_workers), len(chunks))
        if workers <= 1:
            chunk_results = [self._adjudicate_chunk(system_prompt, chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wenyi-cjk-seed") as ex:
                futures = [ex.submit(self._adjudicate_chunk, system_prompt, ch) for ch in chunks]
                chunk_results = [fut.result() for fut in futures]

        context_sources = glossary_context_dependent_sources(self.src)
        generic_words = glossary_generic_target_words(self.tgt)
        existing_keys = {t.source for t in existing_terms}

        merged: list[GlossaryTerm] = []
        seen: set[str] = set()
        for batch in chunk_results:
            for term in batch:
                if (
                    term.source in seen
                    or term.source in existing_keys
                    or term.source in context_sources
                ):
                    continue
                if self._is_generic_non_person(term, generic_words):
                    continue
                seen.add(term.source)
                merged.append(term)
        return merged

    def _build_system_prompt(
        self,
        book_synopsis: str,
        style: str,
        existing_terms: Sequence[GlossaryTerm],
    ) -> str:
        """Build a byte-stable KV-cacheable system prompt shared across all seed chunks."""
        parts = [
            f"You compile a consistent glossary for translating a {self.src} web novel into "
            f"{self.tgt}.",
            "Given candidate recurring source terms with corpus frequency and a short snippet, "
            "KEEP ONLY true recurring proper nouns and world-building terminology: character "
            "names, places, sects/factions, cultivation realms, techniques, artifacts, and "
            "named institutions or titles tied to one specific entity.",
            "DROP ordinary words, verbs, adjectives, generic pronouns, and context-dependent "
            "address forms (e.g. 大人, 公子, 师兄, 前辈, 孤, 朕, 臣) unless attached to a specific "
            "proper name.",
            "For every kept person, include gender ('male', 'female', or 'unknown') using only "
            "evidence visible in the synopsis/snippet; never spoil later plot twists.",
            'Return strict JSON: {"terms":[{"source":"...","target":"...","type":"person|place|'
            'organization|term","gender":"male|female|unknown|","aliases":["..."]}]}',
        ]
        if style.strip():
            parts.append(prompts.wrap_untrusted_style("Style guide", style.strip()))
        if book_synopsis.strip():
            parts.append(
                prompts.wrap_untrusted_reference(
                    "Whole-book synopsis", "synopsis_data", book_synopsis.strip()
                )
            )
        if existing_terms:
            rendered = prompts.render_glossary(
                list(existing_terms),
                source_lang=self.src,
                genre=self.language_policy.context.genre,
            )
            parts.append(prompts.wrap_untrusted_glossary(rendered))
        return "\n\n".join(parts)

    def _adjudicate_chunk(
        self,
        system_prompt: str,
        chunk: Sequence[SeedCandidate],
    ) -> list[GlossaryTerm]:
        lines = [
            f"- {c.source} (x{c.count}): {c.snippet}" if c.snippet else f"- {c.source} (x{c.count})"
            for c in chunk
        ]
        user_prompt = (
            "[Candidate recurring source terms]\n"
            + "\n".join(lines)
            + '\n\nReturn JSON {"terms":[...]} keeping only true proper nouns and key world terms.'
        )
        try:
            data, _raw = self._complete_json_turn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                operation="glossary.extract",
                require_complete_payload=True,
            )
        except (JsonParseError, TruncatedResponseError, ValueError) as exc:
            _LOGGER.warning("CJK glossary seed chunk failed (%s); skipping chunk", exc)
            return []
        except Exception as exc:
            if is_resumable_provider_interrupt(exc):
                raise
            _LOGGER.warning("CJK glossary seed chunk error (%s); skipping chunk", exc)
            return []

        raw_terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(raw_terms, list):
            return []

        allowed_sources = {c.source for c in chunk}
        out: list[GlossaryTerm] = []
        for item in raw_terms:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            if not source or not target or source not in allowed_sources:
                continue
            term_type = str(item.get("type") or "term").strip().lower()
            if term_type not in _VALID_TYPES:
                term_type = "term"
            gender = str(item.get("gender") or "").strip() if term_type == "person" else ""
            raw_aliases = item.get("aliases")
            aliases = (
                [str(a).strip() for a in raw_aliases if isinstance(a, str) and a.strip()]
                if isinstance(raw_aliases, list)
                else []
            )
            out.append(
                GlossaryTerm(
                    source=source,
                    target=target,
                    type=term_type,
                    gender=gender,
                    aliases=aliases,
                    first_chapter=0,
                )
            )
        return out

    @staticmethod
    def _is_generic_non_person(term: GlossaryTerm, generic_words: frozenset[str]) -> bool:
        if not generic_words:
            return False
        words = _GENERIC_SINGLE_WORD_RE.findall(term.target)
        if not words:
            return False
        # Preserve capitalized single-word personal names like "Grace" or "Duke".
        if term.type == "person" and len(words) == 1 and words[0][0].isupper():
            return False
        return all(w.casefold() in generic_words for w in words)


__all__ = ["CjkGlossarySeeder", "GlossaryStore", "SeedCandidate", "mine_cjk_candidates"]
