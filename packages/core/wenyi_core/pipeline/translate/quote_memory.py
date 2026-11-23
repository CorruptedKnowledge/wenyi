"""Book-wide verbatim memory for repeated quotations, verse and epigraphs.

Detect non-trivial source spans (whole verse lines, quoted passages, or repeated aphoristic
clauses) that recur in two or more distinct paragraphs. Supports:
- Thread-safe ``clone()`` for parallel chapter waves so intra-chapter repeats work live.
- ``cross_chapter_spans()`` + ``seed_canonical()`` so cross-chapter repeated quotes can be
  pre-rendered once in compiled mode and shared identically across all parallel workers.
- Adaptive minimum length (10 normalized chars for CJK/Japanese, 24 for Latin scripts) so
  trivial dialogue like "Come with me" is never treated as a verbatim callback.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

_MIN_CJK_SPAN_CHARS = 10
_MIN_LATIN_SPAN_CHARS = 24
_CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
_QUOTED_SPAN_RE = re.compile(r"[「『“\"«„]([^「」『』“”\"«»„]{10,240})[」』”\"»“]")
_STRIP_PUNCT_RE = re.compile(r"[\s，。！？；：、,.!?;:\"'“”‘’「」『』«»„（）()【】\[\]—…·-]+")


def _normalize_span(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def _meets_min_length(norm_key: str) -> bool:
    if _CJK_OR_KANA_RE.search(norm_key):
        return len(norm_key) >= _MIN_CJK_SPAN_CHARS
    return len(norm_key) >= _MIN_LATIN_SPAN_CHARS


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


def extract_candidate_spans(source: str) -> list[str]:
    """Return candidate quotable spans (whole short passage, inner quotes, and sub-sentences)."""
    stripped = source.strip()
    if not stripped:
        return []
    spans: list[str] = []
    seen: set[str] = set()

    def _add(span: str) -> None:
        key = _normalize_span(span)
        if _meets_min_length(key) and key not in seen:
            spans.append(span)
            seen.add(key)

    if len(stripped) <= 240:
        _add(stripped)
    for match in _QUOTED_SPAN_RE.finditer(stripped):
        inner = match.group(1).strip()
        _add(inner)
        # Also index non-trivial sub-sentences inside a longer quoted poem/passage so a later
        # callback quoting just the second half still hits the same key.
        for part in _SENTENCE_SPLIT_RE.split(inner):
            part = part.strip()
            if part and part != inner:
                _add(part)
    return spans


class QuoteMemory:
    """Track canonical translations for repeated source quotations and verse across a book."""

    def __init__(
        self,
        repeated_keys: set[str],
        *,
        cross_chapter_examples: dict[str, str] | None = None,
    ):
        self._repeated_keys = repeated_keys
        self._cross_chapter_examples: dict[str, str] = dict(cross_chapter_examples or {})
        # normalized source key -> (earliest position, canonical source, canonical target)
        self._canonical: dict[str, tuple[tuple[int, int], str, str]] = {}

    @classmethod
    def from_chapters(cls, chapters: Sequence[tuple[int, Sequence[str]]]) -> QuoteMemory:
        """Index spans that appear in at least two distinct paragraphs across the book."""
        occurrences: dict[str, set[tuple[int, int]]] = defaultdict(set)
        chapters_by_key: dict[str, set[int]] = defaultdict(set)
        first_span_by_key: dict[str, str] = {}
        for chapter_index, sources in chapters:
            for seg_index, source in enumerate(sources):
                pos = (chapter_index, seg_index)
                for span in extract_candidate_spans(source):
                    key = _normalize_span(span)
                    if _meets_min_length(key):
                        occurrences[key].add(pos)
                        chapters_by_key[key].add(chapter_index)
                        first_span_by_key.setdefault(key, span)
        repeated = {key for key, positions in occurrences.items() if len(positions) >= 2}
        cross_chapter = {
            key: first_span_by_key[key] for key in repeated if len(chapters_by_key[key]) >= 2
        }
        return cls(repeated, cross_chapter_examples=cross_chapter)

    def clone(self) -> QuoteMemory:
        """Return an independent copy safe for wave-local updates inside a worker thread."""
        copy = QuoteMemory(
            set(self._repeated_keys),
            cross_chapter_examples=self._cross_chapter_examples,
        )
        copy._canonical = dict(self._canonical)
        return copy

    @property
    def repeated_count(self) -> int:
        return len(self._repeated_keys)

    def cross_chapter_spans(self, *, max_items: int = 15) -> list[str]:
        """Return raw source spans that recur across >=2 distinct chapters (for pre-rendering)."""
        return [
            span for key, span in self._cross_chapter_examples.items() if key not in self._canonical
        ][:max_items]

    def seed_canonical(self, source_span: str, target_rendering: str) -> None:
        """Pre-seed a canonical rendering (position (-1, -1)) before parallel waves start."""
        cleaned = (target_rendering or "").strip()
        stripped = (source_span or "").strip()
        if not cleaned or not stripped:
            return
        key = _normalize_span(stripped)
        if key in self._repeated_keys and key not in self._canonical:
            self._canonical[key] = ((-1, -1), stripped, cleaned)

    def observe(
        self,
        position: tuple[int, int],
        source: str,
        target: str,
        *,
        prefer_update: bool = False,
    ) -> None:
        """Record the earliest translated rendering for any repeated span present in ``source``."""
        cleaned_target = (target or "").strip()
        stripped_source = (source or "").strip()
        if not cleaned_target or not self._repeated_keys:
            return
        quoted_targets = [m.group(1).strip() for m in _QUOTED_SPAN_RE.finditer(cleaned_target)]
        for span in extract_candidate_spans(stripped_source):
            key = _normalize_span(span)
            if key not in self._repeated_keys:
                continue
            if span == stripped_source:
                stored_target = cleaned_target
            elif len(quoted_targets) == 1:
                stored_target = quoted_targets[0]
            elif len(quoted_targets) > 1:
                # Pick the longest quoted span (typically the lyric/inscription rather than short speech).
                stored_target = max(quoted_targets, key=len)
            else:
                continue
            existing = self._canonical.get(key)
            if (
                existing is None
                or position < existing[0]
                or (prefer_update and position == existing[0])
            ):
                self._canonical[key] = (position, span, stored_target)

    def exact_match(self, source: str) -> str | None:
        """Return a canonical full-paragraph target if ``source`` is an exact repeat of one."""
        stripped = source.strip()
        if not stripped or not self._repeated_keys:
            return None
        key = _normalize_span(stripped)
        entry = self._canonical.get(key)
        if entry is None:
            return None
        _, canonical_src, canonical_tgt = entry
        if _normalize_span(canonical_src) == key and len(stripped) <= 240:
            return canonical_tgt
        return None

    def references_for_batch(self, sources: Sequence[str]) -> list[tuple[str, str]]:
        """Return ``(source_span, established_target)`` pairs relevant to a batch."""
        if not self._canonical:
            return []
        seen: set[str] = set()
        refs: list[tuple[str, str]] = []
        for source in sources:
            for span in extract_candidate_spans(source):
                key = _normalize_span(span)
                if key in seen or key not in self._canonical:
                    continue
                seen.add(key)
                _, canonical_src, canonical_tgt = self._canonical[key]
                refs.append((canonical_src, canonical_tgt))
        return refs

    def format_prompt_block(self, sources: Sequence[str], *, max_items: int = 5) -> str:
        """Render compact verbatim callback guidance for any repeated quotes in ``sources``."""
        refs = self.references_for_batch(sources)[:max_items]
        if not refs:
            return ""
        lines = [
            "[Repeated quotation reference — reuse the established wording consistently when the same verse or quote recurs]"
        ]
        for src, tgt in refs:
            lines.append(f"- {src} => {tgt}")
        return "\n".join(lines)
