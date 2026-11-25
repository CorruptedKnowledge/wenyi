"""Deterministic post-translation repairs that hold for any book style.

Pure functions with zero model calls. Each rule must be grounded in the paragraph's own
source, never in one book's punctuation or naming conventions:
- ``repair_escaped_newlines`` turns a leaked literal ``\\n`` escape into a real line break,
  but only when the source paragraph has no literal ``\\n`` of its own, and never inside
  protected-object markers, Markdown code spans, LaTeX math spans or URLs.
- ``normalize_minority_names`` replaces renderings that the Book Bible probe observed for
  the same source term with the canonical glossary target, in cased scripts only, using
  word-boundary and collision guards and logging every substitution for auditability.

Quotation marks, dashes, speech tags and line layout are left to the model and to the
opt-in export-time ``output.quote_style``: deterministic rewrites of them corrupted correct
text in other languages, genres and house styles.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ...glossary.store import STATUS_OK, source_matches_text
from ...i18n.languages import glossary_generic_target_words
from ...models.glossary import TYPE_NAVIGATION, GlossaryTerm

# Matches protected-object transport tokens such as 〘object:1〙 or ⟦N0⟧.
_PROTECTED_MARKER_RE = re.compile(r"〘object:[A-Za-z0-9_.:-]+〙|⟦[^⟧]+⟧")
# Also shields fenced/inline Markdown code spans, LaTeX math spans, and URLs during hygiene.
_TYPOGRAPHY_PROTECTED_RE = re.compile(
    r"〘object:[A-Za-z0-9_.:-]+〙|⟦[^⟧]+⟧"
    r"|(?<!`)(`{3,})(?!`)[\s\S]*?(?<!`)\1(?!`)"
    r"|(?<!`)``(?!`)[^\n]+?(?<!`)``(?!`)"
    r"|(?<!`)`(?!`)[^`\n]+?(?<!`)`(?!`)"
    r"|(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$"
    r"|(?<!\\)\\\[[\s\S]*?(?<!\\)\\\]"
    r"|(?<!\\)\\\([\s\S]*?(?<!\\)\\\)"
    r"|(?<!\\)\$(?!\$)(?!\s)(?:\\.|[^$\\\n])+?(?<!\s)\$(?![$\d])"
    r"|https?://[^\s\"'“”‘’]+"
)
# LaTeX commands beginning with '\n' whose suffixes should never be mistaken for leaked newlines
# even when written outside math delimiters immediately after punctuation (e.g. "a,\nu", ".\nRightarrow").
_LATEX_N_CMD_SUFFIXES = (
    r"eq|e|eg|i|abla|approx|leq|geq|leqslant|geqslant|less|gtr|mid|parallel|vdash|Vdash|vDash|VDash|"
    r"cong|sim|subseteq|supseteq|subset|supset|rightarrow|leftarrow|leftrightarrow|"
    r"Rightarrow|Leftarrow|Leftrightarrow|earrow|warrow|exists|shortmid|shortparallel|"
    r"succ|prec|triangleleft|triangleright|otin|olimits|obreak|ewline|ewpage|ewcommand|"
    r"ewenvironment|ewtheorem|oindent|olinebreak|umberwithin|onumber|opagebreak|ormalfont|"
    r"ormalsize|otag|ocite"
)
# Bare '\nu' is protected unless followed by a hyphen/dot/letter (so "a,\nu-boats" still normalizes).
_LATEX_N_GUARD = rf"(?:(?:{_LATEX_N_CMD_SUFFIXES})(?![A-Za-z])|u(?![A-Za-z.\-]))"
# After a period '.', also protect relative paths like '.\new' and bare '.\not'.
_PERIOD_N_GUARD = rf"(?:(?:{_LATEX_N_CMD_SUFFIXES}|ew|ot)(?![A-Za-z])|u(?![A-Za-z.\-]))"
# Match standalone leaked literal '\n' line breaks without touching LaTeX commands
# ('\neq', '\ne', '\nu', '\nabla', '\nRightarrow'), escaped backslashes ('\\n'), or paths ('C:\new', '.\new').
_ESCAPED_NEWLINE_RE = re.compile(
    rf"(?<=[,;!?…—\"”'’】》」』])(?<!\\)(?:\\n)+(?=[A-Za-z])(?!{_LATEX_N_GUARD})"
    rf"|(?<=\.)(?<!\\)(?:\\n)+(?=[A-Za-z])(?!{_PERIOD_N_GUARD})"
    r"|(?<!\\)(?:\\n)+(?![A-Za-z])"
)


def _mask_markers(
    text: str, pattern: re.Pattern[str] = _PROTECTED_MARKER_RE
) -> tuple[str, list[str]]:
    saved: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        idx = len(saved)
        saved.append(match.group(0))
        return f"\ue000OBJ{idx}\ue001"

    return pattern.sub(_repl, text), saved


def _unmask_markers(text: str, saved: list[str]) -> str:
    for idx, marker in enumerate(saved):
        text = text.replace(f"\ue000OBJ{idx}\ue001", marker)
    return text


def repair_escaped_newlines(source: str, target: str) -> str:
    """Turn leaked literal ``\\n`` escapes in ``target`` into real line breaks.

    A literal backslash-n that the source paragraph also contains is content (a programming
    book explaining escape sequences, a Windows path), so the target is left unchanged.
    """
    if not target or "\\n" not in target or "\\n" in (source or ""):
        return target
    if target.lstrip().startswith("```"):
        return target
    masked, saved = _mask_markers(target, _TYPOGRAPHY_PROTECTED_RE)
    updated = _ESCAPED_NEWLINE_RE.sub(lambda m: "\n" * (len(m.group(0)) // 2), masked)
    if updated == masked:
        return target
    return _unmask_markers(updated, saved)


_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def _is_generic_english(phrase: str, generic_words: frozenset[str]) -> bool:
    if not generic_words:
        return False
    words = _WORD_RE.findall(phrase.casefold())
    if not words:
        return True
    return all(word.replace("’", "'") in generic_words for word in words)


def _latin_boundary_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![^\W\d_]){re.escape(phrase)}(?![^\W\d_])")


def _has_cased_letter(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def normalize_minority_names(
    pairs: Sequence[tuple[str, str]],
    terms: Sequence[GlossaryTerm],
    variants_by_source: Mapping[str, Sequence[str]] | None = None,
    *,
    target_lang: str = "en",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replace observed minority renderings of glossary names with their canonical target.

    Only variants supplied in ``variants_by_source`` (renderings the Book Bible probe saw
    for the same source term) are replaced, and only in cased scripts (Latin, Cyrillic,
    Greek), where word boundaries and capitalization separate a name from an ordinary
    word. Uncased scripts such as Chinese or Japanese have no reliable word boundary, so a
    variant can sit inside a different name (亚历山德 inside 亚历山德拉); they are left to
    the glossary prompt and Review. Protected markers, code spans, and LaTeX are masked.
    """
    generic_words = glossary_generic_target_words(target_lang)
    variants_map = variants_by_source or {}

    entity_terms = [t for t in terms if t.type != TYPE_NAVIGATION]
    all_canonical_lower = [t.target.strip().casefold() for t in entity_terms if t.target.strip()]

    def collides(variant: str, canonical: str) -> bool:
        """Never replace a variant that is part of another glossary entry's target."""
        v_low, c_low = variant.casefold(), canonical.casefold()
        return any(v_low in other and other != c_low for other in all_canonical_lower)

    eligible_terms: list[tuple[GlossaryTerm, list[tuple[str, re.Pattern[str]]]]] = []
    for term in entity_terms:
        if term.status != STATUS_OK:
            continue
        canonical = term.target.strip()
        if len(canonical) < 2 or not _has_cased_letter(canonical):
            continue
        if _is_generic_english(canonical, generic_words):
            continue

        raw_variants = [v.strip() for v in variants_map.get(term.source, ()) if v and v.strip()]
        safe_variants: list[tuple[str, re.Pattern[str]]] = []
        seen_v: set[str] = set()
        for variant in sorted(raw_variants, key=len, reverse=True):
            v_low = variant.casefold()
            if v_low == canonical.casefold() or v_low in seen_v:
                continue
            if len(variant) < 2 or not _has_cased_letter(variant):
                continue
            if _is_generic_english(variant, generic_words):
                continue
            if collides(variant, canonical):
                continue
            seen_v.add(v_low)
            safe_variants.append((variant, _latin_boundary_pattern(variant)))
        if safe_variants:
            eligible_terms.append((term, safe_variants))

    results: list[str] = []
    substitutions: list[dict[str, Any]] = []
    for idx, (source, target) in enumerate(pairs):
        updated = target
        if not updated or not source or not eligible_terms:
            results.append(updated)
            continue
        masked, saved = _mask_markers(updated, _TYPOGRAPHY_PROTECTED_RE)
        for term, safe_variants in eligible_terms:
            if not source_matches_text(term.source, source):
                continue
            canonical = term.target.strip()
            candidates = [variant for variant, var_pat in safe_variants if var_pat.search(masked)]
            if not candidates:
                continue
            # Match the canonical spelling alongside every variant in one non-overlapping
            # longest-first pass so a shorter variant (e.g. "Gao") never matches inside an
            # existing canonical occurrence (e.g. "Old Gao" -> never "Old Old Gao").
            ordered = sorted(dict.fromkeys([canonical, *candidates]), key=len, reverse=True)
            alternatives = [
                rf"(?i:{re.escape(v)})" if v.casefold() == canonical.casefold() else re.escape(v)
                for v in ordered
            ]
            combined_pat = re.compile(rf"(?<![^\W\d_])(?:{'|'.join(alternatives)})(?![^\W\d_])")
            matched_variants: list[str] = []
            canon_cf = canonical.casefold()

            def _replace(
                match: re.Match[str],
                _canon: str = canonical,
                _canon_cf: str = canon_cf,
            ) -> str:
                matched = match.group(0)
                if matched.casefold() == _canon_cf:
                    return matched
                if matched not in matched_variants:
                    matched_variants.append(matched)
                return _canon

            masked = combined_pat.sub(_replace, masked)
            for variant in matched_variants:
                substitutions.append(
                    {
                        "index": idx,
                        "source": term.source,
                        "from": variant,
                        "to": canonical,
                    }
                )
        results.append(_unmask_markers(masked, saved))
    return results, substitutions
