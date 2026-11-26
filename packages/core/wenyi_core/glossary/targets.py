"""Decide whether a glossary target names someone or is only a pronoun or form of address."""

from __future__ import annotations

import re

from .store import normalize_match_text

NAME_TYPES = frozenset({"person", "place", "organization"})

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def is_name_shaped(target: str) -> bool:
    """Return True for one capitalized word longer than one letter, such as "Grace"."""
    words = _WORD_RE.findall(target)
    return len(words) == 1 and len(words[0]) > 1 and words[0][0].isupper()


def is_generic_target(
    target: str, generic_words: frozenset[str], *, term_type: str | None = None
) -> bool:
    """Return True when every word of ``target`` is a pronoun or title word of the target language.

    ``generic_words`` comes from the target-language profile ("I", "your son", "Your
    Excellency"). A person, place or organization whose target is one capitalized word, such as
    a character named "Grace" or a city named "King", is a name even when the word is on that
    list. Scripts the profile has no list for are never generic.
    """
    if not generic_words:
        return False
    words = _WORD_RE.findall(normalize_match_text(target))
    if not words or not all(w.replace("’", "'") in generic_words for w in words):
        return False
    return not (term_type in NAME_TYPES and is_name_shaped(target))
