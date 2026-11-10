"""Bound the glossary one model request carries, without reordering it."""

from __future__ import annotations

from functools import lru_cache

from ..ingest.tokens import count_tokens
from ..models.glossary import GlossaryTerm
from .store import GlossaryStore


@lru_cache(maxsize=1 << 16)
def _mapping_tokens(source: str, target: str, term_type: str, aliases: tuple[str, ...]) -> int:
    """Estimate one rendered mapping line; the budget is a size target, not an exact cap."""
    return count_tokens(f"- {source} → {target}({term_type}) {' '.join(aliases)}")


def prompt_terms(terms: list[GlossaryTerm], text: str, *, budget: int) -> list[GlossaryTerm]:
    """Return the terms a request about ``text`` carries, in insertion order.

    A glossary that fits ``budget`` tokens is sent whole. A larger one keeps its longest
    leading run that fits, then only the later terms whose source or alias occurs in
    ``text``. The leading run never changes as terms are appended, so the provider's
    prompt-prefix cache still covers it; what follows is bounded by ``text`` rather than
    by how far the book has progressed.
    """
    spent = 0
    for position, term in enumerate(terms):
        spent += _mapping_tokens(term.source, term.target, term.type, tuple(term.aliases))
        if spent > budget:
            return [*terms[:position], *GlossaryStore.terms_in(terms[position:], text)]
    return list(terms)
