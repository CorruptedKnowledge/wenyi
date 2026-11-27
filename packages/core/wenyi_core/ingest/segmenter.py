"""Dispatch document readers and split translation batches.
load_document selects a reader by extension and optionally splits long segments.
batch_segments partitions chapter segments into balanced, token-bounded batches and
requires equally sized model output for alignment. split_long_segments splits oversized
segments at sentences, marks continuations and lets the writer merge them into the
original paragraph/EPUB element.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable
from copy import deepcopy

from .epub_reader import read_epub
from .fb2_reader import read_fb2
from .html_reader import read_html
from .models import KIND_TEXT, Chapter, Document, Segment
from .pdf_reader import read_pdf
from .text_reader import read_text
from .tokens import count_tokens

# Common sentence-ending punctuation used for splitting long paragraphs.
_SENT_SPLIT = re.compile(r"(?<=[。．.!！？!?…\n])")


def _prefix_within_tokens(text: str, max_tokens: int) -> int:
    """Largest character end index with ``count_tokens(text[:end]) <= max_tokens``.

    Prefers a nearby whitespace cut when one exists inside the token-safe prefix so
    hard mid-word splits remain a last resort.
    """
    if not text:
        return 0
    if max_tokens <= 0:
        return 0
    if count_tokens(text) <= max_tokens:
        return len(text)

    low, high = 1, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    end = low
    # Prefer breaking on whitespace inside the safe prefix when the remainder is long.
    for sep in (" ", "\t", "\n"):
        cut = text.rfind(sep, 0, end + 1)
        if cut > 0:
            return cut
    return end


def _split_oversized_sentence(text: str, max_tokens: int) -> list[str]:
    """Split an oversized sentence within the token budget; hard-split only as a fallback."""
    chunks: list[str] = []
    rest = text
    while rest and count_tokens(rest) > max_tokens:
        cut = _prefix_within_tokens(rest, max_tokens)
        if cut <= 0:
            # A single token wider than the budget cannot shrink further; emit one codepoint.
            cut = max(1, min(len(rest), 1))
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def _split_text(text: str, max_tokens: int) -> list[str]:
    """Greedily group sentences by token length, falling back for oversized sentences."""
    chunks: list[str] = []
    cur = ""
    for part in _SENT_SPLIT.split(text):
        if not part:
            continue
        part_tokens = count_tokens(part)
        if part_tokens > max_tokens:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_oversized_sentence(part, max_tokens))
            continue
        if cur and count_tokens(cur + part) > max_tokens:
            chunks.append(cur)
            cur = ""
        cur += part
    if cur:
        chunks.append(cur)
    return chunks or [text]


def split_long_segments(chapters: list[Chapter], max_tokens: int) -> None:
    """Split oversized segments in place; mark continuations cont=True without independent
    anchors.
    """
    if not max_tokens or max_tokens <= 0:
        return
    for ch in chapters:
        new_segs: list[Segment] = []
        idx = 0
        for s in ch.segments:
            if count_tokens(s.source) <= max_tokens:
                s.index = idx
                new_segs.append(s)
                idx += 1
                continue
            for k, piece in enumerate(_split_text(s.source, max_tokens)):
                if k == 0:
                    new_segs.append(
                        Segment(
                            index=idx,
                            source=piece,
                            kind=s.kind,
                            anchor=s.anchor,
                            resource_href=s.resource_href,
                            cont=False,
                            meta=deepcopy(s.meta),
                        )
                    )
                else:  # Merge continuations back into the first segment; they have no independent anchor.
                    new_segs.append(
                        Segment(
                            index=idx,
                            source=piece,
                            kind=KIND_TEXT,
                            anchor=None,
                            resource_href=s.resource_href,
                            cont=True,
                        )
                    )
                idx += 1
        ch.segments = new_segs


def load_document(
    path: str,
    source_lang: str,
    target_lang: str,
    split_segments: int = 0,
    *,
    cache_dir: str | None = None,
    source_hash: str | None = None,
    pdf_backend: str = "mineru",
    babeldoc_bridge_url: str = "http://127.0.0.1:8765",
    babeldoc_pages: str | None = None,
    babeldoc_timeout: float = 600.0,
    mineru_token_resolver: Callable[[], str | None] | None = None,
) -> Document:
    """Dispatch by extension; resolve MinerU credentials only for uncached PDF conversion."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".epub":
        doc = read_epub(path, source_lang, target_lang)
    elif ext in (".md", ".markdown", ".txt", ".text"):
        doc = read_text(path, source_lang, target_lang)
    elif ext == ".fb2":
        doc = read_fb2(path, source_lang, target_lang)
    elif ext in (".html", ".htm", ".xhtml"):
        doc = read_html(path, source_lang, target_lang)
    elif ext == ".pdf":
        if cache_dir is None:
            raise ValueError("PDF input requires a run-state cache directory")
        if pdf_backend == "babeldoc":
            from .pdf_babeldoc import read_pdf_babeldoc

            doc = read_pdf_babeldoc(
                path,
                source_lang,
                target_lang,
                bridge_url=babeldoc_bridge_url,
                pages=babeldoc_pages,
                cache_dir=cache_dir,
                timeout=babeldoc_timeout,
            )
        else:
            doc = read_pdf(
                path,
                source_lang,
                target_lang,
                cache_dir=cache_dir,
                source_hash=source_hash,
                mineru_token_resolver=mineru_token_resolver,
            )
    elif ext == ".docx":
        from .docx_reader import read_docx

        doc = read_docx(path, source_lang, target_lang)
    else:
        raise ValueError(
            f"Unsupported format: {ext} (supported: .epub / .txt / .md / .fb2 / .html / .xhtml / .pdf / .docx)"
        )

    # BabelDOC IDs are tied to layout; never split those paragraphs by token budget.
    if split_segments and split_segments > 0 and not (doc.meta or {}).get("babeldoc"):
        split_long_segments(doc.chapters, split_segments)
    return doc


def _greedy_batch_count(weights: list[int], max_tokens: int) -> int:
    """Return the minimum number of contiguous batches needed under ``max_tokens``."""
    batches = 1
    cur = 0
    for weight in weights:
        if cur + weight > max_tokens:
            batches += 1
            cur = weight
        else:
            cur += weight
    return batches


def _min_bounded_batches(prefix: list[int], m: int, floor: int, cap: int) -> list[int]:
    """Return ``min_b`` where ``min_b[i]`` is the minimum number of non-empty
    contiguous batches for ``run[i:m]`` whose weights all lie in ``[floor, cap]``
    (or ``m + 1`` when impossible). Runs in O(m) time and space via a monotone deque.
    """
    inf = m + 1
    min_b = [inf] * (m + 1)
    min_b[m] = 0
    dq: deque[int] = deque()
    add_j = m
    hi_j = m
    for i in range(m - 1, -1, -1):
        base = prefix[i]
        min_pref = base + floor
        max_pref = base + cap
        while hi_j > i and prefix[hi_j] > max_pref:
            hi_j -= 1
        while dq and dq[0] > hi_j:
            dq.popleft()
        while add_j >= i + 1 and prefix[add_j] >= min_pref:
            if add_j <= hi_j and min_b[add_j] < inf:
                val = min_b[add_j]
                while dq and min_b[dq[-1]] >= val:
                    dq.pop()
                dq.append(add_j)
            add_j -= 1
        if dq:
            min_b[i] = 1 + min_b[dq[0]]
    return min_b


def _partition_balanced_run(
    run: list[Segment], weights: list[int], max_tokens: int
) -> list[list[Segment]]:
    """Split a run where every segment fits ``max_tokens`` into balanced batches."""
    m = len(run)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    total = prefix[m]
    if total <= max_tokens:
        return [run]

    k = _greedy_batch_count(weights, max_tokens)
    if k == 1:
        return [run]
    if k == m:
        return [[segment] for segment in run]

    lo = max(max(weights), (total + k - 1) // k)
    hi = max_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if _greedy_batch_count(weights, mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    # Maximize the smallest batch weight under fixed (k, cap) in O(m log cap) time and
    # O(m) space. Because k is minimal under cap, no [floor, cap] partition has < k
    # batches, so min_b[0] <= k iff a valid k-batch partition exists (and every valid
    # step satisfies min_b[end] == remaining - 1).
    best_floor = 0
    best_min_b = _min_bounded_batches(prefix, m, 0, cap)
    fl_lo, fl_hi = 1, total // k
    while fl_lo <= fl_hi:
        mid = (fl_lo + fl_hi) // 2
        cand = _min_bounded_batches(prefix, m, mid, cap)
        if cand[0] <= k:
            best_floor = mid
            best_min_b = cand
            fl_lo = mid + 1
        else:
            fl_hi = mid - 1

    batches: list[list[Segment]] = []
    start = 0
    for remaining in range(k, 0, -1):
        if remaining == 1:
            batches.append(run[start:m])
            break
        rem_tokens = total - prefix[start]
        rem_count = m - start
        cut_lo = max(start + 1, bisect_left(prefix, prefix[start] + best_floor, start + 1, m + 1))
        cap_hi = bisect_right(prefix, prefix[start] + cap, start + 1, m + 1) - 1
        cut_hi = min(cap_hi, m - (remaining - 1))
        need_rem = remaining - 1
        best_end = min(
            (end for end in range(cut_lo, cut_hi + 1) if best_min_b[end] == need_rem),
            key=lambda end: (
                abs((prefix[end] - prefix[start]) * remaining - rem_tokens),
                abs((end - start) * remaining - rem_count),
                end,
            ),
        )
        batches.append(run[start:best_end])
        start = best_end
    return batches


def batch_segments(segments: list[Segment], max_tokens: int) -> list[list[Segment]]:
    """Group segments into balanced batches within a shared token budget (cl100k_base).

    Preserves source order and the minimum batch count needed under ``max_tokens``.
    Any single segment larger than ``max_tokens`` stays in its own batch; intervening
    runs are partitioned into roughly equal-weight batches to avoid tiny orphan tails.
    """
    if not segments:
        return []
    weights = [count_tokens(s.source) for s in segments]
    if max_tokens <= 0:
        batches: list[list[Segment]] = []
        cur: list[Segment] = []
        cur_tokens = 0
        for segment, weight in zip(segments, weights):
            if cur and cur_tokens + weight > max_tokens:
                batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(segment)
            cur_tokens += weight
        if cur:
            batches.append(cur)
        return batches

    batches = []
    run: list[Segment] = []
    run_weights: list[int] = []
    for segment, weight in zip(segments, weights):
        if weight > max_tokens:
            if run:
                batches.extend(_partition_balanced_run(run, run_weights, max_tokens))
                run, run_weights = [], []
            batches.append([segment])
        else:
            run.append(segment)
            run_weights.append(weight)
    if run:
        batches.extend(_partition_balanced_run(run, run_weights, max_tokens))
    return batches
