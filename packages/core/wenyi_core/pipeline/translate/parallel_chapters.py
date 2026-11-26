"""Concurrent chapter translation executor with a single ordered state writer.

Worker threads perform only model calls (translate + optional polish) over detached
chapter specs with a frozen glossary snapshot and deterministic positional source
lookbehind. The calling thread alone reads/writes Storage, aligns annotations/styles,
flushes usage, and publishes ``STATUS_DONE`` in deterministic chapter order.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from ...events import ProgressFn
from ...llm.retrying import is_resumable_provider_interrupt
from ...markup.contexts import annotation_contexts_for_segments
from ...models.document import Chapter, Segment
from ...models.glossary import GlossaryTerm
from ...storage.protocol import Storage
from ...storage.run_store import STATUS_DONE
from .babeldoc_context import BabeldocContext
from .context import render_recent_translations
from .post_normalize import repair_escaped_newlines
from .translation_batch import BatchPlan, BatchResult, TranslationBatchExecutor


@dataclass(frozen=True)
class _ChapterBatchSpec:
    batch_start: int
    segment_indices: tuple[int, ...]
    sources: tuple[str, ...]
    annotation_contexts: list[list[dict[str, str]]]
    next_source: str
    body_context: bool
    already_done: bool
    existing_targets: tuple[str | None, ...]


@dataclass(frozen=True)
class _ChapterSpec:
    chapter: int
    title: str
    chapter_digest: str
    opening_context: str
    style: str
    book_synopsis: str
    terms: tuple[GlossaryTerm, ...]
    rolling_context_tokens: int
    batches: tuple[_ChapterBatchSpec, ...]


@dataclass(frozen=True)
class _BatchOutcome:
    batch_start: int
    sources: tuple[str, ...]
    result: BatchResult


@dataclass(frozen=True)
class _ChapterOutcome:
    chapter: int
    batches: tuple[_BatchOutcome, ...]
    error: BaseException | None
    completed: bool


def _stops_run(exc: BaseException) -> bool:
    return not isinstance(exc, Exception) or is_resumable_provider_interrupt(exc)


def _source_lookbehind(prev_chapter: Chapter | None, budget_tokens: int) -> str:
    """Build deterministic opening context from the preceding chapter's raw source tail."""
    if prev_chapter is None or budget_tokens <= 0:
        return ""
    parts = [seg.source for seg in prev_chapter.text_segments if seg.source.strip()]
    if not parts:
        return ""
    tail = render_recent_translations(reversed(parts), budget_tokens)
    if not tail:
        return ""
    return f"[Preceding chapter source tail (reference only)]\n{tail}"


class ParallelChapterRunner:
    """Translate pending chapters concurrently with bounded workers and one ordered writer."""

    def __init__(
        self,
        batches: TranslationBatchExecutor,
        *,
        chapter_batches_fn: Callable[[Chapter], list[list[Segment]]],
        align_batch_fn: Callable[[Storage, Chapter, list[Segment], int, int], None],
        progress_label_fn: Callable[[str, int], str],
        flush_usage_fn: Callable[[Storage, str], None],
    ):
        self._batches = batches
        self._chapter_batches = chapter_batches_fn
        self._align_batch = align_batch_fn
        self._progress_label = progress_label_fn
        self._flush_usage = flush_usage_fn

    def run(
        self,
        targets: list[int],
        store: Storage,
        *,
        frozen_terms: list[GlossaryTerm],
        style: str,
        book_synopsis: str,
        annotation_context_registry: dict[str, Any] | None,
        chapter_concurrency: int,
        rolling_context_tokens: int,
        polish: bool,
        polish_mode: str,
        progress: ProgressFn | None,
        done: int,
        total: int,
    ) -> int:
        """Translate ``targets`` concurrently, applying and persisting each chapter on this thread."""
        manifest = store.load_manifest()
        all_indices = sorted(
            row["index"]
            for row in manifest.get("chapters", [])
            if isinstance(row.get("index"), int)
        )
        loaded: dict[int, Chapter] = {ci: store.load_chapter(ci) for ci in all_indices}

        specs: list[_ChapterSpec] = []
        chapters_by_id: dict[int, Chapter] = {}
        align_existing: dict[int, list[tuple[int, int]]] = {}

        for ci in targets:
            chapter = loaded[ci]
            text_segs = chapter.text_segments
            if not text_segs:
                store.set_chapter_status(ci, STATUS_DONE)
                store.log_event("chapter_skipped", chapter=ci, reason="empty")
                continue
            chapters_by_id[ci] = chapter
            digest = str(chapter.meta.get("source_digest") or "")
            policy = BabeldocContext.for_chapter(chapter)
            ann_contexts = annotation_contexts_for_segments(text_segs, annotation_context_registry)
            batches = self._chapter_batches(chapter)

            prev_indices = [idx for idx in all_indices if idx < ci]
            prev_chapter = loaded[prev_indices[-1]] if prev_indices else None
            opening_context = _source_lookbehind(prev_chapter, rolling_context_tokens)

            next_indices = [idx for idx in all_indices if idx > ci]
            next_ch_first = ""
            if next_indices:
                next_segs = loaded[next_indices[0]].text_segments
                if next_segs:
                    next_ch_first = next_segs[0].source

            batch_specs: list[_ChapterBatchSpec] = []
            seg_base = 0
            for b in batches:
                end = seg_base + len(b)
                body_ctx = policy is None or policy.body[seg_base]
                already_done = all(s.target is not None for s in b)
                if already_done:
                    align_existing.setdefault(ci, []).append((seg_base, len(b)))
                next_src = ""
                if body_ctx:
                    if end < len(text_segs):
                        next_src = text_segs[end].source
                    else:
                        next_src = next_ch_first
                batch_specs.append(
                    _ChapterBatchSpec(
                        batch_start=seg_base,
                        segment_indices=tuple(s.index for s in b),
                        sources=tuple(s.source for s in b),
                        annotation_contexts=ann_contexts[seg_base:end],
                        next_source=next_src,
                        body_context=body_ctx,
                        already_done=already_done,
                        existing_targets=tuple(s.target for s in b),
                    )
                )
                seg_base = end

            specs.append(
                _ChapterSpec(
                    chapter=ci,
                    title=chapter.title,
                    chapter_digest=digest,
                    opening_context=opening_context,
                    style=style,
                    book_synopsis=book_synopsis,
                    terms=tuple(frozen_terms),
                    rolling_context_tokens=rolling_context_tokens,
                    batches=tuple(batch_specs),
                )
            )

        if not specs:
            return done

        # Warm the DeepSeek KV prefix with the first pending chapter's first batch implicitly
        # by submitting bounded waves, keeping at most chapter_concurrency chapters in flight.
        workers = min(max(1, chapter_concurrency), len(specs))
        stop = threading.Event()
        next_pos = 0
        in_flight: dict[Future[_ChapterOutcome], int] = {}
        first_error: BaseException | None = None

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="wenyi-chapter-worker"
        ) as executor:
            try:
                while (next_pos < len(specs) and not stop.is_set()) or in_flight:
                    while not stop.is_set() and next_pos < len(specs) and len(in_flight) < workers:
                        spec = specs[next_pos]
                        ctx = contextvars.copy_context()
                        fut = executor.submit(
                            ctx.run,
                            self._run_chapter_worker,
                            spec,
                            polish,
                            polish_mode,
                            stop,
                        )
                        in_flight[fut] = spec.chapter
                        next_pos += 1
                    if not in_flight:
                        break
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    # Flush completed chapters in deterministic chapter-index order.
                    ready: list[_ChapterOutcome] = []
                    for fut in finished:
                        in_flight.pop(fut)
                        ready.append(fut.result())
                    ready.sort(key=lambda item: item.chapter)
                    for outcome in ready:
                        done = self._apply_chapter_outcome(
                            store,
                            chapters_by_id[outcome.chapter],
                            outcome,
                            align_existing.get(outcome.chapter, ()),
                            polish=polish,
                            progress=progress,
                            done=done,
                            total=total,
                        )
                        if outcome.error is not None:
                            stop.set()
                            if first_error is None:
                                first_error = outcome.error
            except BaseException:
                stop.set()
                for fut in in_flight:
                    fut.cancel()
                raise

        if first_error is not None:
            raise first_error
        return done

    def _run_chapter_worker(
        self,
        spec: _ChapterSpec,
        polish: bool,
        polish_mode: str,
        stop: threading.Event,
    ) -> _ChapterOutcome:
        """Execute one chapter's pending batches serially in a worker thread (model calls only)."""
        wave_drafts: list[str] = []
        outcomes: list[_BatchOutcome] = []
        error: BaseException | None = None
        stopped_early = False

        for bspec in spec.batches:
            if bspec.already_done:
                if bspec.body_context:
                    wave_drafts.extend(t or "" for t in bspec.existing_targets)
                continue
            if stop.is_set():
                stopped_early = True
                break

            if not bspec.body_context:
                ctx_text = ""
            elif wave_drafts:
                ctx_text = render_recent_translations(
                    reversed(wave_drafts), spec.rolling_context_tokens
                )
            else:
                ctx_text = spec.opening_context

            plan = BatchPlan(
                chapter=spec.chapter,
                start_index=bspec.batch_start,
                segment_indices=bspec.segment_indices,
                sources=bspec.sources,
                terms=spec.terms,
                context=ctx_text,
                style=spec.style,
                book_synopsis=spec.book_synopsis if bspec.body_context else "",
                chapter_digest=spec.chapter_digest if bspec.body_context else "",
                annotation_contexts=bspec.annotation_contexts,
                next_source=bspec.next_source,
                allow_empty_translations=False,
            )
            try:
                result = self._batches.execute(plan, polish=polish, polish_mode=polish_mode)
            except BaseException as exc:  # noqa: BLE001 - save completed batches before propagating.
                error = exc
                break

            cleaned = tuple(
                repair_escaped_newlines(src, tgt) for src, tgt in zip(bspec.sources, result.targets)
            )
            result = BatchResult(
                cleaned,
                result.before_polish,
                precision_key=result.precision_key,
                polish_failed=result.polish_failed,
                polish_empty_reverted=result.polish_empty_reverted,
                polish_interrupted=result.polish_interrupted,
            )
            outcomes.append(_BatchOutcome(bspec.batch_start, bspec.sources, result))
            if bspec.body_context:
                wave_drafts.extend(cleaned)
            if result.polish_interrupted is not None and _stops_run(result.polish_interrupted):
                error = result.polish_interrupted
                break

        completed = error is None and not stopped_early
        return _ChapterOutcome(spec.chapter, tuple(outcomes), error, completed)

    def _apply_chapter_outcome(
        self,
        store: Storage,
        chapter: Chapter,
        outcome: _ChapterOutcome,
        existing_batches: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        *,
        polish: bool,
        progress: ProgressFn | None,
        done: int,
        total: int,
    ) -> int:
        """Persist one worker's completed batches on the main writer thread."""
        ci = outcome.chapter
        text_segs = chapter.text_segments
        label = self._progress_label(chapter.title, ci)

        for start, count in existing_batches:
            self._align_batch(store, chapter, text_segs, start, count)

        for item in outcome.batches:
            count = len(item.sources)
            batch_segs = text_segs[item.batch_start : item.batch_start + count]
            for seg, target, before_polish in zip(
                batch_segs, item.result.targets, item.result.before_polish
            ):
                seg.target = target
                seg.target_before_polish = before_polish
            store.save_chapter(chapter)
            self._align_batch(store, chapter, text_segs, item.batch_start, count)
            store.log_event(
                "batch_translated",
                chapter=ci,
                start_index=item.batch_start,
                count=count,
                polished=polish and not item.result.polish_failed,
                translation_mode="standard",
                segments=[
                    {"index": item.batch_start + i, "source": s.source, "target": s.target}
                    for i, s in enumerate(batch_segs)
                ],
            )
            done += count
            if progress:
                progress(done, total, label)

        if outcome.completed:
            store.save_chapter_with_status(chapter, STATUS_DONE)
            store.log_event(
                "chapter_done",
                chapter=ci,
                title=chapter.title,
                segment_count=len(text_segs),
            )
        self._flush_usage(store, "chapter")
        return done
