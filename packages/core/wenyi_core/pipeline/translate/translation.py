"""Translation batches, resume, polishing, rolling context, glossary extraction and titles.
Use the prepared book synopsis while the caller holds the book lock. Process chapters and
batches serially. For each batch: translate, persist targets, align/persist annotations,
update context, append the batch event, extract/checkpoint glossary terms, update history,
then proceed.
At chapter end, perform fallback glossary extraction and publish model text plus done
through save_chapter_with_status. Normalize punctuation only in export copies. If targets
exist without a glossary checkpoint, extract missing terms without retranslating or
overwriting targets.
"""

from __future__ import annotations

import contextvars
import re
import threading
from collections.abc import Callable, Iterator
from concurrent import futures
from concurrent.futures import Future
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...agents.glossary_extractor import TranslatedSegmentEvidence
from ...agents.prompts import render_glossary
from ...config import PipelineConfig
from ...events import ProgressFn
from ...glossary.prompt_scope import prompt_terms
from ...glossary.store import GlossaryStore
from ...models.document import Chapter, Segment
from ...models.glossary import GlossaryTerm
from ...storage.protocol import Storage
from ...storage.run_store import STATUS_DONE
from ..annotate.docx_styles import DocxStyleService
from .babeldoc_context import BabeldocContext
from .context import render_recent_translations
from .precision import PrecisionBatchExecutor, PrecisionError
from .terminology import TerminologyService
from .title_translation import TitleTranslationService
from .translation_batch import BatchPlan, BatchResult, TranslationBatchExecutor, resume_batches

if TYPE_CHECKING:
    from ..annotate.annotations import AnnotationService
    from ..run.runtime import PipelineRuntime


def _is_mineru_pdf(manifest: dict[str, Any]) -> bool:
    """True for MinerU PDF state (fmt=pdf without BabelDOC markers)."""
    if manifest.get("fmt") != "pdf":
        return False
    raw_meta = manifest.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    return not bool(meta.get("babeldoc")) and meta.get("pdf_export") != "babeldoc"


class TranslationService:
    """Domain service for body translation, glossary extraction and chapter/TOC titles."""

    def __init__(self, runtime: PipelineRuntime, annotations: AnnotationService):
        self._runtime = runtime
        self._annotations = annotations
        self._docx_styles = DocxStyleService(runtime)
        self._titles = TitleTranslationService(runtime.title_translator)
        self._batches = TranslationBatchExecutor(runtime.translator, runtime.polisher)
        self._precision = PrecisionBatchExecutor(runtime.client, runtime.config)

    def run(
        self,
        store: Storage,
        *,
        book_synopsis: str,
        only_chapter: int | None = None,
        progress: ProgressFn | None = None,
    ) -> Storage:
        """Translate chapters serially and persist usage/progress under the book lock.
        The caller restores languages, validates only_chapter and prepares the synopsis.
        This method performs body and title translation with restored context.
        """
        self._runtime.require_terminology_context_access()
        PipelineConfig.model_validate(self._runtime.config.pipeline.model_dump())
        manifest = store.load_manifest()
        glossary = store
        style = self._runtime.analyzer.style_brief(store.load_analysis() or {})
        allow_empty_translations = _is_mineru_pdf(manifest)

        if only_chapter is not None:
            targets = [only_chapter]
            progress_chapters = targets
        else:
            targets = store.pending_chapters()
            progress_chapters = [chapter["index"] for chapter in manifest.get("chapters", [])]

        total, done = self.progress_counts(store, progress_chapters)
        translation_history, source_corpus = self.load_translation_inputs(store)
        annotation_context_registry = store.load_annotation_contexts()
        policy = self._runtime.config.language_policy("translation")
        terminology = (
            TerminologyService.for_store(
                store,
                self._runtime.terminology_agent,
                checkpoint_usage=lambda: self._runtime.usage.flush(store, scope="terminology"),
            )
            if policy.enabled("terminology.context")
            else None
        )
        store.log_event(
            "translate_run_started",
            only_chapter=only_chapter,
            chapters=targets,
            total_segments=total,
            allow_empty_translations=allow_empty_translations,
        )
        lookahead = self._lookahead(terminology)
        try:
            for position, ci in enumerate(targets):
                done = self.translate_chapter(
                    ci,
                    store,
                    glossary,
                    style,
                    book_synopsis,
                    translation_history=translation_history,
                    source_corpus=source_corpus,
                    annotation_context_registry=annotation_context_registry,
                    progress=progress,
                    done=done,
                    total=total,
                    allow_empty_translations=allow_empty_translations,
                    terminology=terminology,
                    lookahead=lookahead,
                    following=targets[position + 1] if position + 1 < len(targets) else None,
                )
                self._runtime.usage.flush(store, scope="chapter")
            # Translate chapter/TOC titles after the body; keep the original book title and use glossary names.
            if not store.pending_chapters():
                self._titles.run(store, glossary, progress=progress)
                self._annotations.repair_fallbacks(store, progress=progress)
        finally:
            if lookahead is not None:
                lookahead.drain()
            self._runtime.usage.flush(store, scope="translate")
        if progress and total:
            progress(total, total, "Translation complete")
        store.log_event("translate_run_finished", total_segments=total)
        return store

    @staticmethod
    def load_translation_inputs(
        store: Storage,
    ) -> tuple[dict[tuple[int, int], TranslatedSegmentEvidence], str]:
        """Read chapters once to rebuild translated-history indices and concatenate the source
        corpus.
        """
        history: dict[tuple[int, int], TranslatedSegmentEvidence] = {}
        source_parts: list[str] = []
        manifest = store.load_manifest()
        chapter_indices = sorted(
            chapter["index"]
            for chapter in manifest.get("chapters", [])
            if isinstance(chapter.get("index"), int)
        )
        for chapter_index in chapter_indices:
            chapter = store.load_chapter(chapter_index)
            segments = chapter.text_segments
            source_parts.extend(segment.source for segment in segments)
            TranslationService.update_translation_history(history, chapter_index, 0, segments)
        return history, "\n".join(source_parts)

    @staticmethod
    def update_translation_history(
        history: dict[tuple[int, int], TranslatedSegmentEvidence],
        chapter: int,
        start_index: int,
        segments: list[Segment],
    ) -> None:
        """Update the in-memory location index with the latest source/target batch."""
        for offset, segment in enumerate(segments):
            target = (segment.target or "").strip()
            if not target:
                continue
            segment_index = start_index + offset
            history[(chapter, segment_index)] = TranslatedSegmentEvidence(
                chapter=chapter,
                segment=segment_index,
                source=segment.source,
                target=target,
            )

    def progress_counts(self, store: Storage, chapter_indices: list[int]) -> tuple[int, int]:
        """Compute progress from batch checkpoints, starting resume at completed translation
        counts.
        Count a batch as done only when every target is not None (blank ``""`` counts). Counting
        partial batches early would duplicate completion counts if the batch reruns.
        """
        total = 0
        done = 0
        for ci in chapter_indices:
            chapter = store.load_chapter(ci)
            total += len(chapter.text_segments)
            for batch in self._chapter_batches(chapter):
                if all(segment.target is not None for segment in batch):
                    done += len(batch)
        return total, done

    def _chapter_batches(self, chapter: Chapter) -> list[list[Segment]]:
        """Apply source-specific boundaries without changing original text positions."""
        segments = chapter.text_segments
        policy = BabeldocContext.for_chapter(chapter)
        groups = policy.groups(segments) if policy is not None else [segments]
        return [
            batch
            for group in groups
            for batch in resume_batches(group, self._runtime.config.segment.max_tokens_per_batch)
        ]

    @staticmethod
    def _glossary_checkpoint_covers(checkpoints: set[str], start: int, count: int) -> bool:
        """Reuse completed extraction when BabelDOC role boundaries split an older batch."""
        ranges = []
        for key in checkpoints:
            match = re.fullmatch(r"(\d+):([1-9]\d*)", key)
            if match is not None:
                saved_start, saved_count = map(int, match.groups())
                ranges.append((saved_start, saved_start + saved_count))
        cursor = start
        for saved_start, saved_end in sorted(ranges):
            if saved_start <= cursor < saved_end:
                cursor = saved_end
                if cursor >= start + count:
                    return True
        return False

    def translate_chapter(
        self,
        ci: int,
        store: Storage,
        glossary: Storage | GlossaryStore,
        style: str,
        book_synopsis: str = "",
        *,
        translation_history: dict[tuple[int, int], TranslatedSegmentEvidence],
        source_corpus: str,
        annotation_context_registry: dict[str, Any] | None,
        progress: ProgressFn | None = None,
        done: int = 0,
        total: int = 0,
        allow_empty_translations: bool = False,
        terminology: TerminologyService | None = None,
        lookahead: _BatchLookahead | None = None,
        following: int | None = None,
    ) -> int:
        """Translate, polish, extract and persist one chapter; return the updated
        completed-paragraph count.

        With ``lookahead``, a batch is translated while the previous batch's terms are
        extracted, and the first batch of ``following`` while this chapter's last extractions
        run. Early results are used only when the model would see the same request.
        """
        with store.state_lock():
            chapter = store.load_chapter(ci)
            self._precision.recover_publications(store, chapter)
        analysis = store.load_analysis()
        text_segs = chapter.text_segments
        if not text_segs:
            store.set_chapter_status(ci, STATUS_DONE)
            store.log_event("chapter_skipped", chapter=ci, reason="empty")
            return done
        chapter_digest = chapter.meta.get("source_digest", "")
        context_policy = BabeldocContext.for_chapter(chapter)
        annotation_contexts = self._annotations.annotation_contexts_for_segments(
            text_segs,
            annotation_context_registry,
        )

        batches = self._chapter_batches(chapter)
        label = self.chapter_progress_label(chapter.title, ci)
        # Preparation often ends with a parsing label, but resume may first restore glossary terms.
        # Refresh at chapter start so the whole model call is not incorrectly labeled as source parsing.
        if progress:
            progress(done, total, label)
        glossary_checkpoints = store.completed_batch_glossary_keys(ci)
        # Read the glossary snapshot in stable insertion order at chapter start.
        # Refresh lazily only if the glossary may have changed and another batch needs translation.
        # Fully checkpointed skips neither extract nor refresh. Saved translations lacking extraction
        # still extract and mark the snapshot stale, preserving resume completeness without redundant reads.
        term_snapshot = glossary.all_terms()
        term_snapshot_stale = False

        # Process batches serially, deriving context from the latest formal targets.
        # This preserves pronoun, term and voice continuity between batches within a chapter.
        # Skip saved complete batches on resume and reconstruct context without retranslating them.
        seg_base = 0  # Chapter-local index of this batch's first paragraph, used to map local issue indices.
        owns_lookahead = lookahead is None
        if owns_lookahead:
            lookahead = self._lookahead(terminology)
        speculative = lookahead.take_carry(ci) if lookahead is not None else None
        try:
            for position, b in enumerate(batches):
                batch_start = seg_base
                glossary_key = store.batch_glossary_key(batch_start, len(b))
                if all(s.target is not None for s in b):
                    # Reuse a batch translated at this position/context, rebuild rolling context and skip it.
                    # Blank "" is a completed MinerU allowance; only None means not yet translated.
                    self._align_batch_context(store, chapter, batch_start, len(b))
                    if glossary_key in glossary_checkpoints or (
                        context_policy is not None
                        and self._glossary_checkpoint_covers(
                            glossary_checkpoints, batch_start, len(b)
                        )
                    ):
                        summary = {
                            "inserted": 0,
                            "conflict": 0,
                            "unchanged": 0,
                            "updated": 0,
                            "skipped": 1,
                        }
                    else:
                        # Targets exist but extraction checkpoint is missing: extract and store terms now.
                        summary = self.extract_batch_glossary(
                            glossary,
                            store,
                            ci,
                            batch_start,
                            b,
                            translation_history,
                            source_corpus,
                            terminology=terminology,
                        )
                        glossary_checkpoints.add(glossary_key)
                        term_snapshot_stale = True
                    store.log_event(
                        "batch_skipped",
                        chapter=ci,
                        start_index=batch_start,
                        count=len(b),
                        reason="already_translated",
                        glossary_extraction=summary,
                        segments=[
                            {"index": seg_base + i, "source": s.source, "target": s.target}
                            for i, s in enumerate(b)
                        ],
                    )
                    seg_base += len(b)
                    if progress:
                        progress(done, total, label)
                    continue

                if terminology is not None:
                    terminology.prepare_batch(
                        ci,
                        b,
                        extractor=self._runtime.extractor,
                        history=translation_history.values(),
                        before=(ci, batch_start),
                    )
                    term_snapshot_stale = True
                if term_snapshot_stale:
                    term_snapshot = glossary.all_terms()
                    term_snapshot_stale = False
                plan = self._capture_plan(
                    store,
                    chapter,
                    batch_start,
                    b,
                    term_snapshot,
                    analysis,
                    style,
                    book_synopsis,
                    chapter_digest,
                    annotation_contexts,
                    context_policy,
                    allow_empty_translations=allow_empty_translations,
                )
                if self._runtime.config.pipeline.translation_mode == "three_draft_synthesis":

                    def checkpoint_precision_usage() -> None:
                        self._runtime.usage.flush(store, scope="precision")

                    result = self._precision.execute(
                        plan,
                        store,
                        checkpoint=checkpoint_precision_usage,
                        progress=(
                            (lambda phase: progress(done, total, f"{label} · {phase}"))
                            if progress
                            else None
                        ),
                    )
                    chapter = self.save_precision_batch(store, plan, result)
                    text_segs = chapter.text_segments
                    b = text_segs[batch_start : batch_start + len(b)]
                    self._precision.mark_published(store, result)
                else:
                    result = None
                    if speculative is not None:
                        accepted = self._same_request(speculative[0], plan)
                        store.log_event(
                            "batch_lookahead",
                            chapter=ci,
                            start_index=batch_start,
                            count=len(b),
                            accepted=accepted,
                        )
                        if accepted:
                            result = speculative[1].result()
                        speculative = None
                    if result is None:
                        result = self._batches.execute(
                            plan, polish=self._runtime.config.pipeline.polish
                        )
                    for segment, target, before_polish in zip(
                        b, result.targets, result.before_polish
                    ):
                        segment.target = target
                        segment.target_before_polish = before_polish
                    # Persist translations incrementally so interruption resumes after this batch.
                    store.save_chapter(chapter)
                self._align_batch_context(store, chapter, batch_start, len(b))
                store.log_event(
                    "batch_translated",
                    chapter=ci,
                    start_index=batch_start,
                    count=len(b),
                    polished=self._runtime.config.pipeline.polish,
                    translation_mode=self._runtime.config.pipeline.translation_mode,
                    segments=[
                        {
                            "index": batch_start + i,
                            "source": s.source,
                            "target": s.target,
                        }
                        for i, s in enumerate(b)
                    ],
                )
                done += len(b)
                seg_base += len(b)
                if progress:
                    progress(done, total, label)
                if lookahead is not None and position + 1 < len(batches):
                    upcoming = batches[position + 1]
                    if all(segment.target is None for segment in upcoming):
                        # Translate the next batch while this batch's terms are extracted. The
                        # result is used only if the plan built after extraction is identical,
                        # so every accepted request matches what the serial order would send.
                        if term_snapshot_stale:
                            term_snapshot = glossary.all_terms()
                            term_snapshot_stale = False
                        following_plan = self._capture_plan(
                            store,
                            chapter,
                            seg_base,
                            upcoming,
                            term_snapshot,
                            analysis,
                            style,
                            book_synopsis,
                            chapter_digest,
                            annotation_contexts,
                            context_policy,
                            allow_empty_translations=allow_empty_translations,
                        )
                        speculative = (following_plan, lookahead.submit(following_plan))
                elif lookahead is not None and following is not None:
                    # Last batch: start the next chapter while this one finishes extraction.
                    if term_snapshot_stale:
                        term_snapshot = glossary.all_terms()
                        term_snapshot_stale = False
                    first = self._first_plan(
                        store,
                        following,
                        term_snapshot,
                        analysis,
                        style,
                        book_synopsis,
                        annotation_context_registry,
                        allow_empty_translations=allow_empty_translations,
                    )
                    if first is not None:
                        lookahead.carry(following, first)
                # Persist targets before glossary extraction so interruption cannot leave terms ahead of text.
                self.extract_batch_glossary(
                    glossary,
                    store,
                    ci,
                    batch_start,
                    b,
                    translation_history,
                    source_corpus,
                    terminology=terminology,
                )
                self.update_translation_history(translation_history, ci, batch_start, b)
                glossary_checkpoints.add(glossary_key)
                # The glossary may have changed; refresh before the next real translation, not after the final batch.
                term_snapshot_stale = True
        finally:
            if lookahead is not None and owns_lookahead:
                # Never leave a request running past the chapter, including on interruption.
                lookahead.drain()

        # Keep chapter-wide extraction as a fallback for address, speech and fixed expressions needing context.
        # Final review reads the stable glossary after the entire book finishes translating.
        src_text = "\n".join(s.source for s in text_segs)
        tgt_text = "\n".join(s.target or "" for s in text_segs)
        chapter_glossary_summary = self._runtime.extractor.extract_and_store(
            glossary,
            src_text,
            tgt_text,
            ci,
            history=translation_history.values(),
            before=(ci, len(text_segs)),
            source_corpus=source_corpus,
            **({"admit": terminology.admit} if terminology is not None else {}),
        )
        store.log_event(
            "chapter_glossary_extracted",
            chapter=ci,
            summary=chapter_glossary_summary,
        )

        store.save_chapter_with_status(chapter, STATUS_DONE)
        store.log_event(
            "chapter_done",
            chapter=ci,
            title=chapter.title,
            segment_count=len(text_segs),
        )
        return done

    def _lookahead(self, terminology: TerminologyService | None) -> _BatchLookahead | None:
        """Overlap translation with extraction only where batch plans are comparable."""
        if self._runtime.config.pipeline.translation_mode != "standard" or terminology is not None:
            return None
        polish = self._runtime.config.pipeline.polish
        return _BatchLookahead(lambda plan: self._batches.execute(plan, polish=polish))

    def _first_plan(
        self,
        store: Storage,
        ci: int,
        term_snapshot: list[GlossaryTerm],
        analysis: dict[str, Any] | None,
        style: str,
        book_synopsis: str,
        annotation_context_registry: dict[str, Any] | None,
        *,
        allow_empty_translations: bool,
    ) -> BatchPlan | None:
        """Plan the first batch of chapter ``ci`` when it is still untranslated."""
        chapter = store.load_chapter(ci)
        text_segs = chapter.text_segments
        batches = self._chapter_batches(chapter)
        if not text_segs or not batches or any(s.target is not None for s in batches[0]):
            return None
        return self._capture_plan(
            store,
            chapter,
            0,
            batches[0],
            term_snapshot,
            analysis,
            style,
            book_synopsis,
            chapter.meta.get("source_digest", ""),
            self._annotations.annotation_contexts_for_segments(
                text_segs, annotation_context_registry
            ),
            BabeldocContext.for_chapter(chapter),
            allow_empty_translations=allow_empty_translations,
        )

    def _same_request(self, early: BatchPlan, current: BatchPlan) -> bool:
        """Return whether both plans make the model see identical prompts.

        Terms are compared as the translator and polisher render them, so a status or note
        change that never reaches the prompt does not discard the early result.
        """
        if replace(early, terms=()) != replace(current, terms=()):
            return False
        return self._rendered_terms(early) == self._rendered_terms(current)

    def _rendered_terms(self, plan: BatchPlan) -> str:
        translator = self._runtime.translator
        return render_glossary(
            list(plan.terms),
            source_lang=translator.src,
            genre=translator.language_policy.context.genre,
            note_source="\n".join(plan.sources)
            if self._runtime.config.pipeline.terminology_context
            else None,
        )

    def _capture_plan(
        self,
        store: Storage,
        chapter: Chapter,
        batch_start: int,
        batch: list[Segment],
        term_snapshot: list[GlossaryTerm],
        analysis: dict[str, Any] | None,
        style: str,
        book_synopsis: str,
        chapter_digest: str,
        annotation_contexts: list[list[dict[str, str]]],
        context_policy: BabeldocContext | None,
        *,
        allow_empty_translations: bool,
    ) -> BatchPlan:
        """Build the complete request inputs for one batch from the current formal state."""
        batch_style = (
            self._runtime.analyzer.style_brief(
                analysis,
                terms=term_snapshot if self._runtime.config.pipeline.terminology_context else None,
            )
            if analysis is not None
            else style
        )
        next_index = batch_start + len(batch)
        body_context = context_policy is None or context_policy.body[batch_start]
        ctx_text = (
            self.recent_context(store, chapter, batch_start, context_policy) if body_context else ""
        )
        next_source = (
            self.following_source(store, chapter, next_index, context_policy)
            if body_context
            else ""
        )
        return BatchPlan.capture(
            chapter.index,
            batch_start,
            batch,
            # Send what this batch can use, not everything the book has collected so far.
            prompt_terms(
                term_snapshot,
                "\n".join(segment.source for segment in batch),
                budget=self._runtime.config.pipeline.glossary_prompt_tokens,
            ),
            ctx_text,
            batch_style,
            book_synopsis if body_context else "",
            chapter_digest if body_context else "",
            annotation_contexts[batch_start : batch_start + len(batch)],
            next_source,
            allow_empty_translations=allow_empty_translations,
        )

    def _align_batch_context(
        self,
        store: Storage,
        chapter: Chapter,
        start: int,
        count: int,
    ) -> None:
        """Align formal targets before subsequent batches derive their context.

        Alignment handles touched logical paragraphs in source order, waiting for their
        final continuation. It can also update targets from earlier batches.
        """
        self._annotations.align_annotations_after_batch(chapter.index, chapter, start, count, store)
        self._docx_styles.align_styles_after_batch(chapter.index, chapter, start, count, store)

    @staticmethod
    def save_precision_batch(store: Storage, plan: BatchPlan, result: BatchResult) -> Chapter:
        """Merge accepted pending segments into a fresh chapter without losing manual edits."""
        with store.state_lock():
            publication = PrecisionBatchExecutor.prepared_publication(store, result)
            if (
                publication.get("segment_indices") != list(plan.segment_indices)
                or len(result.targets) != len(plan.sources)
                or len(result.before_polish) != len(plan.sources)
            ):
                raise PrecisionError("The precision publication index is not aligned to its batch")
            chapter = store.load_chapter(plan.chapter)
            segments = chapter.text_segments[
                plan.start_index : plan.start_index + len(plan.sources)
            ]
            if (
                tuple(segment.index for segment in segments) != plan.segment_indices
                or tuple(segment.source for segment in segments) != plan.sources
            ):
                raise PrecisionError("The precision batch source changed before publication")
            for segment, target, before_polish in zip(
                segments, result.targets, result.before_polish
            ):
                if segment.target is not None and (
                    segment.target != target or segment.target_before_polish != before_polish
                ):
                    raise PrecisionError(
                        "The precision batch was edited before publication; saved candidates are retained"
                    )
            for segment, target, before_polish in zip(
                segments, result.targets, result.before_polish
            ):
                segment.target = target
                segment.target_before_polish = before_polish
            store.save_chapter(chapter)
            return chapter

    @staticmethod
    def chapter_progress_label(title: str, index: int) -> str:
        """Prefer the book's chapter title for progress so internal indices cannot contradict
        visible numbering.
        """
        title = (title or "").strip()
        return title or f"Chapter {index + 1}"

    def extract_batch_glossary(
        self,
        glossary: Storage | GlossaryStore,
        store: Storage,
        chapter: int,
        start_index: int,
        batch,
        translation_history: dict[tuple[int, int], TranslatedSegmentEvidence],
        source_corpus: str,
        *,
        terminology: TerminologyService | None = None,
    ) -> dict[str, int]:
        """Extract terms immediately after translating or resuming a batch for use by later
        chapter batches.
        """
        src_text = "\n".join(s.source for s in batch)
        tgt_text = "\n".join(s.target or "" for s in batch)
        summary = self._runtime.extractor.extract_and_store(
            glossary,
            src_text,
            tgt_text,
            chapter,
            history=translation_history.values(),
            before=(chapter, start_index),
            source_corpus=source_corpus,
            **({"admit": terminology.admit} if terminology is not None else {}),
        )
        store.log_event(
            "batch_glossary_extracted",
            chapter=chapter,
            start_index=start_index,
            count=len(batch),
            summary=summary,
        )
        return summary

    def recent_context(
        self,
        store: Storage,
        chapter: Chapter,
        start: int,
        policy: BabeldocContext | None,
    ) -> str:
        """Read the translated prefix backward, loading preceding chapters only as needed.

        Formal chapters are the sole source of context on fresh and resumed runs. Chapter
        boundaries do not stop history; BabelDOC still filters layout noise and stops at headings.
        """

        def targets() -> Iterator[str | None]:
            if policy is not None:
                yield from reversed(policy.recent_targets(chapter.text_segments, start))
                if start and policy.sections[start - 1] != 0:
                    return
            else:
                yield from (segment.target for segment in reversed(chapter.text_segments[:start]))
            for previous in self._neighboring_chapters(store, chapter.index, following=False):
                segments = previous.text_segments
                previous_policy = BabeldocContext.for_chapter(previous)
                if previous_policy is None:
                    yield from (segment.target for segment in reversed(segments))
                else:
                    yield from reversed(previous_policy.recent_targets(segments, len(segments)))
                    if previous_policy.sections and previous_policy.sections[-1] != 0:
                        return

        return render_recent_translations(
            targets(), self._runtime.config.pipeline.rolling_context_tokens
        )

    def following_source(
        self,
        store: Storage,
        chapter: Chapter,
        end: int,
        policy: BabeldocContext | None,
    ) -> str:
        """Find one following source reference in book order without translating it early."""
        segments = chapter.text_segments
        if policy is None:
            if end < len(segments):
                return segments[end].source
        else:
            if end <= 0 or not policy.body[end - 1]:
                return ""
            reference = policy.next_source(segments, end)
            if reference:
                return reference
            if policy.sections[-1] != policy.sections[end - 1]:
                return ""
        for following in self._neighboring_chapters(store, chapter.index, following=True):
            segments = following.text_segments
            following_policy = BabeldocContext.for_chapter(following)
            if following_policy is None:
                if segments:
                    return segments[0].source
            else:
                reference = following_policy.next_source(segments, 0)
                if reference:
                    return reference
                if any(following_policy.sections):
                    return ""
        return ""

    @staticmethod
    def _neighboring_chapters(
        store: Storage, chapter: int, *, following: bool
    ) -> Iterator[Chapter]:
        """Load neighboring chapters lazily in stable source order."""
        indices = sorted(
            row["index"]
            for row in store.load_manifest().get("chapters", [])
            if isinstance(row.get("index"), int)
            and (row["index"] > chapter if following else row["index"] < chapter)
        )
        for index in indices if following else reversed(indices):
            yield store.load_chapter(index)


class _BatchLookahead:
    """Run translate/polish requests ahead of glossary extraction on daemon threads.

    Results are pure; the chapter service decides whether to use them and alone writes
    formal state. A chapter's first batch can be carried over from the previous chapter.
    """

    def __init__(self, execute: Callable[[BatchPlan], BatchResult]):
        self._execute = execute
        self._pending: list[Future[BatchResult]] = []
        self._carried: tuple[int, BatchPlan, Future[BatchResult]] | None = None

    def submit(self, plan: BatchPlan) -> Future[BatchResult]:
        future: Future[BatchResult] = Future()
        context = contextvars.copy_context()

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = context.run(self._execute, plan)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=run, name="wenyi-translation-lookahead", daemon=True).start()
        self._pending = [pending for pending in self._pending if not pending.done()]
        self._pending.append(future)
        return future

    def carry(self, chapter: int, plan: BatchPlan) -> None:
        """Start the first batch of the next chapter and hold it for that chapter."""
        self._carried = (chapter, plan, self.submit(plan))

    def take_carry(self, chapter: int) -> tuple[BatchPlan, Future[BatchResult]] | None:
        carried, self._carried = self._carried, None
        if carried is None or carried[0] != chapter:
            return None
        return carried[1], carried[2]

    def drain(self) -> None:
        """Wait for submitted requests so none outlives the chapter that started it."""
        futures.wait(self._pending)
        self._pending.clear()
