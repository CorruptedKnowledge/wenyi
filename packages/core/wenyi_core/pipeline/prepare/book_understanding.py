"""Resumable chapter digests and whole-book synthesis, independent of runtime assembly."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ...agents.analyzer import Analyzer
from ...agents.synopsis import Synopsizer
from ...config import Config
from ...events import ProgressFn
from ...i18n.policy.models import content_hash
from ...llm.configuration import LLMConfig
from ...llm.operations import OPERATIONS
from ...llm.routing import inference_snapshot
from ...models.document import Chapter
from ...storage.protocol import Storage
from ..run.runtime_usage import UsageCoordinator

_LOGGER = logging.getLogger(__name__)
_DIGEST_VERSION = 2


def _hash_inputs(inputs: dict) -> str:
    return hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _legacy_llm_documents(llm: LLMConfig, operation: str) -> list[dict[str, Any]]:
    """Return whole-configuration documents that releases up to v1.5.1 may have hashed.

    Besides the current configuration, offer it without the routes, model profiles and
    providers that ``operation`` cannot use, so adding unrelated routes in the same step as
    upgrading still adopts paid caches. A reduced candidate is offered only when it resolves
    ``operation`` (primary and fallbacks) exactly as the current configuration does; a stored
    hash equal to the candidate's then proves the cache came from the same route.
    """
    current = llm.model_dump(mode="json")
    chain = set()
    name: str | None = operation
    while name is not None:
        chain.add(name)
        name = OPERATIONS[name].inherits
    routes = {key: route for key, route in current["routes"].items() if key in chain}
    without_routes = {**current, "routes": routes}
    profiles = set(current["tiers"].values())
    for route in routes.values():
        profiles.update(route["fallbacks"])
        if route["model"] is not None:
            profiles.add(route["model"])
    models = {key: model for key, model in current["models"].items() if key in profiles}
    providers = {model["provider"] for model in models.values()}
    minimal = {
        **without_routes,
        "models": models,
        "providers": {
            key: value for key, value in current["providers"].items() if key in providers
        },
    }
    expected = inference_snapshot(llm, (operation,))
    documents = [current]
    for document in (without_routes, minimal):
        if document in documents:
            continue
        try:
            reduced = LLMConfig.model_validate(document)
            if inference_snapshot(reduced, (operation,)) != expected:
                continue
        except (KeyError, ValueError):
            continue
        documents.append(document)
    return documents


def _saved_digest(saved: object) -> str:
    if isinstance(saved, dict) and isinstance(saved.get("digest"), str):
        return saved["digest"]
    return ""


def _digest_complete(chapter: Chapter) -> bool:
    """Only verified full-source digests satisfy preparation."""
    digest = chapter.meta.get("source_digest")
    return bool(
        isinstance(digest, str)
        and digest.strip()
        and chapter.meta.get("source_digest_complete") is True
        and chapter.meta.get("source_digest_version") == _DIGEST_VERSION
    )


class BookUnderstandingService:
    """Own prescan concurrency and cache identity, not initialization or model construction."""

    def __init__(
        self, config: Config, synopsizer: Synopsizer, analyzer: Analyzer, usage: UsageCoordinator
    ):
        self._config = config
        self._synopsizer = synopsizer
        self._analyzer = analyzer
        self._usage = usage

    def ensure_understanding(
        self,
        store: Storage,
        progress: ProgressFn | None = None,
    ) -> str:
        """Prescan source chapters into chapter.meta digests and an analysis synopsis.
        Require digests for nonempty chapters and reuse verified synopsis caches on resume.
        Return empty when book understanding is disabled or optional synopsis synthesis fails.
        """
        if not self._config.pipeline.book_understanding:
            store.log_event("book_understanding_skipped", reason="disabled")
            return ""
        manifest = store.load_manifest()
        chapters = manifest.get("chapters", [])

        loaded = [store.load_chapter(row.get("index", i)) for i, row in enumerate(chapters)]
        digests = self.ensure_chapter_digests(store, loaded, progress)
        if not digests:
            return ""

        analysis = store.load_analysis() or {}
        terms = sorted(store.all_terms(), key=lambda term: term.source)
        style = self._analyzer.style_brief(analysis, terms=terms)
        inputs = {
            "language_policy": self._config.language_policy("analysis").task_fingerprint(
                "book_synopsis"
            ),
            "chapters": [
                [chapter.index, chapter.meta["source_digest"]]
                for chapter in loaded
                if chapter.text_segments
            ],
            "style": style,
            "source_lang": self._config.source_lang,
            "target_lang": self._config.target_lang,
            "glossary": [
                [term.source, term.target, term.type, term.gender, term.note] for term in terms
            ],
        }
        # Bind the synopsis to the route that writes it; unrelated model edits keep it valid.
        metadata = {
            "version": 2,
            "inputs_sha256": _hash_inputs(
                {**inputs, "inference": inference_snapshot(self._config.llm, ("synopsis.book",))}
            ),
        }
        synopsis = analysis.get("book_synopsis", "")
        if isinstance(synopsis, str) and synopsis.strip():
            saved = analysis.get("book_synopsis_meta")
            if saved == metadata:
                return synopsis
            # Releases up to v1.5.1 hashed the whole LLM configuration; adopt and re-key a
            # synopsis whose legacy identity still matches the synopsis.book route.
            if any(
                saved == {"version": 2, "inputs_sha256": _hash_inputs({**inputs, "routing": llm})}
                for llm in _legacy_llm_documents(self._config.llm, "synopsis.book")
            ):
                analysis["book_synopsis_meta"] = metadata
                store.save_analysis(analysis)
                return synopsis
        if progress:
            progress(0, 0, "Generating whole-book synopsis…")
        synopsis = self._synopsizer.book_synopsis(digests, style, glossary=terms)
        if synopsis:
            analysis["book_synopsis"] = synopsis
            analysis["book_synopsis_meta"] = metadata
            store.save_analysis(analysis)
            store.log_event("book_synopsis_saved", synopsis=synopsis)
        else:
            store.log_event("book_synopsis_failed", reason="generation_failed")
            _LOGGER.warning(
                "Whole-book synopsis generation failed; translation continues with chapter digests. "
                "The synopsis will be retried on the next prepare/translate run."
            )
        return synopsis

    def ensure_chapter_digests(
        self,
        store: Storage,
        chapters: list[Chapter],
        progress: ProgressFn | None,
    ) -> list[str]:
        """Checkpoint all incurred requests even when prescan cannot finish."""
        try:
            return self._build_chapter_digests(store, chapters, progress)
        finally:
            self._usage.flush(store, scope="prescan")

    def _build_chapter_digests(
        self,
        store: Storage,
        chapters: list[Chapter],
        progress: ProgressFn | None,
    ) -> list[str]:
        """Prescan staged or initialized chapters and return digests in source order."""
        # Persist results on the calling thread, preserving chapter-level resume and source order.
        loaded = {chapter.index: chapter for chapter in chapters}
        sources = {
            ci: "\n".join(s.source for s in ch.text_segments)
            for ci, ch in loaded.items()
            if ch.text_segments
        }
        policy = self._config.language_policy("analysis").task_fingerprint("chapter_digest")
        # Bind digests to the prompts and the synopsis.chapter route (including fallbacks), so
        # edits to other operations or unused models keep the paid prescan.
        fingerprint = content_hash(
            {
                "policy": policy,
                "inference": inference_snapshot(self._config.llm, ("synopsis.chapter",)),
            }
        )
        # Releases up to v1.5.1 hashed the whole LLM configuration. Entries under an identity
        # that still resolves synopsis.chapter identically are adopted instead of paying for a
        # one-time rebuild after upgrade.
        legacy = [
            (content_hash([policy, llm]), llm)
            for llm in _legacy_llm_documents(self._config.llm, "synopsis.chapter")
        ]
        source_hashes = {
            ci: hashlib.sha256(source.encode("utf-8")).hexdigest() for ci, source in sources.items()
        }

        def current(ci: int, identity: str) -> bool:
            chapter = loaded[ci]
            return (
                _digest_complete(chapter)
                and chapter.meta.get("source_digest_policy") == identity
                and chapter.meta.get("source_digest_sha256") == source_hashes[ci]
            )

        cache_keys = {
            ci: "preparation/digests/"
            + content_hash([_DIGEST_VERSION, source_hashes[ci], fingerprint])
            + ".json"
            for ci in sources
        }
        for ci in sources:
            if current(ci, fingerprint):
                continue
            digest = _saved_digest(store.read_artifact(cache_keys[ci]))
            if not digest:
                if any(current(ci, legacy_fingerprint) for legacy_fingerprint, _ in legacy):
                    digest = loaded[ci].meta["source_digest"]
                for legacy_fingerprint, llm in legacy:
                    if digest:
                        break
                    legacy_key = content_hash(
                        [_DIGEST_VERSION, source_hashes[ci], legacy_fingerprint, llm]
                    )
                    digest = _saved_digest(
                        store.read_artifact(f"preparation/digests/{legacy_key}.json")
                    )
                if not digest:
                    continue
                # Re-key before marking the chapter; an interrupted migration simply repeats.
                store.write_artifact(cache_keys[ci], {"digest": digest})
            loaded[ci].meta.update(
                source_digest=digest,
                source_digest_complete=True,
                source_digest_policy=fingerprint,
                source_digest_version=_DIGEST_VERSION,
                source_digest_sha256=source_hashes[ci],
            )
            store.save_chapter(loaded[ci])
        todo = [(ci, source) for ci, source in sources.items() if not current(ci, fingerprint)]
        failed: list[int] = []
        error: Exception | None = None
        if todo:
            store.log_event(
                "book_understanding_chapter_digest_started",
                chapters=[ci for ci, _ in todo],
                workers=max(1, self._config.pipeline.prescan_concurrency),
            )
            workers = max(1, self._config.pipeline.prescan_concurrency)
            if progress:
                progress(0, len(todo), "Prescanning chapter digests")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._synopsizer.digest_chapter, src): ci for ci, src in todo}
                for n_done, fut in enumerate(as_completed(futs), 1):
                    ci = futs[fut]
                    try:
                        digest = fut.result().strip()
                    except Exception as exc:
                        # Drain submitted work and save successes before propagating failures.
                        error = error or exc
                        digest = ""
                    if digest:
                        loaded[ci].meta["source_digest"] = digest
                        loaded[ci].meta["source_digest_complete"] = True
                        loaded[ci].meta["source_digest_policy"] = fingerprint
                        loaded[ci].meta["source_digest_version"] = _DIGEST_VERSION
                        loaded[ci].meta["source_digest_sha256"] = source_hashes[ci]
                        store.write_artifact(cache_keys[ci], {"digest": digest})
                        store.save_chapter(loaded[ci])
                        store.log_event(
                            "book_understanding_chapter_digest_saved", chapter=ci, digest=digest
                        )
                    else:
                        failed.append(ci)
                        store.log_event("book_understanding_chapter_digest_failed", chapter=ci)
                    if progress:
                        progress(n_done, len(todo), "Prescanning chapter digests")

        if error is not None:
            raise error
        if failed:
            indices = ", ".join(str(ci) for ci in sorted(failed))
            raise ValueError(
                f"Chapter digests could not be generated for chapters: {indices}. "
                "Retry prepare/translate; completed digests have been saved."
            )

        # Assemble in manifest chapter order, independent of worker completion order.
        return [loaded[ci].meta["source_digest"] for ci in sources]
