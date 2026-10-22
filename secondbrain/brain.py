"""The Brain: one object wiring settings, store, embedder, search, indexing and Q&A together."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import Settings
from .embeddings import resolve_embedder
from .graph import discovered_connections, graph_payload
from .indexer import Indexer, IndexReport
from .llm import active_provider, ProviderStore, detect_llm
from .qa import answer, answer_text
from .search import Searcher
from .store import Store

log = logging.getLogger("secondbrain")


class Brain:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.store = Store(self.settings.db_path)
        stored = self.store.get_meta("embedder")
        choice = resolve_embedder(self.settings, stored.get("id") if stored else None)
        self.embedder, self.embedder_note = choice.embedder, choice.note
        if choice.degraded:
            log.warning("embedding model unavailable: %s", choice.note)
        self.searcher = Searcher(self.store, self.embedder)
        self.progress: dict = {"stage": "idle"}
        self.last_report: IndexReport | None = None
        self.indexer = Indexer(self.store, self.embedder, self.settings, self._on_progress)
        self.indexer.vectors_enabled = not choice.degraded
        self._job: threading.Thread | None = None
        self.on_quit = None  # set by the desktop app

    def _on_progress(self, p: dict) -> None:
        self.progress = {**p, "at": time.time()}

    def ensure_ready(self) -> bool:
        """Re-embed if the embedder changed since the last run. Returns True if work was done."""
        return self.indexer.ensure_embedder()

    # --------------------------------------------------------- indexing

    def add(self, paths: list[str | Path]) -> IndexReport:
        self.last_report = self.indexer.index_paths(paths)
        return self.last_report

    def reindex(self) -> IndexReport:
        self.last_report = self.indexer.reindex_all()
        return self.last_report

    def pause(self) -> None:
        self.indexer.pause()

    def resume(self) -> None:
        self.indexer.resume()

    def catch_up(self) -> None:
        """On startup: finish an interrupted index and pick up changes made while the app was closed."""
        if self.store.roots():
            self.run_in_background(self.reindex)

    def remove(self, path: str) -> int:
        return self.indexer.remove_path(path)

    def run_in_background(self, fn, *args) -> bool:
        """Run an indexing job on a thread. Returns False if one is already running."""
        if self._job and self._job.is_alive():
            return False

        def work():
            try:
                result = fn(*args)
                if isinstance(result, IndexReport):
                    self.last_report = result
            except Exception as e:
                log.exception("indexing failed")
                self.progress = {"stage": "error", "error": str(e)}

        self._job = threading.Thread(target=work, daemon=True)
        self._job.start()
        return True

    @property
    def busy(self) -> bool:
        return bool(self._job and self._job.is_alive())

    # --------------------------------------------------------- querying

    def search(self, q: str, **kw):
        return self.searcher.search(q, **kw)

    def ask_stream(self, question: str, **kw):
        return answer(question, self.searcher, self.settings, **kw)

    def ask(self, question: str, **kw):
        return answer_text(question, self.searcher, self.settings, **kw)

    def graph(self) -> dict:
        return graph_payload(self.store)

    def connections(self, limit: int = 50) -> list[dict]:
        return discovered_connections(self.store, limit)

    def status(self) -> dict:
        llm = detect_llm(self.settings)
        return {
            **self.store.stats(),
            "embedder": {"id": self.embedder.id, "description": self.embedder.describe(),
                         "degraded": not self.indexer.vectors_enabled, "note": self.embedder_note},
            "llm": llm.name if llm else None,
            "llm_cloud": bool(llm and llm.cloud),
            "llm_choice": active_provider(self.settings, ProviderStore(self.settings.home)),
            "progress": self.progress,
            "busy": self.busy,
            "paused": self.indexer.paused,
            "last_report": self.last_report.to_dict() if self.last_report else None,
            "home": str(self.settings.home),
            "desktop": self.on_quit is not None,
        }
