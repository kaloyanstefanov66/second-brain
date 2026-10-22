"""Keep the index in sync with your folders: re-index files shortly after they change."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .loaders import IGNORED_DIRS, is_supported

log = logging.getLogger("secondbrain")
FILE_ROOT_INTERVAL = 15 * 60  # seconds


class FolderWatcher:
    def __init__(self, brain, debounce: float = 2.0):
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        self.brain, self.debounce = brain, debounce
        self.pending: dict[Path, float] = {}
        self.lock = threading.Lock()
        self.observer = Observer()
        self.watched: set[str] = set()
        watcher = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                for attr in ("src_path", "dest_path"):
                    p = getattr(event, attr, None)
                    if p:
                        watcher._queue(Path(p), event.is_directory)

        self.handler = Handler()

    def _queue(self, p: Path, is_dir: bool) -> None:
        if any(part in IGNORED_DIRS or part.startswith(".") for part in p.parts[1:]):
            return
        if not is_dir and not is_supported(p) and p.exists():
            return
        with self.lock:
            self.pending[p] = time.time()

    def sync_roots(self) -> None:
        for root in self.brain.store.roots():
            if root not in self.watched and Path(root).is_dir():
                self.observer.schedule(self.handler, root, recursive=True)
                self.watched.add(root)

    def start(self) -> None:
        self.sync_roots()
        self.observer.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        last_file_sync = time.time()
        while True:
            time.sleep(0.5)
            self.sync_roots()
            now = time.time()
            if now - last_file_sync > FILE_ROOT_INTERVAL and not self.brain.busy:
                # Browser bookmark stores change on every visit; re-read them periodically instead.
                last_file_sync = now
                files = [Path(r) for r in self.brain.store.roots() if Path(r).is_file()]
                if files:
                    try:
                        self.brain.indexer.sync_files(files)
                    except Exception:
                        log.exception("bookmark sync failed")
            with self.lock:
                ready = [p for p, t in self.pending.items() if now - t >= self.debounce]
                for p in ready:
                    del self.pending[p]
            ready = [p for p in ready if p.exists() and p.is_file() or not p.exists()]
            if ready and not self.brain.busy:
                try:
                    report = self.brain.indexer.sync_files(ready)
                    if report.added or report.updated or report.removed:
                        log.info("watcher: %s", report.summary())
                        self.brain.last_report = report
                except Exception:
                    log.exception("watcher sync failed")
            elif ready:
                with self.lock:  # indexing job running; retry later
                    for p in ready:
                        self.pending.setdefault(p, now)

    def stop(self) -> None:
        self.observer.stop()
