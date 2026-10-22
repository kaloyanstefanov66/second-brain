"""Incremental indexing: scan folders, load changed files, chunk, embed, store, rebuild the graph."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .chunking import Chunk, chunk_sections
from .embeddings import Embedder
from .graph import build_graph
from .loaders import LoadedDoc, Section, fetch_page_text, is_supported, iter_files, load_file
from .search import VectorIndex
from .store import Store


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    files: int = 0
    failed: list[dict] = field(default_factory=list)
    graph: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        s = (f"{self.files} files: {self.added} added, {self.updated} updated, {self.unchanged} unchanged, "
             f"{self.removed} removed")
        if self.graph:
            s += (f" · graph: {self.graph['links']} links, {self.graph['semantic']} semantic edges, "
                  f"{self.graph['discovered']} discovered connections")
        if self.failed:
            s += f" · {len(self.failed)} failed"
        return s + f" ({self.seconds:.1f}s)"


def embed_text(doc: LoadedDoc, chunk: Chunk) -> str:
    """What gets embedded: chunk text with its document title and heading for context."""
    head = " — ".join(p for p in (doc.title, chunk.heading) if p)
    return f"{head}\n{chunk.text}"


class Indexer:
    def __init__(self, store: Store, embedder: Embedder, settings,
                 on_progress: Callable[[dict], None] | None = None):
        self.store, self.embedder, self.settings = store, embedder, settings
        self.on_progress = on_progress or (lambda p: None)
        self.lock = threading.RLock()

    # ------------------------------------------------------------ public API

    def ensure_embedder(self) -> bool:
        """Re-embed everything if the index was built with a different embedder. Returns True if it did."""
        stored = self.store.get_meta("embedder")
        if stored and stored.get("id") == self.embedder.id:
            return False
        has_chunks = self.store.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
        if has_chunks:
            self.reembed_all()
        else:
            self._save_embedder()
        return bool(has_chunks)

    def reembed_all(self) -> None:
        with self.lock:
            rows = list(self.store.iter_chunk_texts())
            for i in range(0, len(rows), 256):
                batch = rows[i:i + 256]
                self.on_progress({"stage": "re-embedding", "current": i, "total": len(rows)})
                texts = [f"{' — '.join(p for p in (r['title'], r['heading']) if p)}\n{r['text']}" for r in batch]
                vecs = self.embedder.embed(texts)
                with self.store.tx() as c:
                    self.store.set_embeddings(c, zip((r["id"] for r in batch), vecs))
            self._save_embedder()
            self.finalize(IndexReport())

    def index_paths(self, paths: list[str | Path], remember: bool = True) -> IndexReport:
        with self.lock:
            t0 = time.time()
            report = IndexReport()
            for raw in paths:
                root = Path(raw).expanduser().resolve()
                if not root.exists():
                    report.failed.append({"path": str(root), "error": "path does not exist"})
                    continue
                if remember:
                    self.store.add_root(str(root))
                files = list(iter_files(root))
                for i, f in enumerate(files):
                    self.on_progress({"stage": "indexing", "current": i, "total": len(files), "file": str(f)})
                    self.index_file(f, report)
                if root.is_dir():
                    seen = {str(f) for f in files}
                    for source in self.store.sources_under(str(root)) - seen:
                        report.removed += self.remove_source(source)
            self.finalize(report)
            report.seconds = time.time() - t0
            return report

    def reindex_all(self) -> IndexReport:
        return self.index_paths(self.store.roots(), remember=False)

    def sync_files(self, paths: list[Path]) -> IndexReport:
        """Re-index specific files (used by the folder watcher); deleted files are removed."""
        with self.lock:
            t0 = time.time()
            report = IndexReport()
            for p in paths:
                if p.exists() and p.is_file() and is_supported(p):
                    self.index_file(p, report)
                elif not p.exists():
                    report.removed += self.remove_source(str(p))
                    for source in self.store.sources_under(str(p)):  # a deleted folder
                        report.removed += self.remove_source(source)
            self.finalize(report)
            report.seconds = time.time() - t0
            return report

    def remove_path(self, path: str) -> int:
        with self.lock:
            root = str(Path(path).expanduser().resolve())
            removed = sum(self.remove_source(s) for s in self.store.sources_under(root))
            self.store.remove_root(root)
            self.finalize(IndexReport())
            return removed

    def remove_source(self, source: str) -> int:
        docs = self.store.docs_from_source(source)
        with self.store.tx() as c:
            for d in docs:
                self.store.delete_document(c, d["id"])
        return len(docs)

    # ------------------------------------------------------------ per file

    def index_file(self, path: Path, report: IndexReport) -> None:
        report.files += 1
        try:
            st = path.stat()
            if st.st_size > self.settings.max_file_mb * 1024 * 1024:
                report.failed.append({"path": str(path), "error": f"larger than {self.settings.max_file_mb}MB"})
                return
            existing = {d["path"]: d for d in self.store.docs_from_source(str(path))}
            if existing and all(d["mtime"] == st.st_mtime and d["size"] == st.st_size for d in existing.values()):
                report.unchanged += len(existing)
                return

            docs = load_file(path)
            pending: list[tuple[LoadedDoc, str, bool]] = []
            with self.store.tx() as c:
                for doc in docs:
                    h = doc.hash
                    ex = existing.get(doc.path)
                    if ex and ex["hash"] == h:
                        self.store.touch_document(c, ex["id"], st.st_mtime, st.st_size)
                        report.unchanged += 1
                    else:
                        pending.append((doc, h, ex is not None))
                gone = set(existing) - {d.path for d in docs}
                for p in gone:
                    self.store.delete_document(c, existing[p]["id"])
                report.removed += len(gone)

            if self.settings.fetch_bookmarks:
                self._fetch_bookmarks([d for d, _, _ in pending if d.kind == "bookmark"])
            self._write(pending, st.st_mtime, st.st_size, report)
        except Exception as e:  # one bad file never stops the run
            report.failed.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})

    def _write(self, pending, mtime: float, size: int, report: IndexReport) -> None:
        if not pending:
            return
        plans = []
        texts: list[str] = []
        for doc, h, existed in pending:
            chunks = chunk_sections(doc.sections, self.settings.chunk_chars, self.settings.chunk_overlap)
            if not chunks:  # empty file: keep the title searchable
                chunks = [Chunk(doc.title)]
            plans.append((doc, h, existed, chunks, len(texts)))
            texts += [embed_text(doc, ch) for ch in chunks]
        vecs = self.embedder.embed(texts)
        with self.store.tx() as c:
            for doc, h, existed, chunks, offset in plans:
                self.store.upsert_document(c, doc, chunks, vecs[offset:offset + len(chunks)], mtime, size, h)
                if existed:
                    report.updated += 1
                else:
                    report.added += 1

    def _fetch_bookmarks(self, docs: list[LoadedDoc]) -> None:
        def fetch(doc: LoadedDoc):
            try:
                text = fetch_page_text(doc.url)
                if text:
                    doc.sections.append(Section(text, "Page content"))
            except Exception:
                pass

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(fetch, docs))

    # ------------------------------------------------------------ graph

    def finalize(self, report: IndexReport) -> None:
        self.on_progress({"stage": "building graph"})
        self.store.bump_version()  # so the vector index picks up new chunks
        index = VectorIndex(self.store, self.embedder)
        report.graph = build_graph(self.store, index, self.embedder, self.settings)
        self._save_embedder()
        self.store.set_meta("last_indexed", time.time())
        self.store.bump_version()
        self.on_progress({"stage": "idle"})

    def _save_embedder(self) -> None:
        self.store.set_meta("embedder", {"id": self.embedder.id, "dim": self.embedder.dim,
                                         "description": self.embedder.describe()})
