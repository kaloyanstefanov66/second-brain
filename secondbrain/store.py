"""SQLite persistence: documents, chunks (+ embeddings), full-text index and graph edges.

Everything lives in a single file (~/.secondbrain/brain.db by default). Each thread gets its
own connection; WAL mode lets the web server read while the indexer writes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Bump SCHEMA_VERSION and add an entry to MIGRATIONS for every schema change. SCHEMA below is the
# version-1 baseline and must never be edited: libraries made by any earlier release are upgraded in
# place, so updating the app never means rebuilding the index.
SCHEMA_VERSION = 1
MIGRATIONS: dict[int, str] = {
    # 2: "ALTER TABLE documents ADD COLUMN example TEXT;",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY, added REAL);
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    content TEXT DEFAULT '',
    hash TEXT,
    mtime REAL,
    size INTEGER,
    words INTEGER DEFAULT 0,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    heading TEXT DEFAULT '',
    page INTEGER,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, title, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS raw_links(
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS raw_links_doc ON raw_links(doc_id);
CREATE TABLE IF NOT EXISTS edges(
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    kind TEXT NOT NULL,          -- link | semantic
    weight REAL NOT NULL,
    data TEXT DEFAULT '{}',
    PRIMARY KEY(src, dst, kind)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
"""


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    if "tags" in d and isinstance(d["tags"], str):
        d["tags"] = json.loads(d["tags"] or "[]")
    if "data" in d and isinstance(d["data"], str):
        d["data"] = json.loads(d["data"] or "{}")
    return d


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._local = threading.local()
        self.write_lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        c = self.conn
        version = c.execute("PRAGMA user_version").fetchone()[0] or 1  # 0.1.0/0.1.1 libraries are version 1
        if version > SCHEMA_VERSION:
            raise RuntimeError("This library was created by a newer version of Second Brain. "
                               "Please update the app (your data is untouched).")
        for target in range(version + 1, SCHEMA_VERSION + 1):
            with self.write_lock:
                c.executescript(f"BEGIN; {MIGRATIONS[target]} PRAGMA user_version = {target}; COMMIT;")
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self):
        with self.write_lock:
            c = self.conn
            try:
                c.execute("BEGIN")
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------ meta / roots

    def get_meta(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key: str, value, conn=None) -> None:
        (conn or self.conn).execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        if conn is None:
            self.conn.commit()

    @property
    def version(self) -> int:
        """Bumped after every index change; caches key off it."""
        return int(self.get_meta("version", 0))

    def bump_version(self) -> None:
        with self.tx() as c:
            self.set_meta("version", self.version + 1, c)

    def add_root(self, path: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO roots VALUES (?, ?)", (path, time.time()))

    def remove_root(self, path: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM roots WHERE path=?", (path,))

    def roots(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT path FROM roots ORDER BY path")]

    # ------------------------------------------------------------ documents

    def doc_by_path(self, path: str) -> dict | None:
        return _row(self.conn.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone())

    def doc(self, doc_id: int) -> dict | None:
        return _row(self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())

    def docs_from_source(self, source: str) -> list[dict]:
        return [_row(r) for r in self.conn.execute(
            "SELECT id, path, hash, mtime, size FROM documents WHERE source=?", (source,))]

    def all_docs(self, with_content: bool = False) -> list[dict]:
        cols = "*" if with_content else "id, path, title, kind, source, url, tags, words, mtime, indexed_at"
        return [_row(r) for r in self.conn.execute(f"SELECT {cols} FROM documents ORDER BY title COLLATE NOCASE")]

    def sources_under(self, root: str) -> set[str]:
        prefix = root.rstrip("/\\") + os.sep
        return {r[0] for r in self.conn.execute(
            "SELECT DISTINCT source FROM documents WHERE source = ? OR substr(source, 1, ?) = ?",
            (root, len(prefix), prefix))}

    def upsert_document(self, c, doc, chunks, embeddings: np.ndarray, mtime: float, size: int,
                        doc_hash: str) -> int:
        """Replace a document and all its chunks. Caller holds a transaction."""
        existing = c.execute("SELECT id FROM documents WHERE path=?", (doc.path,)).fetchone()
        fields = (doc.title, doc.kind, doc.source, doc.url, json.dumps(doc.tags), doc.content,
                  doc_hash, mtime, size, len(doc.content.split()), time.time())
        if existing:
            doc_id = existing[0]
            self._delete_children(c, doc_id)
            c.execute("""UPDATE documents SET title=?, kind=?, source=?, url=?, tags=?, content=?, hash=?,
                         mtime=?, size=?, words=?, indexed_at=? WHERE id=?""", (*fields, doc_id))
        else:
            doc_id = c.execute("""INSERT INTO documents(title, kind, source, url, tags, content, hash, mtime,
                                  size, words, indexed_at, path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                               (*fields, doc.path)).lastrowid
        for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
            cid = c.execute("INSERT INTO chunks(doc_id, ord, text, heading, page, embedding) VALUES (?,?,?,?,?,?)",
                            (doc_id, i, ch.text, ch.heading, ch.page,
                             None if emb is None else np.asarray(emb, np.float32).tobytes())).lastrowid
            c.execute("INSERT INTO chunks_fts(rowid, text, heading, title) VALUES (?,?,?,?)",
                      (cid, ch.text, ch.heading, doc.title))
        c.executemany("INSERT INTO raw_links VALUES (?, ?)", [(doc_id, t) for t in doc.links])
        return doc_id

    def touch_document(self, c, doc_id: int, mtime: float, size: int) -> None:
        c.execute("UPDATE documents SET mtime=?, size=? WHERE id=?", (mtime, size, doc_id))

    def _delete_children(self, c, doc_id: int) -> None:
        c.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM raw_links WHERE doc_id=?", (doc_id,))

    def delete_document(self, c, doc_id: int) -> None:
        self._delete_children(c, doc_id)
        c.execute("DELETE FROM edges WHERE src=? OR dst=?", (doc_id, doc_id))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    # ------------------------------------------------------------ chunks / embeddings

    def chunk(self, chunk_id: int) -> dict | None:
        return _row(self.conn.execute(
            "SELECT id, doc_id, ord, text, heading, page FROM chunks WHERE id=?", (chunk_id,)).fetchone())

    def chunks_for(self, doc_id: int) -> list[dict]:
        return [_row(r) for r in self.conn.execute(
            "SELECT id, doc_id, ord, text, heading, page FROM chunks WHERE doc_id=? ORDER BY ord", (doc_id,))]

    def chunks_by_ids(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(f"""
            SELECT c.id, c.doc_id, c.ord, c.text, c.heading, c.page, d.title, d.path, d.kind, d.url, d.tags
            FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({q})""", ids)
        return {r["id"]: _row(r) for r in rows}

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(chunk_ids, doc_ids, matrix) for every chunk."""
        rows = self.conn.execute("SELECT id, doc_id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id").fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        ids = np.fromiter((r[0] for r in rows), np.int64, len(rows))
        docs = np.fromiter((r[1] for r in rows), np.int64, len(rows))
        mat = np.stack([np.frombuffer(r[2], np.float32) for r in rows])
        return ids, docs, mat

    def iter_chunk_texts(self, missing_only: bool = False):
        where = "WHERE c.embedding IS NULL" if missing_only else ""
        yield from self.conn.execute(
            f"SELECT c.id, c.text, c.heading, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id {where} ORDER BY c.id")

    def set_embeddings(self, c, pairs) -> None:
        c.executemany("UPDATE chunks SET embedding=? WHERE id=?",
                      [(np.asarray(e, np.float32).tobytes(), cid) for cid, e in pairs])

    def fts(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts, 1.0, 0.6, 2.0) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                (query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------ links / edges

    def all_raw_links(self) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self.conn.execute("SELECT doc_id, target FROM raw_links")]

    def replace_edges(self, kind: str, edges: list[tuple[int, int, float, dict]]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM edges WHERE kind=?", (kind,))
            c.executemany("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                          [(s, d, kind, w, json.dumps(data)) for s, d, w, data in edges])

    def edges(self, kinds: tuple[str, ...] = ("link", "semantic")) -> list[dict]:
        q = ",".join("?" * len(kinds))
        return [_row(r) for r in self.conn.execute(f"SELECT * FROM edges WHERE kind IN ({q})", kinds)]

    def edges_for(self, doc_id: int) -> list[dict]:
        return [_row(r) for r in self.conn.execute("""
            SELECT e.*, s.title AS src_title, s.kind AS src_kind, d.title AS dst_title, d.kind AS dst_kind
            FROM edges e JOIN documents s ON s.id=e.src JOIN documents d ON d.id=e.dst
            WHERE e.src=? OR e.dst=? ORDER BY e.weight DESC""", (doc_id, doc_id))]

    def stats(self) -> dict:
        c = self.conn
        kinds = {r[0]: r[1] for r in c.execute("SELECT kind, count(*) FROM documents GROUP BY kind")}
        return {
            "documents": sum(kinds.values()),
            "by_kind": kinds,
            "chunks": c.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "links": c.execute("SELECT count(*) FROM edges WHERE kind='link'").fetchone()[0],
            "semantic": c.execute("SELECT count(*) FROM edges WHERE kind='semantic'").fetchone()[0],
            "discovered": c.execute(
                "SELECT count(*) FROM edges WHERE kind='semantic' AND json_extract(data, '$.discovered')").fetchone()[0],
            "words": c.execute("SELECT coalesce(sum(words),0) FROM documents").fetchone()[0],
            "roots": self.roots(),
            "embedder": self.get_meta("embedder"),
            "last_indexed": self.get_meta("last_indexed"),
        }
