"""Hybrid retrieval: dense vectors + SQLite FTS5 (BM25), fused with reciprocal rank fusion."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass

import numpy as np

from .embeddings import STOPWORDS, Embedder, normalize
from .store import Store

RRF_K = 60


class VectorIndex:
    """All chunk embeddings in one in-memory matrix (fine up to a few hundred thousand chunks)."""

    def __init__(self, store: Store, embedder: Embedder):
        ids, docs, mat = store.load_embeddings()
        if mat.size and mat.shape[1] != embedder.dim:
            ids, docs, mat = ids[:0], docs[:0], np.zeros((0, embedder.dim), np.float32)
        self.weights = None
        if embedder.idf_weighting and len(ids):
            # Lexical embeddings: down-weight hash buckets that appear in many chunks (IDF).
            df = np.count_nonzero(mat, axis=0)
            self.weights = (np.log((len(ids) + 1) / (df + 1)) + 1).astype(np.float32)
            mat = normalize(mat * self.weights)
        self.chunk_ids, self.doc_ids, self.matrix = ids, docs, mat
        self.row_of = {int(c): i for i, c in enumerate(ids)}

    def __len__(self):
        return len(self.chunk_ids)

    def project(self, vec: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            vec = vec * self.weights
        return normalize(vec)

    def query(self, vec: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        if not len(self):
            return []
        sims = self.matrix @ self.project(vec)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.chunk_ids[i]), float(sims[i])) for i in top]

    def doc_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """One vector per document: the normalized mean of its chunk vectors."""
        if not len(self):
            return np.zeros(0, np.int64), np.zeros((0, self.matrix.shape[1] if self.matrix.ndim == 2 else 0), np.float32)
        uniq, inverse = np.unique(self.doc_ids, return_inverse=True)
        sums = np.zeros((len(uniq), self.matrix.shape[1]), np.float32)
        np.add.at(sums, inverse, self.matrix)
        return uniq, normalize(sums)


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    title: str
    path: str
    kind: str
    url: str
    heading: str
    page: int | None
    text: str
    snippet: str
    score: float
    similarity: float
    keyword: bool
    tags: list

    def to_dict(self):
        return asdict(self)


def query_terms(q: str) -> list[str]:
    return [t for t in re.findall(r"[^\W_]+", q.lower()) if t not in STOPWORDS and len(t) > 1]


def fts_query(q: str) -> str:
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]+"', " ", q)
    parts = [f'"{p.strip()}"' for p in phrases if p.strip()]
    parts += [f'"{t}"' for t in query_terms(rest)]
    return " OR ".join(parts)


def make_snippet(text: str, terms: list[str], width: int = 280) -> str:
    low = text.lower()
    positions = [p for p in (low.find(t) for t in terms) if p >= 0]
    if not positions or len(text) <= width:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, min(positions) - width // 3)
    end = min(len(text), start + width)
    start = text.rfind(" ", 0, start) + 1 if start > 0 else 0
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


class Searcher:
    def __init__(self, store: Store, embedder: Embedder):
        self.store, self.embedder = store, embedder
        self._index: VectorIndex | None = None
        self._version = -1
        self._lock = threading.Lock()

    @property
    def index(self) -> VectorIndex:
        with self._lock:
            v = self.store.version
            if self._index is None or v != self._version:
                self._index = VectorIndex(self.store, self.embedder)
                self._version = v
            return self._index

    def search(self, q: str, k: int = 10, kinds: list[str] | None = None,
               max_per_doc: int = 2, vector_only: bool = False) -> list[Hit]:
        q = q.strip()
        if not q:
            return []
        index = self.index
        qvec = self.embedder.embed_query(q)
        qproj = index.project(qvec) if len(index) else None
        vec_hits = index.query(qvec, 80)
        sims = dict(vec_hits)
        kw_hits = [] if vector_only else self.store.fts(fts_query(q), 80)
        kw_set = {cid for cid, _ in kw_hits}

        fused: dict[int, float] = {}
        for rank, (cid, _) in enumerate(vec_hits):
            fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)
        for rank, (cid, _) in enumerate(kw_hits):
            fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)

        ranked = sorted(fused, key=fused.get, reverse=True)[: max(k * 6, 60)]
        rows = self.store.chunks_by_ids(ranked)
        terms = query_terms(q)
        hits: list[Hit] = []
        per_doc: dict[int, int] = {}
        for cid in ranked:
            r = rows.get(cid)
            if not r or (kinds and r["kind"] not in kinds):
                continue
            if per_doc.get(r["doc_id"], 0) >= max_per_doc:
                continue
            per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
            sim = sims.get(cid)
            if sim is None and cid in index.row_of:
                sim = float(index.matrix[index.row_of[cid]] @ qproj)
            hits.append(Hit(cid, r["doc_id"], r["title"], r["path"], r["kind"], r["url"], r["heading"],
                            r["page"], r["text"], make_snippet(r["text"], terms), round(fused[cid] * RRF_K, 4),
                            round(sim or 0.0, 4), cid in kw_set, r["tags"]))
            if len(hits) >= k:
                break
        return hits
