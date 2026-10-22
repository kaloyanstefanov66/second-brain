"""Build the knowledge graph.

Edges come from two places:
  * link     - explicit [[wikilinks]] / relative Markdown links you wrote yourself
  * semantic - documents whose embeddings are close (top-k neighbours above a threshold)

A semantic edge is flagged as a *discovered connection* when nothing you wrote ties the two
documents together: no shared tags and no short path between them through your explicit links.
Each discovery is explained with the terms that bridge the two documents and the pair of
passages that are most similar.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .embeddings import STOPWORDS, TOKEN_RE, Embedder, stem
from .search import VectorIndex
from .store import Store

DISCOVERY_MIN_DISTANCE = 3  # explicit-link hops; closer pairs are "already known"
GUARANTEED = 2              # neighbours every document keeps if above background


# ------------------------------------------------------------------ explicit links

def resolve_links(store: Store) -> list[tuple[int, int, float, dict]]:
    docs = store.all_docs()
    by_path = {d["path"]: d["id"] for d in docs}
    by_title: dict[str, int] = {}
    by_stem: dict[str, int] = {}
    for d in docs:
        by_title.setdefault(d["title"].strip().lower(), d["id"])
        if d["kind"] != "bookmark":
            by_stem.setdefault(Path(d["path"]).stem.lower(), d["id"])
    paths = {d["id"]: d["path"] for d in docs}

    edges = {}
    for src, target in store.all_raw_links():
        t = unquote(target.strip())
        dst = None
        if "/" in t or re.search(r"\.\w{1,5}$", t):
            base = Path(paths.get(src, "")).parent
            cand = (base / t).resolve()
            for c in (cand, cand.with_suffix(cand.suffix + ".md"), cand.with_suffix(".md")):
                dst = by_path.get(str(c))
                if dst:
                    break
            if dst is None:
                dst = by_stem.get(Path(t).stem.lower())
        else:
            key = t.lower()
            dst = by_title.get(key) or by_stem.get(key)
        if dst and dst != src:
            edges[(src, dst)] = (src, dst, 1.0, {"target": target})
    return list(edges.values())


# ------------------------------------------------------------------ term statistics

def term_stats(store: Store) -> tuple[dict[int, Counter], dict[str, float], dict[str, str]]:
    """Per-document stem counts, IDF per stem, and the most common surface form of each stem."""
    tf: dict[int, Counter] = {}
    surface: dict[str, Counter] = defaultdict(Counter)
    df: Counter = Counter()
    for d in store.all_docs(with_content=True):
        c: Counter = Counter()
        for w in TOKEN_RE.findall(f"{d['title']} {d['content']}".lower()):
            if w in STOPWORDS or w.isdigit() or len(w) < 3:
                continue
            s = stem(w)
            c[s] += 1
            surface[s][w] += 1
        tf[d["id"]] = c
        df.update(c.keys())
    n = max(len(tf), 1)
    idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
    return tf, idf, {s: forms.most_common(1)[0][0] for s, forms in surface.items()}


def bridging_terms(a: Counter, b: Counter, idf: dict, surface: dict, n_docs: int, k: int = 6) -> list[str]:
    max_df_idf = math.log((n_docs + 1) / (0.35 * n_docs + 1)) + 1 if n_docs > 8 else 0
    scores = {}
    for t in a.keys() & b.keys():
        w = idf.get(t, 0)
        if w <= max_df_idf:
            continue  # too common across the whole corpus to be interesting
        scores[t] = (1 + math.log(a[t])) * (1 + math.log(b[t])) * w * w
    return [surface.get(t, t) for t in sorted(scores, key=scores.get, reverse=True)[:k]]


# ------------------------------------------------------------------ semantic edges

def link_distances(edges: list[tuple[int, int, float, dict]], max_depth: int) -> dict[int, dict[int, int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for s, d, *_ in edges:
        adj[s].add(d)
        adj[d].add(s)
    dist: dict[int, dict[int, int]] = {}
    for start in adj:
        seen = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            if seen[u] >= max_depth:
                continue
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    q.append(v)
        dist[start] = seen
    return dist


def semantic_edges(store: Store, index: VectorIndex, link_edges, k: int, floor: float,
                   terms, lexical: bool = False) -> list[tuple[int, int, float, dict]]:
    """Top-k neighbours per document, thresholded against this corpus's own similarity distribution.

    Absolute cosine values differ wildly between embedding models (and corpora), so an edge is kept if
    it is unusually strong (mean + 2.5σ, never below `floor`), and every document keeps its best
    GUARANTEED neighbours if they are at least moderately above background (mean + 0.75σ).
    """
    doc_ids, D = index.doc_matrix()
    n = len(doc_ids)
    if n < 2:
        return []
    k = min(k, n - 1)
    cand: list[tuple[int, int, float, int]] = []  # (i, j, sim, rank among i's neighbours)
    total = total_sq = 0.0
    for start in range(0, n, 512):
        S = D[start:start + 512] @ D.T
        rows = np.arange(S.shape[0])
        S[rows, start + rows] = np.nan
        total += float(np.nansum(S))
        total_sq += float(np.nansum(S * S))
        S = np.nan_to_num(S, nan=-1.0)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for r in rows:
            order = top[r][np.argsort(-S[r, top[r]])]
            cand += [(start + r, int(j), float(S[r, j]), rank) for rank, j in enumerate(order)]
    count = n * (n - 1)
    mean = total / count
    std = max((total_sq / count - mean * mean) ** 0.5, 1e-6)
    strong = max(floor, mean + 2.5 * std)
    weak = max(floor * 0.5, mean + 0.75 * std)

    pairs: dict[tuple[int, int], float] = {}
    for i, j, s, rank in cand:
        if s >= strong or (rank < GUARANTEED and s >= weak):
            key = (int(doc_ids[min(i, j)]), int(doc_ids[max(i, j)]))
            pairs[key] = max(pairs.get(key, 0.0), s)
    if not pairs:
        return []

    tags = {d["id"]: set(d["tags"]) for d in store.all_docs()}
    dist = link_distances(link_edges, DISCOVERY_MIN_DISTANCE - 1)
    tf, idf, surface = terms
    rows_of: dict[int, list[int]] = defaultdict(list)
    for row, d in enumerate(index.doc_ids):
        rows_of[int(d)].append(row)

    out = []
    for (a, b), s in pairs.items():
        hops = dist.get(a, {}).get(b)
        shared = sorted(tags.get(a, set()) & tags.get(b, set()))
        discovered = hops is None and not shared
        bridge = bridging_terms(tf.get(a, Counter()), tf.get(b, Counter()), idf, surface, len(tf))
        if lexical and not bridge:
            continue  # lexical embeddings with no distinctive shared words: noise, not a connection
        data: dict = {"discovered": discovered, "terms": bridge}
        if shared:
            data["shared_tags"] = shared
        if hops is not None:
            data["hops"] = hops
        if discovered:
            ra, rb = rows_of[a], rows_of[b]
            M = index.matrix[ra] @ index.matrix[rb].T
            ia, ib = np.unravel_index(int(np.argmax(M)), M.shape)
            data["evidence"] = [int(index.chunk_ids[ra[ia]]), int(index.chunk_ids[rb[ib]])]
            data["evidence_similarity"] = round(float(M[ia, ib]), 4)
        out.append((a, b, round(s, 4), data))
    return out


# ------------------------------------------------------------------ clusters

def label_propagation(nodes: list[int], edges, iterations: int = 30) -> dict[int, int]:
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for s, d, w, *_ in edges:
        adj[s].append((d, w))
        adj[d].append((s, w))
    label = {n: n for n in nodes}
    order = sorted(nodes)
    for it in range(iterations):
        changed = 0
        # deterministic but not always the same direction, to avoid oscillation
        for n in (order if it % 2 == 0 else reversed(order)):
            if not adj[n]:
                continue
            votes: dict[int, float] = defaultdict(float)
            for m, w in adj[n]:
                votes[label[m]] += w
            best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != label[n]:
                label[n] = best
                changed += 1
        if not changed:
            break
    # relabel by cluster size: 0 = biggest
    sizes = Counter(label.values())
    ranking = {lab: i for i, (lab, _) in enumerate(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])))}
    return {n: ranking[l] for n, l in label.items()}


def cluster_labels(clusters: dict[int, int], terms, k: int = 3) -> dict[int, dict]:
    tf, idf, surface = terms
    members: dict[int, list[int]] = defaultdict(list)
    for doc, c in clusters.items():
        members[c].append(doc)
    out = {}
    for c, docs in members.items():
        if len(docs) < 2:
            continue
        agg: Counter = Counter()
        presence: Counter = Counter()
        for d in docs:
            agg.update(tf.get(d, Counter()))
            presence.update(tf.get(d, Counter()).keys())
        # terms that are frequent here, present in several members, and rare elsewhere
        scores = {t: (1 + math.log(cnt)) * idf.get(t, 1) * (presence[t] / len(docs))
                  for t, cnt in agg.items() if presence[t] >= min(2, len(docs))}
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        out[c] = {"label": " · ".join(surface.get(t, t) for t in top) or f"cluster {c + 1}", "size": len(docs)}
    return out


# ------------------------------------------------------------------ entry points

def build_graph(store: Store, index: VectorIndex, embedder: Embedder, settings) -> dict:
    links = resolve_links(store)
    store.replace_edges("link", links)
    floor = settings.min_similarity or embedder.min_similarity
    terms = term_stats(store)
    sem = semantic_edges(store, index, links, settings.neighbors_per_doc, floor, terms, embedder.idf_weighting)
    store.replace_edges("semantic", sem)

    nodes = [d["id"] for d in store.all_docs()]
    weighted = [(s, d, 1.0) for s, d, *_ in links] + [(s, d, w) for s, d, w, _ in sem]
    clusters = label_propagation(nodes, weighted)
    store.set_meta("clusters", {str(k): v for k, v in clusters.items()})
    store.set_meta("cluster_labels", {str(k): v for k, v in cluster_labels(clusters, terms).items()})
    return {"links": len(links), "semantic": len(sem), "discovered": sum(1 for *_, d in sem if d["discovered"])}


def graph_payload(store: Store) -> dict:
    docs = store.all_docs()
    clusters = store.get_meta("clusters", {})
    labels = store.get_meta("cluster_labels", {})
    edges = store.edges()
    degree: Counter = Counter()
    for e in edges:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
    nodes = [{"id": d["id"], "title": d["title"], "kind": d["kind"], "tags": d["tags"], "words": d["words"],
              "cluster": clusters.get(str(d["id"]), -1), "degree": degree[d["id"]]} for d in docs]
    links = [{"source": e["src"], "target": e["dst"], "kind": e["kind"], "weight": e["weight"],
              "discovered": bool(e["data"].get("discovered"))} for e in edges]
    return {"nodes": nodes, "edges": links,
            "clusters": [{"id": int(k), **v} for k, v in sorted(labels.items(), key=lambda kv: int(kv[0]))]}


def discovered_connections(store: Store, limit: int = 50) -> list[dict]:
    rows = store.conn.execute("""
        SELECT e.src, e.dst, e.weight, e.data, a.title AS a_title, a.kind AS a_kind, b.title AS b_title, b.kind AS b_kind
        FROM edges e JOIN documents a ON a.id=e.src JOIN documents b ON b.id=e.dst
        WHERE e.kind='semantic' AND json_extract(e.data, '$.discovered')
        ORDER BY e.weight DESC LIMIT ?""", (limit,)).fetchall()
    out = []
    evidence_ids = []
    for r in rows:
        data = json.loads(r["data"])
        evidence_ids += data.get("evidence", [])
        out.append({"a": {"id": r["src"], "title": r["a_title"], "kind": r["a_kind"]},
                    "b": {"id": r["dst"], "title": r["b_title"], "kind": r["b_kind"]},
                    "similarity": r["weight"], "terms": data.get("terms", []), "evidence": data.get("evidence", [])})
    chunks = store.chunks_by_ids(evidence_ids)
    for c in out:
        c["evidence"] = [{"chunk_id": cid, "text": chunks[cid]["text"][:400], "heading": chunks[cid]["heading"],
                          "page": chunks[cid]["page"]} for cid in c["evidence"] if cid in chunks]
    return out
