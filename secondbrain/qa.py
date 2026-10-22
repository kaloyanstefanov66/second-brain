"""Answer questions over your files, with numbered citations.

The answer is written by the selected engine (see llm.py): a local model by default, or a
cloud provider you added a key for. Without one, an extractive answer is assembled from the
most relevant sentences, each cited.
"""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np

from .chunking import split_sentences
from .embeddings import normalize, tokenize
from .llm import ProviderError, detect_llm
from .search import Hit, Searcher

SYSTEM_PROMPT = """You are the user's second brain. Answer the question using ONLY the numbered excerpts \
from their personal notes and documents.

Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [1] or [2][4], right after the claim.
- If the excerpts do not contain the answer, say so plainly instead of guessing.
- Be concise and concrete. Prefer the user's own wording and terminology.
- Never cite a number that is not in the excerpts."""


# ------------------------------------------------------------------ prompt building

def source_label(h: Hit) -> str:
    bits = [h.title]
    if h.heading:
        bits.append(h.heading)
    if h.page:
        bits.append(f"p. {h.page}")
    return " — ".join(bits)


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    excerpts = "\n\n".join(f"[{i}] ({source_label(h)})\n{h.text}" for i, h in enumerate(hits, 1))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            msgs.append({"role": turn["role"], "content": turn["content"][:2000]})
    msgs.append({"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"})
    return msgs


def source_dict(i: int, h: Hit) -> dict:
    return {"n": i, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "kind": h.kind,
            "heading": h.heading, "page": h.page, "url": h.url, "text": h.text, "similarity": h.similarity}


# ------------------------------------------------------------------ extractive fallback

def extractive_answer(question: str, hits: list[Hit], embedder, max_sentences: int = 4) -> str:
    q_terms = set(tokenize(question))
    candidates: list[tuple[int, str]] = []
    for i, h in enumerate(hits, 1):
        text = h.text
        if h.kind == "bookmark":  # skip the URL / folder lines, keep the description
            lines = [l for l in text.splitlines()[1:] if not l.startswith(("http://", "https://", "Folder:"))]
            text = " ".join(lines) or h.title
        for s in split_sentences(text.replace("\n", " ")):
            if 30 <= len(s) <= 500:
                candidates.append((i, s))
    if not candidates:
        return "I couldn't find anything in your files that answers this."
    vecs = normalize(embedder.embed([s for _, s in candidates]))
    q = normalize(embedder.embed_query(question)[None, :])[0]
    sims = vecs @ q
    scores = []
    for (i, s), sim in zip(candidates, sims):
        toks = set(tokenize(s))
        overlap = len(q_terms & toks) / max(len(q_terms), 1)
        scores.append(float(sim) + 0.6 * overlap + 0.02 * (len(hits) - i))  # slight bias to top-ranked sources
    picked: list[tuple[int, str]] = []
    picked_toks: list[set] = []
    for idx in np.argsort(scores)[::-1]:
        i, s = candidates[idx]
        toks = set(tokenize(s))
        if any(len(toks & p) / max(len(toks | p), 1) > 0.5 for p in picked_toks):
            continue  # near-duplicate (chunk overlap)
        if scores[idx] < 0.15 and picked:
            break
        picked.append((i, s))
        picked_toks.append(toks)
        if len(picked) >= max_sentences:
            break
    lines = [f"- {s} [{i}]" for i, s in picked]
    return "Here's what your files say:\n\n" + "\n".join(lines)


# ------------------------------------------------------------------ main entry

def answer(question: str, searcher: Searcher, settings, k: int = 8,
           history: list[dict] | None = None, use_llm: bool = True) -> Iterator[dict]:
    """Yield events: sources, meta, delta (answer text), done | error."""
    hits = searcher.search(question, k=k, max_per_doc=3)
    yield {"type": "sources", "sources": [source_dict(i, h) for i, h in enumerate(hits, 1)]}
    if not hits:
        yield {"type": "meta", "mode": "none"}
        yield {"type": "delta", "text": "Your knowledge base is empty or nothing matched. Add some sources first."}
        yield {"type": "done"}
        return

    llm = detect_llm(settings) if use_llm else None
    if llm:
        yield {"type": "meta", "mode": "llm", "model": llm.name, "cloud": llm.cloud}
        try:
            for piece in llm.stream(build_messages(question, hits, history)):
                yield {"type": "delta", "text": piece}
            yield {"type": "done"}
            return
        except (ProviderError, OSError, ValueError) as e:
            yield {"type": "delta", "text": f"\n\n_({llm.name} failed: {e}. Falling back to quoting your files.)_\n\n"}
    yield {"type": "meta", "mode": "extractive"}
    yield {"type": "delta", "text": extractive_answer(question, hits, searcher.embedder)}
    yield {"type": "done"}


def answer_text(question: str, searcher: Searcher, settings, **kw) -> tuple[str, list[dict], dict]:
    text, sources, meta = [], [], {}
    for ev in answer(question, searcher, settings, **kw):
        if ev["type"] == "sources":
            sources = ev["sources"]
        elif ev["type"] == "meta":
            meta = ev
        elif ev["type"] == "delta":
            text.append(ev["text"])
    return "".join(text), sources, meta


def cited_numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}
