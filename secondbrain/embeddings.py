"""Local embedding backends. Nothing here sends your text to a third party.

- FastEmbedEmbedder: ONNX sentence-embedding model on CPU (downloads ~70MB once).
- OllamaEmbedder: any embedding model served by a local Ollama.
- HashingEmbedder: dependency-free, zero-download lexical embedding. Always available,
  and weighted by corpus IDF at query time (see search.VectorIndex), so it works decently
  for keyword-ish similarity even with no model installed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
import urllib.error
import urllib.request
import zlib
from typing import Sequence

import numpy as np

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves one two may might must shall us get got like make
made many much use used using via etc e g ie eg vs per new also within without
need needs every first next last long longer day days week weeks time times way ways good better best
thing things lot lots really still even back well around today yesterday tomorrow always never often
sometimes usually twice take takes took put go goes going went come comes see saw know think want say
said look find keep let give another sure maybe quite rather enough else ever yet already though
""".split())

TOKEN_RE = re.compile(r"[^\W_]{2,}", re.U)


def stem(w: str) -> str:
    for suf in ("ization", "ational", "fulness", "ousness", "iveness", "ations", "ation", "ments",
                "ment", "ness", "ings", "ing", "ies", "ied", "edly", "ed", "ers", "er", "ly", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            if suf in ("ies", "ied"):
                w += "y"
            break
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and not t.isdigit()]


def normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-9)


class Embedder:
    id: str = "base"           # stored in the DB; changing it triggers a re-embed
    dim: int = 0
    min_similarity: float = 0.5  # calibrated floor for "these documents are related"
    idf_weighting: bool = False

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def describe(self) -> str:
        return self.id


class HashingEmbedder(Embedder):
    min_similarity = 0.12
    idf_weighting = True

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.id = f"hashing-v1-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        counts: dict[str, float] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1.0
        for a, b in zip(toks, toks[1:]):
            key = f"{a}_{b}"
            counts[key] = counts.get(key, 0) + 0.5
        for feat, c in counts.items():
            h = zlib.crc32(feat.encode())
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + np.log(c)) if c >= 1 else sign * c
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def describe(self):
        return "built-in lexical embeddings (install fastembed or Ollama for semantic search)"


class FastEmbedEmbedder(Embedder):
    min_similarity = 0.62

    def __init__(self, model: str, cache_dir: str, threads: int | None = None):
        from fastembed import TextEmbedding  # optional dependency

        self.model_name = model
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=threads)
        self.dim = len(next(iter(self._model.embed(["probe"]))))
        self.id = f"fastembed:{model}"

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return normalize(np.stack(list(self._model.embed(list(texts), batch_size=32))))

    def embed_query(self, text):
        return normalize(np.stack(list(self._model.query_embed([text]))))[0]

    def describe(self):
        return f"{self.model_name} (local ONNX)"


class OllamaEmbedder(Embedder):
    min_similarity = 0.6

    def __init__(self, url: str, model: str, timeout: float = 120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._nomic = "nomic" in model
        self.dim = len(self._post(["probe"])[0])
        self.id = f"ollama:{model}"

    def _post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.url}/api/embed", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["embeddings"]

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        prefix = "search_document: " if self._nomic else ""
        out = []
        for i in range(0, len(texts), 64):
            out += self._post([prefix + t for t in texts[i:i + 64]])
        return normalize(np.array(out))

    def embed_query(self, text):
        prefix = "search_query: " if self._nomic else ""
        return normalize(np.array(self._post([prefix + text])))[0]

    def describe(self):
        return f"{self.model} via local Ollama"


def embed_threads(settings) -> int:
    """Leave half the cores to the rest of the computer unless told otherwise."""
    return settings.index_threads or max(1, (os.cpu_count() or 2) // 2)


def lower_priority() -> None:
    """Run at background priority: the OS serves your other apps first while indexing runs."""
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x4000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass


def _model_cached(settings, model: str | None = None) -> bool:
    name = (model or settings.fastembed_model).split("/")[-1].lower()
    return any(name in p.name.lower() for p in settings.models_dir.glob("*"))


def _can_download(timeout: float = 3.0) -> bool:
    """Quick reachability check, so an offline first run doesn't sit through fastembed's retries."""
    try:
        urllib.request.urlopen("https://huggingface.co/api/models?limit=1", timeout=timeout).close()
        return True
    except (OSError, ValueError):
        return False


HASHING_VERSION = 1  # bump only together with a new HashingEmbedder version: libraries built with it re-embed


def _setting_default(name: str):
    from .config import Settings

    return Settings.__dataclass_fields__[name].default


def _fastembed_ready(settings, model: str) -> bool:
    import fastembed  # noqa: F401  (fail fast if not installed)

    return _model_cached(settings, model) or _can_download()


def _try(kind: str, settings, model: str | None = None) -> Embedder | None:
    try:
        if kind == "fastembed":
            model = model or settings.fastembed_model
            if not _fastembed_ready(settings, model):
                return None
            return FastEmbedEmbedder(model, str(settings.models_dir), embed_threads(settings))
        if kind == "ollama":
            return OllamaEmbedder(settings.ollama_url, model or settings.ollama_embed_model)
        if kind == "hashing":
            return HashingEmbedder(int(model) if model else 2048)
    except Exception:
        return None
    raise ValueError(f"unknown embedder {kind!r} (use auto, fastembed, ollama or hashing)")


def parse_id(embedder_id: str | None) -> tuple[str | None, str | None]:
    """'fastembed:BAAI/bge-small-en-v1.5' -> ('fastembed', 'BAAI/bge-small-en-v1.5'); 'hashing-v1-2048' -> ('hashing', '2048')."""
    if not embedder_id:
        return None, None
    if embedder_id.startswith("hashing-v"):
        _, version, dim = embedder_id.split("-")
        return ("hashing", dim) if version == f"v{HASHING_VERSION}" else ("hashing-old", None)
    kind, _, model = embedder_id.partition(":")
    return kind, model or None


def kind_of(embedder_id: str | None) -> str | None:
    return parse_id(embedder_id)[0]


@dataclass
class EmbedderChoice:
    embedder: Embedder
    degraded: bool = False  # the library's own model is unavailable: keep its vectors, don't re-embed
    note: str = ""


def resolve_embedder(settings, stored_id: str | None = None) -> EmbedderChoice:
    """Pick the embedder, protecting an existing library from needless re-embedding.

    An existing library keeps the exact model it was built with, even if a newer version of the app
    ships a different default. Only an explicit choice (`--embedder`, or a model set in config.toml)
    switches models. If the library's model can't be loaded right now, fall back to keyword search
    instead of re-embedding everything with something else.
    """
    explicit = settings.embedder != "auto"
    kind, model = parse_id(stored_id)
    if kind in ("fastembed", "ollama", "hashing") and (not explicit or settings.embedder == kind):
        for name, setting in (("fastembed", "fastembed_model"), ("ollama", "ollama_embed_model")):
            if kind == name and getattr(settings, setting) != _setting_default(setting):
                model = getattr(settings, setting)  # you configured a model on purpose: honour it
        e = _try(kind, settings, model)
        if e is not None:
            return EmbedderChoice(e)
        if explicit:
            raise RuntimeError(f"embedder {settings.embedder!r} is not available")
        fallback = next((x for x in (_try(k, settings) for k in ("ollama", "hashing") if k != kind) if x), HashingEmbedder())
        what = f"{model} ({kind})" if model and kind != "hashing" else kind
        return EmbedderChoice(fallback, degraded=True,
                              note=f"{what} couldn't be loaded, so search is keyword-only for now. "
                                   "Your index is kept as is and nothing is re-embedded.")
    if explicit:
        e = _try(settings.embedder, settings)
        if e is None:
            raise RuntimeError(f"embedder {settings.embedder!r} is not available")
        return EmbedderChoice(e)
    for k in ("fastembed", "ollama", "hashing"):
        e = _try(k, settings)
        if e is not None:
            return EmbedderChoice(e)
    return EmbedderChoice(HashingEmbedder())


def get_embedder(settings, stored_id: str | None = None) -> Embedder:
    return resolve_embedder(settings, stored_id).embedder
