"""Updating the app must never re-index documents that were already indexed."""

import hashlib
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from secondbrain import embeddings, store as store_mod
from secondbrain.brain import Brain
from secondbrain.config import Settings
from secondbrain.embeddings import Embedder, HashingEmbedder
from secondbrain.store import Store

SAMPLE = Path(__file__).resolve().parent.parent / "sample_vault"


@pytest.fixture()
def vault(tmp_path):
    dst = tmp_path / "vault"
    shutil.copytree(SAMPLE, dst)
    return dst


class Counter:
    def __init__(self):
        self.texts = 0


@pytest.fixture()
def count_embeds(monkeypatch):
    """Count every text that goes through any embedder (indexing, re-embedding, backfill)."""
    counter = Counter()
    original = HashingEmbedder.embed

    def counting(self, texts):
        counter.texts += len(texts)
        return original(self, texts)

    monkeypatch.setattr(HashingEmbedder, "embed", counting)
    return counter


class FakeFastEmbed(Embedder):
    """Stands in for the ONNX model (no download in tests). `available` simulates it failing to load."""

    available = True
    built: list = []

    def __init__(self, model, cache_dir, threads=None):
        if not FakeFastEmbed.available:
            raise RuntimeError("model failed to load")
        self.model_name, self.dim, self.id = model, 64, f"fastembed:{model}"
        self._inner = HashingEmbedder(64)
        FakeFastEmbed.built.append(model)

    def embed(self, texts):
        return self._inner.embed(texts)

    def describe(self):
        return self.model_name


@pytest.fixture()
def fake_fastembed(monkeypatch):
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", FakeFastEmbed)
    monkeypatch.setattr(embeddings, "_fastembed_ready", lambda settings, model: True)
    FakeFastEmbed.available, FakeFastEmbed.built = True, []
    return FakeFastEmbed


def settings(tmp_path, **kw):
    return Settings.load(tmp_path / "home", llm="none", **kw)


def null_embeddings(brain) -> int:
    return brain.store.conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]


def test_new_app_version_reuses_the_index(tmp_path, vault, count_embeds):
    old = Brain(settings(tmp_path, embedder="hashing"))
    old.add([vault])
    assert count_embeds.texts > 0

    count_embeds.texts = 0
    new = Brain(settings(tmp_path))  # the updated app, default settings ("auto")
    assert new.embedder.id == old.embedder.id
    assert new.ensure_ready() is False
    report = new.reindex()  # what the app does on every launch
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 17)
    assert report.graph == {}, "nothing changed, so the graph isn't rebuilt either"
    assert count_embeds.texts == 0, "not a single passage was re-embedded"
    assert new.search("quorum sensing bees", k=1)[0].title == "Beekeeping Log"


def test_new_default_model_does_not_replace_the_librarys_model(tmp_path, vault, fake_fastembed):
    Brain(settings(tmp_path, embedder="fastembed", fastembed_model="old-org/old-model")).add([vault])
    # A later release ships a different default model; the user changed nothing.
    new = Brain(settings(tmp_path))
    assert new.embedder.id == "fastembed:old-org/old-model"
    assert fake_fastembed.built[-1] == "old-org/old-model"
    assert new.ensure_ready() is False


def test_configured_model_change_is_respected(tmp_path, vault, fake_fastembed):
    Brain(settings(tmp_path, embedder="fastembed", fastembed_model="old-org/old-model")).add([vault])
    new = Brain(settings(tmp_path, fastembed_model="new-org/new-model"))  # set on purpose in config.toml
    assert new.embedder.id == "fastembed:new-org/new-model"
    assert new.ensure_ready() is True  # a deliberate switch re-embeds


def test_unavailable_model_keeps_index_and_backfills_only_new_passages(tmp_path, vault, fake_fastembed):
    first = Brain(settings(tmp_path))
    first.add([vault])
    library_id = first.store.get_meta("embedder")["id"]
    assert library_id.startswith("fastembed:")
    semantic_before = first.store.stats()["semantic"]
    assert semantic_before > 0

    fake_fastembed.available = False  # e.g. a broken model cache after an update
    degraded = Brain(settings(tmp_path))
    status = degraded.status()["embedder"]
    assert status["degraded"] and "keyword-only" in status["note"]
    assert degraded.ensure_ready() is False
    assert degraded.store.get_meta("embedder")["id"] == library_id, "the library's model is not forgotten"

    (vault / "Zebra migration.md").write_text("# Zebra migration\nZebras cross the Mara river every July.\n")
    report = degraded.reindex()
    assert (report.added, report.updated) == (1, 0)
    new_passages = null_embeddings(degraded)
    assert new_passages >= 1
    assert degraded.store.stats()["semantic"] == semantic_before, "connections are kept"
    assert degraded.search("zebras mara river", k=1)[0].title == "Zebra migration"  # keyword search works
    assert degraded.store.get_meta("embedder")["id"] == library_id

    fake_fastembed.available = True
    embedded = []
    original = FakeFastEmbed.embed
    FakeFastEmbed.embed = lambda self, texts: embedded.extend(texts) or original(self, texts)
    try:
        healed = Brain(settings(tmp_path))
        assert healed.ensure_ready() is True
    finally:
        FakeFastEmbed.embed = original
    assert len(embedded) == new_passages, "only the passages stored without vectors were embedded"
    assert null_embeddings(healed) == 0


def test_old_libraries_are_upgraded_in_place(tmp_path, vault, monkeypatch):
    brain = Brain(settings(tmp_path, embedder="hashing"))
    brain.add([vault])
    db = brain.settings.db_path
    assert sqlite3.connect(db).execute("PRAGMA user_version").fetchone()[0] == store_mod.SCHEMA_VERSION

    con = sqlite3.connect(db)
    con.execute("PRAGMA user_version = 0")  # a library made by 0.1.0 / 0.1.1
    con.commit()
    con.close()
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(store_mod, "MIGRATIONS", {2: "ALTER TABLE documents ADD COLUMN pinned INTEGER DEFAULT 0;"})
    upgraded = Store(db)
    cols = {r[1] for r in upgraded.conn.execute("PRAGMA table_info(documents)")}
    assert "pinned" in cols
    assert upgraded.stats()["documents"] == 17, "documents survive the migration"
    assert upgraded.conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_library_from_newer_version_is_refused_not_damaged(tmp_path):
    db = tmp_path / "brain.db"
    Store(db)
    con = sqlite3.connect(db)
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="newer version"):
        Store(db)


def test_builtin_embedder_output_is_pinned():
    """Changing HashingEmbedder's output silently would mix incompatible vectors in existing libraries.
    If you change it on purpose, bump HASHING_VERSION (libraries built with it then re-embed) and update this."""
    v = HashingEmbedder().embed(["The quick brown fox jumps over the lazy dog while bees decide by quorum."])[0]
    digest = hashlib.sha256(np.round(v, 5).astype(np.float32).tobytes()).hexdigest()
    assert digest == "ff067fc146b9242817d82c6ab8e3d1b70b63339d108491eddbfec3c13134e4d3"
