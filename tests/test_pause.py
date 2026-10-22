import shutil
import time
from pathlib import Path

from secondbrain.brain import Brain
from secondbrain.config import Settings

SAMPLE = Path(__file__).resolve().parent.parent / "sample_vault"


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_pause_stops_indexing_and_resume_finishes(tmp_path):
    vault = tmp_path / "vault"
    shutil.copytree(SAMPLE, vault)
    settings = Settings.load(tmp_path / "home", embedder="hashing", llm="none")
    brain = Brain(settings)
    brain.pause()
    assert brain.status()["paused"]
    assert brain.run_in_background(brain.add, [vault])

    assert wait_for(lambda: brain.progress.get("stage") == "paused")
    time.sleep(0.5)
    assert brain.store.stats()["documents"] == 0, "nothing is indexed while paused"
    assert brain.busy

    # the pause is remembered: a fresh indexer on the same library starts paused
    assert Brain(settings).indexer.paused

    brain.resume()
    brain._job.join(30)
    assert not brain.busy and not brain.status()["paused"]
    assert brain.store.stats()["documents"] == 17
    assert not Brain(settings).indexer.paused


def test_unchanged_reindex_skips_graph_rebuild(tmp_path):
    vault = tmp_path / "vault"
    shutil.copytree(SAMPLE, vault)
    brain = Brain(Settings.load(tmp_path / "home", embedder="hashing", llm="none"))
    first = brain.add([vault])
    assert first.graph
    again = brain.reindex()
    assert again.unchanged == 17 and again.graph == {}
    assert brain.progress["stage"] == "idle"
