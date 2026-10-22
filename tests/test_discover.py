import json
from pathlib import Path

from fastapi.testclient import TestClient

from fakehome import build
from secondbrain.brain import Brain
from secondbrain.config import Settings
from secondbrain.discover import discover, outermost
from secondbrain.loaders import iter_files, load_file
from secondbrain.server import create_app


def test_discovers_vaults_folders_and_browsers(tmp_path):
    home = build(tmp_path / "home")
    found = {c.label: c for c in discover(home)}
    assert found["Brain"].kind == "vault" and found["Brain"].app == "Obsidian" and found["Brain"].selected
    docs = found["Documents"]
    assert docs.kind == "folder" and docs.selected and (docs.notes, docs.docs) == (2, 1)  # code project skipped
    assert not found["Downloads"].selected
    assert found["Notes"].inside == ["Brain"]
    assert "Pictures" not in found and ".hidden-stuff" not in found
    assert found["Chrome"].bookmarks == 2 and found["Chrome"].selected
    assert found["Firefox (default-release)"].bookmarks == 1


def test_browser_stores_load_in_place(tmp_path):
    home = build(tmp_path / "home")
    ff = load_file(next(home.glob(".mozilla/firefox/*/places.sqlite")))
    assert [(d.title, d.url, d.tags) for d in ff] == [("LOOPY: thinking in systems", "https://ncase.me/loopy/", ["systems"])]
    chrome = load_file(home / ".config/google-chrome/Default/Bookmarks")
    assert {d.title for d in chrome} == {"Raft Consensus Algorithm", "Honeybee Democracy"}
    assert all(d.tags == ["research"] for d in chrome)


def test_walk_skips_code_projects_and_tool_folders(tmp_path):
    home = build(tmp_path / "home")
    names = {p.name for p in iter_files(home / "Documents")}
    assert names == {"Thesis outline.md", "Reading list.txt", "aco.pdf"}


def test_outermost_drops_nested_paths(tmp_path):
    home = build(tmp_path / "home")
    assert outermost([str(home / "Notes" / "Brain"), str(home / "Notes")]) == [str(home / "Notes")]


def test_one_click_onboarding_to_organised_home(tmp_path, monkeypatch):
    home = build(tmp_path / "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    brain = Brain(Settings.load(tmp_path / "data", embedder="hashing", llm="none"))
    c = TestClient(create_app(brain), base_url="http://localhost")

    cands = c.get("/api/discover").json()["candidates"]
    chosen = [x["path"] for x in cands if x["selected"]]
    r = c.post("/api/sources/bulk", json={"paths": chosen})
    assert r.status_code == 200
    brain._job.join(30)

    kinds = brain.store.stats()["by_kind"]
    assert kinds["note"] >= 13 and kinds["bookmark"] == 3 and kinds["pdf"] == 2
    titles = {d["title"] for d in brain.store.all_docs()}
    assert "webapp" not in titles and "invoice" not in titles  # code README skipped, Downloads not selected

    ov = c.get("/api/overview").json()
    assert ov["total"] == len(brain.store.all_docs()) and ov["topics"]
    placed = sum(len(t["docs"]) for t in ov["topics"]) + len(ov["unsorted"])
    assert placed == ov["total"], "every document appears exactly once on Home"
    assert ov["recent"] and all(d["kind"] != "bookmark" for d in ov["recent"])

    again = {x["path"]: x for x in c.get("/api/discover").json()["candidates"]}
    assert all(again[p]["added"] and not again[p]["selected"] for p in chosen if p in again)
