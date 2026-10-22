"""Builds a realistic home folder for discovery tests."""

import json
import shutil
import sqlite3
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent.parent / "sample_vault"


def build(home: Path) -> Path:
    # an Obsidian vault in ~/Notes
    vault = home / "Notes" / "Brain"
    shutil.copytree(SAMPLE, vault, ignore=shutil.ignore_patterns("bookmarks.html"))
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "app.json").write_text("{}")
    # Documents: some notes, a PDF, and a code project whose README must be ignored
    docs = home / "Documents"
    (docs / "uni").mkdir(parents=True)
    (docs / "uni" / "Thesis outline.md").write_text("# Thesis outline\nSwarm intelligence and distributed consensus.\n")
    (docs / "uni" / "Reading list.txt").write_text("Seeley - Honeybee Democracy\nMeadows - Thinking in Systems\n")
    shutil.copy(SAMPLE / "papers" / "ant-colony-optimization.pdf", docs / "aco.pdf")
    app = docs / "code" / "webapp"
    (app / ".git").mkdir(parents=True)
    (app / "package.json").write_text("{}")
    (app / "README.md").write_text("# webapp\nnpm install && npm start\n")
    (app / "node_modules" / "left-pad").mkdir(parents=True)
    (app / "node_modules" / "left-pad" / "README.md").write_text("left pad")
    # Downloads: listed but not pre-selected
    (home / "Downloads").mkdir()
    shutil.copy(SAMPLE / "papers" / "ant-colony-optimization.pdf", home / "Downloads" / "invoice.pdf")
    # media and system folders are skipped
    (home / "Pictures").mkdir()
    (home / "Pictures" / "notes.md").write_text("should not be counted")
    (home / ".hidden-stuff").mkdir()
    (home / ".hidden-stuff" / "secret.md").write_text("should not be counted")
    # Chrome profile with live Bookmarks file
    chrome = home / ".config" / "google-chrome" / "Default"
    chrome.mkdir(parents=True)
    (chrome / "Bookmarks").write_text(json.dumps({"roots": {"bookmark_bar": {"type": "folder", "name": "Bookmarks bar", "children": [
        {"type": "folder", "name": "Research", "children": [
            {"type": "url", "name": "Raft Consensus Algorithm", "url": "https://raft.github.io/"},
            {"type": "url", "name": "Honeybee Democracy", "url": "https://press.princeton.edu/books/honeybee-democracy"}]}]}}}))
    # Firefox profile with places.sqlite
    ff = home / ".mozilla" / "firefox" / "abc123.default-release"
    ff.mkdir(parents=True)
    con = sqlite3.connect(ff / "places.sqlite")
    con.executescript("""
        CREATE TABLE moz_places(id INTEGER PRIMARY KEY, url TEXT, title TEXT);
        CREATE TABLE moz_bookmarks(id INTEGER PRIMARY KEY, type INTEGER, fk INTEGER, parent INTEGER, title TEXT);
        INSERT INTO moz_places VALUES (1, 'https://ncase.me/loopy/', 'LOOPY'), (2, 'place:sort=8', NULL);
        INSERT INTO moz_bookmarks VALUES (1, 2, NULL, 0, ''), (2, 2, NULL, 1, 'toolbar'),
            (3, 2, NULL, 2, 'Systems'), (4, 1, 1, 3, 'LOOPY: thinking in systems'), (5, 1, 2, 2, 'Most visited');
    """)
    con.commit()
    con.close()
    return home
