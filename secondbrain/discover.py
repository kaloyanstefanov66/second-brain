"""Find where your notes live, so you don't have to point at every folder.

Discovery only looks at folder structure and file names (it opens browser bookmark files to count
them). Nothing is indexed until you confirm the selection. It looks for:

  * Obsidian vaults and Logseq graphs anywhere in your home folder (and Obsidian's iCloud folder)
  * folders under your home folder that contain notes or documents (Documents, Desktop, Notes, cloud drives...)
  * bookmarks in installed browsers: Chrome, Edge, Brave, Vivaldi, Arc, Chromium, Firefox, Safari
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .loaders import load_file, prune_dirs

NOTE_EXT = {".md", ".markdown", ".mdx", ".txt", ".org", ".rst"}
DOC_EXT = {".pdf", ".docx"}
# Top-level folders that hold system files or media, not notes.
SKIP_TOP = {"Library", "AppData", "Applications", "Pictures", "Music", "Movies", "Videos", "Photos", "snap",
            "go", "miniconda3", "anaconda3", "Public", "Templates", "Creative Cloud Files", "VirtualBox VMs"}
LARGE = 5000  # files; bigger folders are listed but not pre-selected


@dataclass
class Candidate:
    path: str
    kind: str                 # vault | logseq | folder | bookmarks
    label: str
    app: str = ""
    notes: int = 0
    docs: int = 0
    bookmarks: int = 0
    selected: bool = True
    added: bool = False
    partial: bool = False     # counting stopped at the time limit
    detail: str = ""
    inside: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _scan(top: Path, deadline: float) -> tuple[int, int, list[tuple[str, str]], bool]:
    """Count notes/documents under `top` and find vaults. Returns (notes, docs, vaults, partial)."""
    notes = docs = 0
    vaults: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(top):
        if time.monotonic() > deadline:
            return notes, docs, vaults, True
        names = set(dirnames)
        if ".obsidian" in names:
            vaults.append((dirpath, "vault"))
        elif "logseq" in names and "pages" in names:
            vaults.append((dirpath, "logseq"))
        prune_dirs(dirpath, dirnames, filenames)
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in NOTE_EXT:
                notes += 1
            elif ext in DOC_EXT:
                docs += 1
    return notes, docs, vaults, False


def _browser_stores(home: Path) -> list[tuple[str, Path]]:
    """(browser label, bookmark file) for every installed browser profile."""
    mac = home / "Library" / "Application Support"
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    roaming = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    chromium = {
        "Chrome": [mac / "Google/Chrome", home / ".config/google-chrome", local / "Google/Chrome/User Data"],
        "Edge": [mac / "Microsoft Edge", home / ".config/microsoft-edge", local / "Microsoft/Edge/User Data"],
        "Brave": [mac / "BraveSoftware/Brave-Browser", home / ".config/BraveSoftware/Brave-Browser",
                  local / "BraveSoftware/Brave-Browser/User Data"],
        "Vivaldi": [mac / "Vivaldi", home / ".config/vivaldi", local / "Vivaldi/User Data"],
        "Arc": [mac / "Arc/User Data"],
        "Chromium": [mac / "Chromium", home / ".config/chromium", local / "Chromium/User Data"],
        "Opera": [mac / "com.operasoftware.Opera", home / ".config/opera", roaming / "Opera Software/Opera Stable"],
    }
    found: list[tuple[str, Path]] = []
    for name, roots in chromium.items():
        for root in roots:
            if not root.is_dir():
                continue
            profiles = [root] + sorted(p for p in root.iterdir() if p.is_dir() and (p.name == "Default" or p.name.startswith("Profile ")))
            for prof in profiles:
                f = prof / "Bookmarks"
                if f.is_file():
                    label = name if prof == root or prof.name == "Default" else f"{name} ({prof.name})"
                    found.append((label, f))
    for root in (mac / "Firefox/Profiles", home / ".mozilla/firefox", home / "snap/firefox/common/.mozilla/firefox",
                 roaming / "Mozilla/Firefox/Profiles"):
        if root.is_dir():
            for f in sorted(root.glob("*/places.sqlite")):
                found.append((f"Firefox ({f.parent.name.split('.', 1)[-1]})", f))
    safari = home / "Library/Safari/Bookmarks.plist"
    if safari.is_file():
        found.append(("Safari", safari))
    return found


def _icloud_vaults(home: Path) -> list[Path]:
    root = home / "Library/Mobile Documents/iCloud~md~obsidian/Documents"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def discover(home: Path | None = None, added_roots: list[str] | None = None, budget: float = 12.0) -> list[Candidate]:
    home = (home or Path.home()).expanduser().resolve()
    added = [Path(r) for r in (added_roots or [])]
    deadline = time.monotonic() + budget
    out: list[Candidate] = []

    tops = []
    try:
        tops = sorted(p for p in home.iterdir() if p.is_dir() and not p.name.startswith(".")
                      and p.name not in SKIP_TOP)
    except OSError:
        pass
    icloud = home / "Library/Mobile Documents/com~apple~CloudDocs"
    if icloud.is_dir():
        tops.append(icloud)

    vaults: list[tuple[str, str]] = [(str(v), "vault") for v in _icloud_vaults(home)]
    per_folder_budget = max(budget / max(len(tops), 1), 1.0)
    for top in tops:
        notes, docs, found, partial = _scan(top, min(deadline, time.monotonic() + per_folder_budget * 2))
        vaults += found
        if notes + docs == 0:
            continue
        name = "iCloud Drive" if top == icloud else top.name
        c = Candidate(str(top), "folder", name, notes=notes, docs=docs, partial=partial)
        if top.name.lower() == "downloads":
            c.selected, c.detail = False, "Usually downloads, not notes"
        elif notes + docs > LARGE:
            c.selected, c.detail = False, "Very large: select it if it's really your notes"
        out.append(c)

    for path, kind in dict.fromkeys(vaults):
        notes, docs, _, partial = _scan(Path(path), time.monotonic() + 3)
        out.append(Candidate(path, kind, Path(path).name, app="Obsidian" if kind == "vault" else "Logseq",
                             notes=notes, docs=docs, partial=partial))

    for label, f in _browser_stores(home):
        try:
            n = len(load_file(f))
        except Exception as e:  # e.g. Safari without Full Disk Access
            out.append(Candidate(str(f), "bookmarks", label, app=label.split(" (")[0], selected=False,
                                 detail=f"Can't read: {type(e).__name__}"))
            continue
        if n:
            out.append(Candidate(str(f), "bookmarks", label, app=label.split(" (")[0], bookmarks=n))

    # nesting: a vault inside a selected folder is covered by that folder
    folders = [c for c in out if c.kind == "folder"]
    for c in out:
        if c.kind in ("vault", "logseq"):
            for f in folders:
                if Path(c.path).is_relative_to(f.path):
                    f.inside.append(c.label)
        p = Path(c.path)
        c.added = any(p == r or p.is_relative_to(r) for r in added)
        if c.added:
            c.selected = False

    order = {"vault": 0, "logseq": 0, "folder": 1, "bookmarks": 2}
    out.sort(key=lambda c: (order[c.kind], -(c.notes + c.docs + c.bookmarks)))
    return out


def outermost(paths: list[str]) -> list[str]:
    """Drop paths nested inside other selected paths (they'd be indexed twice)."""
    ps = sorted({str(Path(p).expanduser().resolve()) for p in paths}, key=len)
    keep: list[str] = []
    for p in ps:
        if not any(Path(p).is_relative_to(k) for k in keep if Path(k).is_dir()):
            keep.append(p)
    return keep
