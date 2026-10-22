"""Turn files on disk into LoadedDoc objects: text sections plus metadata, tags and links.

Supported: Markdown/plain text (Obsidian-style [[wikilinks]] and #tags), PDF, HTML pages,
DOCX, and browser bookmark exports (Netscape HTML from any browser, Chrome/Edge JSON).
A single bookmark file produces one document per bookmark.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

TEXT_EXT = {".md", ".markdown", ".mdx", ".txt", ".text", ".org", ".rst"}
HTML_EXT = {".html", ".htm", ".xhtml"}
SUPPORTED_EXT = TEXT_EXT | HTML_EXT | {".pdf", ".docx", ".json"}
IGNORED_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
                ".obsidian", ".trash", ".Trash", ".cache", ".idea", ".vscode", "dist", "build"}


@dataclass
class Section:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass
class LoadedDoc:
    path: str                 # file path, or bookmark://<url> for bookmarks
    title: str
    kind: str                 # note | pdf | web | doc | bookmark
    sections: list[Section]
    source: str               # the file this came from
    url: str = ""
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # raw link targets, resolved by the indexer

    @property
    def content(self) -> str:
        parts = []
        for s in self.sections:
            if s.page is not None:
                parts.append(f"<!-- page {s.page} -->\n{s.text}")
            else:
                parts.append(s.text)
        return "\n\n".join(parts)

    @property
    def hash(self) -> str:
        h = hashlib.sha1()
        for part in (self.title, self.url, ",".join(self.tags), ",".join(self.links), self.content):
            h.update(part.encode("utf-8", "ignore"))
            h.update(b"\0")
        return h.hexdigest()


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXT and not path.name.startswith(".")


def iter_files(root: Path):
    """Yield supported files under root (or root itself), skipping hidden and tool dirs."""
    root = root.expanduser().resolve()
    if root.is_file():
        if is_supported(root):
            yield root
        return
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).parts
        if any(part in IGNORED_DIRS or part.startswith(".") for part in rel[:-1]):
            continue
        if p.is_file() and is_supported(p):
            yield p


def load_file(path: Path) -> list[LoadedDoc]:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return [load_markdown(path)]
    if ext == ".pdf":
        return [load_pdf(path)]
    if ext == ".docx":
        return [load_docx(path)]
    if ext in HTML_EXT:
        raw = read_text(path)
        if "NETSCAPE-Bookmark-file" in raw[:1000]:
            return load_netscape_bookmarks(path, raw)
        return [load_html(path, raw)]
    if ext == ".json":
        try:
            data = json.loads(read_text(path))
        except ValueError:
            return []
        if isinstance(data, dict) and isinstance(data.get("roots"), dict):
            return load_chrome_bookmarks(path, data)
        return []  # arbitrary JSON isn't knowledge
    return []


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------- markdown / text

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
WIKILINK_RE = re.compile(r"!?\[\[([^\]\|#\n]+)(?:#[^\]\|\n]*)?(?:\|[^\]\n]*)?\]\]")
MDLINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
TAG_RE = re.compile(r"(?:(?<=\s)|^)#([A-Za-z][\w/\-]*)", re.M)
CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML frontmatter: `key: value`, `key: [a, b]` and `- item` lists."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^\s*-\s+", line) and key:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(line.split("-", 1)[1].strip().strip("'\""))
            continue
        km = re.match(r"^([\w\-]+)\s*:\s*(.*)$", line)
        if not km:
            continue
        key, val = km.group(1).lower(), km.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        elif val:
            meta[key] = val.strip("'\"")
        else:
            meta[key] = []
    return meta, text[m.end():]


def load_markdown(path: Path) -> LoadedDoc:
    raw = read_text(path)
    meta, body = parse_frontmatter(raw)
    is_md = path.suffix.lower() in {".md", ".markdown", ".mdx"}

    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    if not title and is_md:
        h1 = re.search(r"^#\s+(.+?)\s*$", body, re.M)
        title = h1.group(1).strip() if h1 else None
    title = title or path.stem.replace("_", " ").replace("-", " ").strip() or path.name

    tags = meta.get("tags") or meta.get("tag") or []
    if isinstance(tags, str):
        tags = [t for t in re.split(r"[,\s]+", tags) if t]
    stripped = CODE_RE.sub(" ", body)
    tags = list(tags) + TAG_RE.findall(stripped) if is_md else list(tags)
    tags = sorted({t.lstrip("#").lower() for t in tags if t})

    links: list[str] = []
    if is_md:
        links += [w.strip() for w in WIKILINK_RE.findall(stripped)]
        for target in MDLINK_RE.findall(stripped):
            if not re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) and not target.startswith("#"):
                links.append(target.split("#")[0])
    links = list(dict.fromkeys(l for l in links if l))

    sections = split_markdown_sections(body) if is_md else [Section(body.strip())]
    for sec in sections:  # "Title > Risks" -> "Risks": the title is shown separately
        if sec.heading == title:
            sec.heading = ""
        elif sec.heading.startswith(title + " > "):
            sec.heading = sec.heading[len(title) + 3:]
    return LoadedDoc(str(path), title, "note", [s for s in sections if s.text.strip()],
                     str(path), tags=tags, links=links)


def split_markdown_sections(body: str) -> list[Section]:
    """Split on headings, tracking the heading path (e.g. "Project > Risks")."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    last = 0
    current = ""
    for m in HEADING_RE.finditer(body):
        text = body[last:m.start()].strip()
        if text:
            sections.append(Section(text, current))
        level, name = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        current = " > ".join(n for _, n in stack)
        last = m.end()
    tail = body[last:].strip()
    if tail:
        sections.append(Section(tail, current))
    return sections


# ---------------------------------------------------------------- pdf / docx / html


def load_pdf(path: Path) -> LoadedDoc:
    from pypdf import PdfReader

    sections: list[Section] = []
    title = ""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        info = reader.metadata
        if info and info.title and len(str(info.title).strip()) > 2:
            title = str(info.title).strip()
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = clean_pdf_text(text)
            if text:
                sections.append(Section(text, page=i))
    except Exception as e:  # corrupt or unsupported PDF: index the name so it is still findable
        sections = [Section(f"(Could not extract text: {e})", page=1)]
    return LoadedDoc(str(path), title or path.stem.replace("_", " "), "pdf", sections, str(path))


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)           # re-join hyphenated words
    text = re.sub(r"(?<![.\n:;!?])\n(?=[a-z])", " ", text)  # join soft-wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load_docx(path: Path) -> LoadedDoc:
    sections: list[Section] = []
    title = path.stem
    try:
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
            try:
                core = ElementTree.fromstring(z.read("docProps/core.xml"))
                t = core.find("{http://purl.org/dc/elements/1.1/}title")
                if t is not None and t.text and t.text.strip():
                    title = t.text.strip()
            except KeyError:
                pass
        heading, buf = "", []
        for p in root.iter(f"{W_NS}p"):
            text = "".join(t.text or "" for t in p.iter(f"{W_NS}t")).strip()
            if not text:
                continue
            style = p.find(f"{W_NS}pPr/{W_NS}pStyle")
            style_name = style.get(f"{W_NS}val", "") if style is not None else ""
            if style_name.lower().startswith(("heading", "title")):
                if buf:
                    sections.append(Section("\n\n".join(buf), heading))
                heading, buf = text, []
            else:
                buf.append(text)
        if buf:
            sections.append(Section("\n\n".join(buf), heading))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        sections = [Section(f"(Could not read document: {e})")]
    return LoadedDoc(str(path), title, "doc", sections, str(path))


def html_to_text(raw: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text


def load_html(path: Path, raw: str) -> LoadedDoc:
    title, text = html_to_text(raw)
    return LoadedDoc(str(path), title or path.stem, "web", [Section(text)] if text else [], str(path))


# ---------------------------------------------------------------- bookmarks

BM_TOKEN_RE = re.compile(
    r"<H3[^>]*>(?P<folder>.*?)</H3>|(?P<open><DL>)|(?P<close></DL>)"
    r"|<A\s(?P<attrs>[^>]*)>(?P<title>.*?)</A>(?:\s*<DD>(?P<desc>[^<]*))?",
    re.I | re.S,
)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def bookmark_doc(source: Path, url: str, title: str, folders: list[str],
                 description: str = "", extra_tags: list[str] | None = None) -> LoadedDoc:
    folder = " / ".join(folders)
    tags = sorted({slug(f) for f in folders if slug(f)} | {slug(t) for t in (extra_tags or []) if slug(t)})
    lines = [title, url]
    if folder:
        lines.append(f"Folder: {folder}")
    if description:
        lines.append(description)
    return LoadedDoc(f"bookmark://{url}", title or url, "bookmark", [Section("\n".join(lines))],
                     str(source), url=url, tags=tags)


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.strip().lower()).strip("-")


def load_netscape_bookmarks(path: Path, raw: str) -> list[LoadedDoc]:
    docs: dict[str, LoadedDoc] = {}
    stack: list[str] = []
    pending: str | None = None
    for m in BM_TOKEN_RE.finditer(raw):
        if m.group("folder") is not None:
            pending = html.unescape(m.group("folder")).strip()
        elif m.group("open"):
            stack.append(pending or "")
            pending = None
        elif m.group("close"):
            if stack:
                stack.pop()
        elif m.group("attrs") is not None:
            attrs = {k.upper(): html.unescape(v) for k, v in ATTR_RE.findall(m.group("attrs"))}
            url = attrs.get("HREF", "")
            if not url.startswith(("http://", "https://")):
                continue
            title = html.unescape(re.sub(r"<[^>]+>", "", m.group("title"))).strip()
            desc = html.unescape(m.group("desc") or "").strip()
            folders = [f for f in stack if f and f.lower() not in ("bookmarks", "bookmarks bar", "bookmarks toolbar", "bookmarks menu")]
            tags = [t for t in attrs.get("TAGS", "").split(",") if t]
            docs[url] = bookmark_doc(path, url, title, folders, desc, tags)
    return list(docs.values())


def load_chrome_bookmarks(path: Path, data: dict) -> list[LoadedDoc]:
    docs: dict[str, LoadedDoc] = {}

    def walk(node: dict, folders: list[str]):
        if node.get("type") == "url":
            url = node.get("url", "")
            if url.startswith(("http://", "https://")):
                docs[url] = bookmark_doc(path, url, node.get("name", ""), folders)
        for child in node.get("children", []) or []:
            name = node.get("name", "") if node.get("type") == "folder" else ""
            walk(child, folders + [name] if name else folders)

    for key, root in data["roots"].items():
        if isinstance(root, dict):
            walk({"type": "root", "children": root.get("children", [])}, [])
    return list(docs.values())


def fetch_page_text(url: str, timeout: float = 10.0, max_bytes: int = 2_000_000) -> str:
    """Download a bookmarked page and extract readable text. Only used when fetch_bookmarks is on."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "SecondBrainOS/0.1 (+local indexer)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if "html" not in resp.headers.get("Content-Type", "html"):
            return ""
        raw = resp.read(max_bytes).decode(resp.headers.get_content_charset() or "utf-8", "ignore")
    return html_to_text(raw)[1][:50_000]
