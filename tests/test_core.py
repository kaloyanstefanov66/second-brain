import json
import shutil
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from secondbrain.brain import Brain
from secondbrain.chunking import chunk_sections
from secondbrain.config import Settings
from secondbrain.embeddings import HashingEmbedder
from secondbrain.loaders import Section, load_file
from secondbrain import llm, qa

SAMPLE = Path(__file__).resolve().parent.parent / "sample_vault"


@pytest.fixture()
def vault(tmp_path):
    dst = tmp_path / "vault"
    shutil.copytree(SAMPLE, dst)
    return dst


@pytest.fixture()
def brain(tmp_path, vault, monkeypatch):
    llm.clear_cache()
    s = Settings.load(tmp_path / "home", embedder="hashing", llm="none")
    b = Brain(s)
    b.add([vault])
    return b


# ------------------------------------------------------------------ loaders

def test_markdown_frontmatter_tags_and_links(vault):
    (doc,) = load_file(vault / "learning" / "Zettelkasten Method.md")
    assert doc.title == "Zettelkasten Method"
    assert {"learning", "writing", "pkm"} <= set(doc.tags)
    assert doc.links == ["Spaced Repetition"]
    assert all(not s.heading.startswith("Zettelkasten") for s in doc.sections)


def test_netscape_bookmarks(vault):
    docs = load_file(vault / "bookmarks.html")
    by_url = {d.url: d for d in docs}
    assert len(docs) == 4
    raft = by_url["https://raft.github.io/"]
    assert raft.kind == "bookmark" and raft.tags == ["engineering"]
    assert "leader election" in raft.content


def test_chrome_bookmarks(tmp_path):
    data = {"roots": {"bookmark_bar": {"type": "folder", "name": "Bar", "children": [
        {"type": "folder", "name": "Reading", "children": [{"type": "url", "name": "Essay", "url": "https://example.com/e"}]}]}}}
    p = tmp_path / "Bookmarks.json"
    p.write_text(json.dumps(data))
    (doc,) = load_file(p)
    assert doc.title == "Essay" and doc.tags == ["reading"]


def test_docx(tmp_path):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = (f'<w:document {ns}><w:body>'
           '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Plan</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>Ship the beta in May.</w:t></w:r></w:p></w:body></w:document>')
    p = tmp_path / "plan.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", xml)
    (doc,) = load_file(p)
    assert doc.kind == "doc" and doc.sections[0].heading == "Plan"
    assert "beta" in doc.sections[0].text


def test_pdf_pages(vault):
    (doc,) = load_file(vault / "papers" / "ant-colony-optimization.pdf")
    assert doc.kind == "pdf" and doc.title.startswith("Ant Colony")
    assert [s.page for s in doc.sections] == [1, 2]


# ------------------------------------------------------------------ chunking / embeddings

def test_chunks_are_bounded_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(300))
    chunks = chunk_sections([Section(text, "H")], size=500, overlap=120)
    assert len(chunks) > 5
    assert all(len(c.text) <= 500 + 120 for c in chunks)
    assert chunks[1].text.split(".")[0] in chunks[0].text  # carried-over context


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    v = e.embed(["leader election in raft", "raft elects a leader", "sourdough starter feeding"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


# ------------------------------------------------------------------ indexing / search / graph

def test_index_and_hybrid_search(brain):
    s = brain.store.stats()
    assert s["documents"] == 17 and s["by_kind"] == {"bookmark": 4, "note": 12, "pdf": 1}
    hits = brain.search("how do nodes elect a leader", k=5)
    assert hits[0].title in ("Consensus Algorithms", "Raft Consensus Algorithm")
    pdf_hits = brain.search("pheromone evaporation", k=3, kinds=["pdf"])
    assert pdf_hits and pdf_hits[0].page == 1


def test_incremental_reindex(brain, vault):
    r = brain.reindex()
    assert (r.added, r.updated, r.removed) == (0, 0, 0) and r.unchanged == 17
    note = vault / "hobbies" / "Sourdough Starter.md"
    note.write_text(note.read_text() + "\nUpdate: switched to spelt flour.\n")
    (vault / "journal" / "2026-10-02.md").unlink()
    r = brain.reindex()
    assert (r.updated, r.removed) == (1, 1)
    assert brain.search("spelt flour", k=1)[0].title == "Sourdough Starter"


def test_graph_links_and_discoveries(brain):
    titles = {d["id"]: d["title"] for d in brain.store.all_docs()}
    links = {(titles[e["src"]], titles[e["dst"]]) for e in brain.store.edges(("link",))}
    assert ("Spaced Repetition", "Zettelkasten Method") in links
    assert ("Meeting — Q3 Planning", "Project Atlas — Kickoff") in links

    conns = brain.connections()
    assert conns, "expected discovered connections in the sample vault"
    tags = {d["id"]: set(d["tags"]) for d in brain.store.all_docs()}
    for c in conns:
        assert not (tags[c["a"]["id"]] & tags[c["b"]["id"]]), "discoveries never share tags"
        assert (c["a"]["title"], c["b"]["title"]) not in links and (c["b"]["title"], c["a"]["title"]) not in links
        assert len(c["evidence"]) == 2
    pairs = {frozenset((c["a"]["title"], c["b"]["title"])) for c in conns}
    assert frozenset(("Beekeeping Log", "Ant Colony Optimization: A Short Survey")) in pairs

    g = brain.graph()
    assert len(g["nodes"]) == 17 and g["clusters"]


def test_embedder_change_triggers_reembed(brain):
    class Other(HashingEmbedder):
        def __init__(self):
            super().__init__(512)
            self.id = "test-other-512"

    brain.indexer.embedder = brain.searcher.embedder = brain.embedder = Other()
    assert brain.ensure_ready() is True
    assert brain.store.get_meta("embedder")["id"] == "test-other-512"
    assert brain.search("quorum sensing bees", k=1)[0].title == "Beekeeping Log"


# ------------------------------------------------------------------ question answering

def test_extractive_answer_cites_sources(brain):
    text, sources, meta = brain.ask("How do honeybee swarms choose a new nest site?")
    assert meta["mode"] == "extractive"
    cited = qa.cited_numbers(text)
    assert cited and cited <= {s["n"] for s in sources}
    assert any(s["title"] == "Beekeeping Log" for s in sources)


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3.2:latest"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":  # deterministic stand-in vectors
            vecs = HashingEmbedder(64).embed([t.split(": ", 1)[-1] for t in req["input"]]).tolist()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"embeddings": vecs}).encode())
            return
        assert "Excerpts" in req["messages"][-1]["content"]
        self.send_response(200)
        self.end_headers()
        for piece in ["Scouts dance ", "and the swarm uses a quorum [1]."]:
            self.wfile.write(json.dumps({"message": {"content": piece}, "done": False}).encode() + b"\n")
        self.wfile.write(json.dumps({"done": True}).encode() + b"\n")


def test_llm_answer_streams_from_local_ollama(brain, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        brain.settings.llm = "ollama"
        brain.settings.ollama_url = f"http://127.0.0.1:{server.server_port}"
        llm.clear_cache()
        text, sources, meta = brain.ask("How do bees pick a home?")
        assert meta == {"type": "meta", "mode": "llm", "model": "llama3.2:latest (Ollama)", "cloud": False}
        assert text == "Scouts dance and the swarm uses a quorum [1]."
    finally:
        server.shutdown()


# ------------------------------------------------------------------ HTTP API

def test_api(brain, tmp_path):
    from fastapi.testclient import TestClient

    from secondbrain.server import create_app

    c = TestClient(create_app(brain), base_url="http://localhost")
    assert c.get("/api/status").json()["documents"] == 17
    assert c.get("/").status_code == 200
    res = c.get("/api/search", params={"q": "waggle dance"}).json()["results"]
    assert res[0]["title"] == "Beekeeping Log"

    events = [json.loads(l) for l in c.post("/api/ask", json={"question": "what is stigmergy?"}).text.splitlines()]
    assert [e["type"] for e in events][0] == "sources" and events[-1]["type"] == "done"

    doc_id = res[0]["doc_id"]
    doc = c.get(f"/api/documents/{doc_id}").json()
    assert doc["has_file"] and doc["related"]
    assert c.get(f"/api/documents/{doc_id}/file").status_code == 200
    assert c.get("/api/resolve", params={"title": "spaced repetition"}).status_code == 200

    # local-only guard
    assert c.get("/api/status", headers={"host": "attacker.example"}).status_code == 403
    assert c.post("/api/reindex", headers={"origin": "https://attacker.example"}).status_code == 403

    r = c.post("/api/upload", files=[("files", ("idea.md", b"# Idea\nGardens are slow databases.", "text/markdown"))])
    assert r.status_code == 200
    brain._job.join(10)
    assert any(d["title"] == "Idea" for d in brain.store.all_docs())


def test_ollama_embedder(tmp_path, vault):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        s = Settings.load(tmp_path / "h2", embedder="ollama", llm="none",
                          ollama_url=f"http://127.0.0.1:{server.server_port}")
        b = Brain(s)
        assert b.embedder.id == "ollama:nomic-embed-text" and b.embedder.dim == 64
        b.add([vault / "work"])
        assert "Consensus Algorithms" in [h.title for h in b.search("raft leader election", k=3)]
    finally:
        server.shutdown()
