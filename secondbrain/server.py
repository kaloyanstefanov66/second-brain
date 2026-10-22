"""Local web app + JSON API. Binds to 127.0.0.1 by default: nothing is exposed to your network."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .brain import Brain
from .discover import discover, outermost
from .graph import overview
from .llm import PROVIDERS, ProviderError, ProviderStore, detect_llm, list_models, pick_model, provider_status

STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str
    history: list[dict] = []
    k: int = 8


class PathBody(BaseModel):
    path: str


class ProviderBody(BaseModel):
    api_key: str | None = None
    model: str | None = None


class ActiveBody(BaseModel):
    provider: str


class BulkBody(BaseModel):
    paths: list[str]


def create_app(brain: Brain) -> FastAPI:
    app = FastAPI(title="Second Brain", docs_url="/api/docs", openapi_url="/api/openapi.json")
    allowed_hosts = {"localhost", "127.0.0.1", "[::1]", brain.settings.host}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Any web page you visit can send requests to localhost. Reject foreign Host headers
        # (DNS rebinding) and cross-site Origins so only this app can read or change your index.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        origin = request.headers.get("origin")
        if host not in allowed_hosts:
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        if origin and urlparse(origin).hostname not in {h.strip("[]") for h in allowed_hosts}:
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    @app.get("/api/status")
    def status():
        return brain.status()

    @app.post("/api/llm/refresh")
    def refresh_llm():
        llm = detect_llm(brain.settings, refresh=True)
        return {"llm": llm.name if llm else None}

    # ---- answer engines & API keys (keys go in, only masked keys come out) ----

    def _spec(pid: str):
        if pid not in PROVIDERS:
            raise HTTPException(404, f"unknown provider {pid!r}")
        return PROVIDERS[pid]

    @app.get("/api/providers")
    def providers():
        return provider_status(brain.settings)

    @app.get("/api/providers/{pid}/models")
    def provider_models(pid: str):
        _spec(pid)
        try:
            return {"models": list_models(pid, brain.settings, ProviderStore(brain.settings.home))}
        except (ProviderError, ImportError) as e:
            raise HTTPException(400, str(e))

    @app.put("/api/providers/{pid}")
    def save_provider(pid: str, body: ProviderBody):
        spec = _spec(pid)
        store = ProviderStore(brain.settings.home)
        key = (body.api_key or "").strip() or None
        if key and not spec.cloud:
            raise HTTPException(400, f"{spec.name} runs locally and needs no key")
        model = body.model
        if key:  # validate before saving: listing models fails fast on a bad key
            try:
                available = list_models(pid, brain.settings, store, api_key=key)
            except (ProviderError, ImportError) as e:
                raise HTTPException(400, f"{spec.name}: {e}. The key was not saved.")
            model = pick_model(spec, available, model or store.load()["providers"].get(pid, {}).get("model"))
            store.update(pid, api_key=key, model=model)
            if store.load()["active"] in (None, "auto", "none"):
                store.set_active(pid)  # adding a key means you want to use it
        elif model:
            store.update(pid, model=model)
        return provider_status(brain.settings)

    @app.delete("/api/providers/{pid}/key")
    def delete_key(pid: str):
        _spec(pid)
        store = ProviderStore(brain.settings.home)
        store.update(pid, api_key=None)
        if store.load()["active"] == pid and not store.key(pid)[0]:
            store.set_active("auto")
        return provider_status(brain.settings)

    @app.post("/api/providers/active")
    def set_active(body: ActiveBody):
        if body.provider not in ("auto", "none"):
            spec = _spec(body.provider)
            if spec.cloud and not ProviderStore(brain.settings.home).key(spec.id)[0]:
                raise HTTPException(400, f"add a {spec.name} API key first")
        ProviderStore(brain.settings.home).set_active(body.provider)
        llm = detect_llm(brain.settings, refresh=True)
        return {**provider_status(brain.settings), "llm": llm.name if llm else None}

    @app.get("/api/search")
    def search(q: str, k: int = 20, kind: str | None = None):
        kinds = [x for x in kind.split(",") if x] if kind else None
        return {"query": q, "results": [h.to_dict() for h in brain.search(q, k=k, kinds=kinds)]}

    @app.post("/api/ask")
    def ask(body: AskBody):
        def stream():
            try:
                for ev in brain.ask_stream(body.question, k=body.k, history=body.history):
                    yield json.dumps(ev) + "\n"
            except Exception as e:  # surface errors to the UI instead of a broken stream
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/index/pause")
    def pause_indexing():
        brain.pause()
        return {"paused": True}

    @app.post("/api/index/resume")
    def resume_indexing():
        brain.resume()
        return {"paused": False}

    @app.post("/api/quit")
    def quit_app():
        if brain.on_quit is None:
            raise HTTPException(404, "not running as the desktop app")
        threading.Timer(0.3, brain.on_quit).start()  # let this response reach the page first
        return {"ok": True}

    @app.get("/api/overview")
    def overview_():
        return overview(brain.store)

    @app.get("/api/discover")
    def discover_():
        """Look for notes on this machine. Reads folder structure and file names only."""
        return {"candidates": [c.to_dict() for c in discover(added_roots=brain.store.roots())]}

    @app.post("/api/sources/bulk")
    def add_sources(body: BulkBody):
        paths = [p for p in outermost(body.paths) if Path(p).exists()]
        if not paths:
            raise HTTPException(400, "nothing to add")
        if not brain.run_in_background(brain.add, paths):
            raise HTTPException(409, "an indexing job is already running")
        return {"ok": True, "paths": paths}

    @app.get("/api/graph")
    def graph():
        return brain.graph()

    @app.get("/api/connections")
    def connections(limit: int = 60):
        return {"connections": brain.connections(limit)}

    @app.get("/api/documents")
    def documents():
        return {"documents": brain.store.all_docs()}

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: int):
        doc = brain.store.doc(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        backlinks, outlinks, related = [], [], []
        for e in brain.store.edges_for(doc_id):
            outgoing = e["src"] == doc_id
            other = {"id": e["dst"] if outgoing else e["src"],
                     "title": e["dst_title"] if outgoing else e["src_title"],
                     "kind": e["dst_kind"] if outgoing else e["src_kind"],
                     "weight": e["weight"], **{k: e["data"].get(k) for k in ("terms", "discovered", "shared_tags")}}
            if e["kind"] == "link":
                (outlinks if outgoing else backlinks).append(other)
            else:
                related.append(other)
        doc["chunks"] = brain.store.chunks_for(doc_id)
        doc["has_file"] = doc["kind"] != "bookmark" and Path(doc["source"]).exists()
        return {**doc, "backlinks": backlinks, "outlinks": outlinks, "related": related}

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: int):
        doc = brain.store.doc(doc_id)
        if not doc or doc["kind"] == "bookmark":
            raise HTTPException(404, "no file for this document")
        path = Path(doc["source"])
        if not path.is_file():
            raise HTTPException(404, "file has moved or been deleted")
        return FileResponse(path, filename=path.name, content_disposition_type="inline")

    @app.get("/api/resolve")
    def resolve(title: str):
        """Find a document by title or file name (for clicking [[wikilinks]] in the reader)."""
        t = title.strip().lower()
        for d in brain.store.all_docs():
            if d["title"].lower() == t or Path(d["path"]).stem.lower() == t:
                return {"id": d["id"]}
        raise HTTPException(404, "not found")

    @app.post("/api/sources")
    def add_source(body: PathBody):
        p = Path(body.path).expanduser()
        if not p.exists():
            raise HTTPException(400, f"{p} does not exist")
        if not brain.run_in_background(brain.add, [str(p)]):
            raise HTTPException(409, "an indexing job is already running")
        return {"ok": True}

    @app.post("/api/sources/remove")
    def remove_source(body: PathBody):
        return {"removed": brain.remove(body.path)}

    @app.post("/api/reindex")
    def reindex():
        if not brain.run_in_background(brain.reindex):
            raise HTTPException(409, "an indexing job is already running")
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)):
        saved = []
        for f in files:
            name = re.sub(r"[^\w.\- ]+", "_", Path(f.filename or "upload").name).strip() or "upload"
            dest = brain.settings.inbox / name
            stem, suffix, i = dest.stem, dest.suffix, 1
            while dest.exists():
                dest = brain.settings.inbox / f"{stem} ({i}){suffix}"
                i += 1
            dest.write_bytes(await f.read())
            saved.append(str(dest))
        if not brain.run_in_background(brain.add, [str(brain.settings.inbox)]):
            raise HTTPException(409, "saved, but an indexing job is already running; re-index to include them")
        return {"saved": saved}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
