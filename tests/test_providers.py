"""Cloud answer engines and API key handling, tested against fake provider servers."""

import dataclasses
import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from secondbrain import llm
from secondbrain.brain import Brain
from secondbrain.config import Settings
from secondbrain.server import create_app

SAMPLE = Path(__file__).resolve().parent.parent / "sample_vault"
GOOD_KEY = "sk-test-good-key-1234567890"


class FakeCloud(BaseHTTPRequestHandler):
    """OpenAI-compatible and Anthropic-shaped endpoints. Records what it received."""

    seen: list = []

    def log_message(self, *a):
        pass

    def _auth_ok(self):
        return self.headers.get("Authorization") == f"Bearer {GOOD_KEY}" or self.headers.get("x-api-key") == GOOD_KEY

    def _json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        if not self._auth_ok():
            return self._json(401, {"error": {"message": "Incorrect API key provided"}})
        if self.path.startswith("/v1/models") and self.headers.get("anthropic-version"):
            return self._json(200, {"data": [{"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5",
                                              "created_at": "2026-09-01T00:00:00Z"}],
                                    "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-opus-5-5"})
        return self._json(200, {"data": [{"id": "gpt-5-mini"}, {"id": "text-embedding-3-small"}, {"id": "gpt-5"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeCloud.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if not self._auth_ok():
            return self._json(401, {"error": {"message": "Incorrect API key provided"}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.path.startswith("/v1/messages"):  # Anthropic Messages API stream
            events = [
                ("message_start", {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                    "model": body["model"], "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1}}}),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bees vote "}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "by quorum [1]."}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                   "usage": {"output_tokens": 8}}),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, data in events:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            return
        for piece in ["Scouts dance", " and a quorum decides [1]."]:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def fake():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCloud)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeCloud.seen = []
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def client(tmp_path, fake, monkeypatch):
    for spec in llm.PROVIDERS.values():
        for name in spec.env:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(llm.PROVIDERS, "openai", dataclasses.replace(llm.PROVIDERS["openai"], base_url=f"{fake}/v1"))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake)
    llm.clear_cache()
    s = Settings.load(tmp_path / "home", embedder="hashing", ollama_url="http://127.0.0.1:9", local_url="http://127.0.0.1:9/v1")
    brain = Brain(s)
    brain.add([SAMPLE / "hobbies"])
    yield TestClient(create_app(brain), base_url="http://localhost"), brain
    llm.clear_cache()


def test_bad_key_is_rejected_and_not_saved(client):
    c, brain = client
    r = c.put("/api/providers/openai", json={"api_key": "sk-wrong-000000000000"})
    assert r.status_code == 400 and "key rejected" in r.json()["detail"] and "not saved" in r.json()["detail"]
    assert not (brain.settings.home / "providers.json").exists()


def test_key_saved_masked_private_and_used_for_answers(client):
    c, brain = client
    st = c.put("/api/providers/openai", json={"api_key": GOOD_KEY}).json()
    openai = next(p for p in st["providers"] if p["id"] == "openai")
    assert openai["configured"] and openai["masked_key"] == "sk-t…7890" and openai["model"] == "gpt-5-mini"
    assert st["active"] == "openai", "adding the first key makes it the active engine"

    path = brain.settings.home / "providers.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    for endpoint in ("/api/providers", "/api/status"):
        assert GOOD_KEY not in c.get(endpoint).text, "raw keys must never leave the server"

    assert c.get("/api/providers/openai/models").json()["models"] == ["gpt-5", "gpt-5-mini"]  # embeddings filtered

    events = [json.loads(l) for l in c.post("/api/ask", json={"question": "how do bees pick a nest?"}).text.splitlines()]
    meta = next(e for e in events if e["type"] == "meta")
    assert meta == {"type": "meta", "mode": "llm", "model": "gpt-5-mini (OpenAI)", "cloud": True}
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Scouts dance and a quorum decides [1]."
    req = FakeCloud.seen[-1]
    assert req["headers"]["Authorization"] == f"Bearer {GOOD_KEY}" and "temperature" not in req["body"]
    assert "Excerpts" in req["body"]["messages"][-1]["content"]


def test_auto_never_uses_cloud(client):
    c, brain = client
    c.put("/api/providers/openai", json={"api_key": GOOD_KEY})
    st = c.post("/api/providers/active", json={"provider": "auto"}).json()
    assert st["active"] == "auto" and st["llm"] is None  # no local engine running here
    events = [json.loads(l) for l in c.post("/api/ask", json={"question": "sourdough"}).text.splitlines()]
    assert next(e for e in events if e["type"] == "meta")["mode"] == "extractive"
    assert not any(r["path"].endswith("/chat/completions") for r in FakeCloud.seen)


def test_cannot_select_cloud_without_key_and_remove_key(client):
    c, _ = client
    assert c.post("/api/providers/active", json={"provider": "gemini"}).status_code == 400
    c.put("/api/providers/openai", json={"api_key": GOOD_KEY})
    st = c.delete("/api/providers/openai/key").json()
    assert st["active"] == "auto" and not next(p for p in st["providers"] if p["id"] == "openai")["configured"]


def test_env_key_is_detected(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-environment-xyz")
    g = next(p for p in c.get("/api/providers").json()["providers"] if p["id"] == "gemini")
    assert g["configured"] and g["key_source"] == "env:GEMINI_API_KEY" and g["masked_key"] == "AIza…-xyz"


def test_claude_via_anthropic_sdk(client):
    c, _ = client
    st = c.put("/api/providers/anthropic", json={"api_key": GOOD_KEY}).json()
    assert st["active"] == "anthropic"
    events = [json.loads(l) for l in c.post("/api/ask", json={"question": "how do bees decide?"}).text.splitlines()]
    assert next(e for e in events if e["type"] == "meta")["model"] == "claude-opus-5-5 (Anthropic)"
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Bees vote by quorum [1]."
    req = next(r for r in FakeCloud.seen if r["path"].startswith("/v1/messages"))
    assert req["headers"]["Authorization"] == f"Bearer {GOOD_KEY}" and req["headers"]["anthropic-version"]
    assert req["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["headers"]["anthropic-beta"]
    assert req["body"]["system"].startswith("You are the user's second brain")
    assert "temperature" not in req["body"] and "thinking" not in req["body"]
