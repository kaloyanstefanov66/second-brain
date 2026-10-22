"""Answer engines: local models by default, cloud providers only when you add a key and pick one.

Local:  Ollama, or any OpenAI-compatible server on localhost (LM Studio, llama.cpp, vLLM).
Cloud:  Claude (Anthropic SDK), OpenAI, Google Gemini, GitHub Models (the Copilot model catalog).

Keys live in ~/.secondbrain/providers.json (file mode 0600) or come from the usual environment
variables. Using a cloud provider sends your question and the retrieved excerpts (never your
whole library) to that provider. Embeddings always stay local.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    cloud: bool
    default_model: str = ""
    env: tuple[str, ...] = ()
    base_url: str = ""
    key_url: str = ""
    key_hint: str = ""
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {p.id: p for p in [
    ProviderSpec("ollama", "Ollama", cloud=False, notes="Runs models on this machine. Install it from ollama.com, then run: ollama pull llama3.2"),
    ProviderSpec("local", "Local server", cloud=False,
                 notes="Any OpenAI-compatible server on this machine: LM Studio, llama.cpp, vLLM."),
    ProviderSpec("anthropic", "Claude", cloud=True, default_model="claude-opus-5-5",
                 env=("ANTHROPIC_API_KEY",), key_url="https://platform.claude.com/settings/keys",
                 key_hint="sk-ant-…"),
    ProviderSpec("openai", "OpenAI", cloud=True, default_model="gpt-5-mini", env=("OPENAI_API_KEY",),
                 base_url="https://api.openai.com/v1", key_url="https://platform.openai.com/api-keys", key_hint="sk-…"),
    ProviderSpec("gemini", "Google Gemini", cloud=True, default_model="gemini-2.5-flash",
                 env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                 key_url="https://aistudio.google.com/apikey", key_hint="AIza…"),
    ProviderSpec("github", "GitHub Models (Copilot)", cloud=True, default_model="openai/gpt-4.1-mini",
                 env=("GITHUB_TOKEN", "GH_TOKEN"), base_url="https://models.github.ai/inference",
                 key_url="https://github.com/settings/personal-access-tokens/new", key_hint="github_pat_…",
                 notes="Uses a GitHub token with the Models permission. Covers the models in GitHub's catalog, "
                       "billed against your Copilot / GitHub Models allowance."),
]}

# Models with Anthropic's server-side refusal fallback ("default" routing by refusal category).
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class ProviderError(RuntimeError):
    """A provider call failed. The message is safe to show: it never contains the key."""


# ------------------------------------------------------------------ key store

class ProviderStore:
    """providers.json: {"active": "auto", "providers": {"openai": {"api_key": "...", "model": "..."}}}"""

    def __init__(self, home: Path):
        self.path = Path(home) / "providers.json"

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        data.setdefault("active", None)
        data.setdefault("providers", {})
        return data

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".providers-")
        try:
            os.fchmod(fd, 0o600)  # keys are readable by you only
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        clear_cache()

    def key(self, pid: str) -> tuple[str | None, str | None]:
        """(key, source) where source is "saved" or "env:NAME"."""
        saved = self.load()["providers"].get(pid, {}).get("api_key")
        if saved:
            return saved, "saved"
        for name in PROVIDERS[pid].env:
            if os.environ.get(name):
                return os.environ[name], f"env:{name}"
        return None, None

    def model(self, pid: str) -> str:
        return self.load()["providers"].get(pid, {}).get("model") or PROVIDERS[pid].default_model

    def update(self, pid: str, **fields) -> None:
        data = self.load()
        entry = data["providers"].setdefault(pid, {})
        for k, v in fields.items():
            if v is None:
                entry.pop(k, None)
            else:
                entry[k] = v
        if not entry:
            data["providers"].pop(pid, None)
        self.save(data)

    def set_active(self, pid: str | None) -> None:
        data = self.load()
        data["active"] = pid
        self.save(data)


def mask(key: str | None) -> str | None:
    if not key:
        return None
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "••••"


# ------------------------------------------------------------------ HTTP helpers

def _request(url: str, *, body: dict | None = None, headers: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "User-Agent": "SecondBrain/0.1", **(headers or {})}
    req = urllib.request.Request(url, data, hdrs)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise ProviderError(_http_error(e)) from None
    except (urllib.error.URLError, OSError) as e:
        raise ProviderError(f"could not reach {urllib.parse.urlparse(url).netloc}: {getattr(e, 'reason', e)}") from None


def _http_error(e: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(e.read() or b"{}")
        if isinstance(payload, list) and payload:
            payload = payload[0]
        err = payload.get("error", payload) if isinstance(payload, dict) else {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = None
    if e.code in (401, 403):
        return f"key rejected ({e.code}{': ' + msg if msg else ''})"
    return f"HTTP {e.code}: {msg or e.reason}"


# ------------------------------------------------------------------ engines

class LLM:
    name = "llm"
    cloud = False

    def stream(self, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model
        self.name = f"{model} (Ollama)"

    def stream(self, messages):
        body = {"model": self.model, "messages": messages, "stream": True, "options": {"temperature": 0.2}}
        with _request(f"{self.url}/api/chat", body=body, timeout=300) as r:
            for line in r:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise ProviderError(msg["error"])
                piece = msg.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if msg.get("done"):
                    break


class OpenAICompatLLM(LLM):
    """Chat Completions streaming: local servers, OpenAI, Gemini's OpenAI endpoint, GitHub Models."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, label: str = "",
                 cloud: bool = False, temperature: float | None = None):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.name, self.cloud, self.temperature = f"{model} ({label})" if label else model, cloud, temperature

    def stream(self, messages):
        body = {"model": self.model, "messages": messages, "stream": True}
        if self.temperature is not None:  # newer reasoning models reject non-default temperatures
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with _request(f"{self.base_url}/chat/completions", body=body, headers=headers, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ProviderError(chunk["error"].get("message", "provider error"))
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield delta


class AnthropicLLM(LLM):
    """Claude via the official Anthropic SDK, streaming."""

    cloud = True

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=2)
        self.model = model
        self.name = f"{model} (Anthropic)"

    def stream(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] in ("user", "assistant")]
        kwargs = {}
        if self.model in CLAUDE_FALLBACK_MODELS:
            # Server-side refusal fallback: a declined request is retried on Anthropic's recommended model.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            with self.client.beta.messages.stream(model=self.model, max_tokens=64000, system=system,
                                                  messages=convo, **kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n\n_(Claude declined to answer this question.)_"
            elif final.stop_reason == "max_tokens":
                yield "\n\n_(Answer cut off at the output limit.)_"
        except self.anthropic.AuthenticationError:
            raise ProviderError("key rejected (401)") from None
        except self.anthropic.PermissionDeniedError as e:
            raise ProviderError(f"permission denied: {e.message}") from None
        except self.anthropic.NotFoundError:
            raise ProviderError(f"model {self.model!r} not found for this key") from None
        except self.anthropic.RateLimitError:
            raise ProviderError("rate limited by Anthropic, try again shortly") from None
        except self.anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from None
        except self.anthropic.APIConnectionError:
            raise ProviderError("could not reach api.anthropic.com") from None


def build_llm(pid: str, settings, store: ProviderStore) -> LLM:
    spec = PROVIDERS[pid]
    model = store.model(pid)
    if pid == "ollama":
        return OllamaLLM(settings.ollama_url, model or settings.llm_model)
    if pid == "local":
        return OpenAICompatLLM(settings.local_url, model or settings.llm_model, label="local server", temperature=0.2)
    key, _ = store.key(pid)
    if not key:
        raise ProviderError(f"no API key for {spec.name}")
    if pid == "anthropic":
        return AnthropicLLM(key, model)
    return OpenAICompatLLM(spec.base_url, model, api_key=key, label=spec.name, cloud=True)


# ------------------------------------------------------------------ model listing (also validates keys)

_NOT_CHAT = ("embed", "tts", "whisper", "dall-e", "davinci", "babbage", "moderation", "image", "audio",
             "realtime", "transcribe", "aqa", "imagen", "veo", "search", "computer-use")


def list_models(pid: str, settings, store: ProviderStore, api_key: str | None = None) -> list[str]:
    spec = PROVIDERS[pid]
    key = api_key or store.key(pid)[0]
    if pid == "ollama":
        with _request(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", []) if "embed" not in m["name"]]
    if pid == "local":
        with _request(f"{settings.local_url.rstrip('/')}/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    if not key:
        raise ProviderError(f"no API key for {spec.name}")
    if pid == "anthropic":
        import anthropic

        try:
            return [m.id for m in anthropic.Anthropic(api_key=key, max_retries=1).models.list(limit=100)]
        except anthropic.AuthenticationError:
            raise ProviderError("key rejected (401)") from None
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from None
        except anthropic.APIConnectionError:
            raise ProviderError("could not reach api.anthropic.com") from None
    headers = {"Authorization": f"Bearer {key}"}
    if pid == "github":
        with _request("https://models.github.ai/catalog/models", headers=headers, timeout=15) as r:
            items = json.loads(r.read())
        return [m["id"] for m in items if not any(x in m["id"].lower() for x in _NOT_CHAT)]
    with _request(f"{spec.base_url}/models", headers=headers, timeout=15) as r:
        ids = [m["id"].removeprefix("models/") for m in json.loads(r.read()).get("data", [])]
    return sorted(i for i in ids if not any(x in i.lower() for x in _NOT_CHAT))


def pick_model(spec: ProviderSpec, available: list[str], current: str | None) -> str:
    for candidate in (current, spec.default_model):
        if candidate and (not available or candidate in available):
            return candidate
    return available[0] if available else spec.default_model


# ------------------------------------------------------------------ choosing the engine

_cache: tuple[float, str, LLM | None] | None = None


def clear_cache() -> None:
    global _cache
    _cache = None


def active_provider(settings, store: ProviderStore) -> str:
    """What the user selected: auto (local only), none, or a provider id."""
    return store.load()["active"] or settings.llm


def detect_llm(settings, refresh: bool = False) -> LLM | None:
    """The engine that will answer. Cached for 30s so the UI can poll cheaply.

    "auto" only ever uses local engines. Cloud providers are used only when explicitly selected.
    """
    global _cache
    store = ProviderStore(settings.home)
    choice = active_provider(settings, store)
    if not refresh and _cache and _cache[1] == choice and time.time() - _cache[0] < 30:
        return _cache[2]
    llm: LLM | None = None
    candidates = ["ollama", "local"] if choice == "auto" else ([] if choice == "none" else [choice])
    for pid in candidates:
        if pid not in PROVIDERS:
            continue
        try:
            if not PROVIDERS[pid].cloud:
                available = list_models(pid, settings, store)
                if not available:
                    continue
                store_model = store.load()["providers"].get(pid, {}).get("model")
                model = store_model if store_model in available else next(
                    (m for m in available if m == settings.llm_model or m.split(":")[0] == settings.llm_model), available[0])
                llm = OllamaLLM(settings.ollama_url, model) if pid == "ollama" else OpenAICompatLLM(
                    settings.local_url, model, label="local server", temperature=0.2)
            else:
                llm = build_llm(pid, settings, store)
            break
        except (ProviderError, ImportError):
            continue
    _cache = (time.time(), choice, llm)
    return llm


def provider_status(settings) -> dict:
    """Everything the settings screen needs. Keys are only ever returned masked."""
    store = ProviderStore(settings.home)
    data = store.load()
    out = []
    for spec in PROVIDERS.values():
        key, source = store.key(spec.id) if spec.cloud else (None, None)
        out.append({
            "id": spec.id, "name": spec.name, "cloud": spec.cloud, "notes": spec.notes,
            "key_url": spec.key_url, "key_hint": spec.key_hint, "env": list(spec.env),
            "configured": bool(key) if spec.cloud else None,
            "masked_key": mask(key), "key_source": source,
            "model": data["providers"].get(spec.id, {}).get("model") or spec.default_model,
        })
    return {"active": active_provider(settings, store), "providers": out}
