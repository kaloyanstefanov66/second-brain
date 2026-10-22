"""Settings, loaded from defaults < ~/.secondbrain/config.toml < environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def default_home() -> Path:
    return Path(os.environ.get("SECONDBRAIN_HOME", Path.home() / ".secondbrain")).expanduser()


@dataclass
class Settings:
    home: Path = field(default_factory=default_home)

    # Embeddings: "auto" tries fastembed, then Ollama, then the built-in hashing embedder.
    embedder: str = "auto"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Question answering. "auto" uses local engines only: Ollama, then an OpenAI-compatible
    # local server (LM Studio, llama.cpp, vLLM), else extractive answers. Cloud providers
    # (anthropic, openai, gemini, github) are used only when selected; the choice made in the
    # web UI or with `brain keys use` (stored in providers.json) overrides this.
    llm: str = "auto"
    llm_model: str = "llama3.2"
    local_url: str = "http://127.0.0.1:1234/v1"

    # Indexing
    chunk_chars: int = 1000
    chunk_overlap: int = 150
    max_file_mb: int = 50
    fetch_bookmarks: bool = False  # download bookmarked pages (the only network access, off by default)

    # Graph
    neighbors_per_doc: int = 6
    min_similarity: float = 0.0  # 0 = use the embedder's calibrated default

    # Server
    host: str = "127.0.0.1"
    port: int = 7777

    @property
    def db_path(self) -> Path:
        return self.home / "brain.db"

    @property
    def inbox(self) -> Path:
        return self.home / "inbox"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def ensure_dirs(self) -> None:
        for p in (self.home, self.inbox, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, home: Path | str | None = None, **overrides) -> "Settings":
        s = cls(home=Path(home).expanduser() if home else default_home())
        cfg = s.home / "config.toml"
        values: dict = {}
        if cfg.exists():
            values.update(tomllib.loads(cfg.read_text()))
        for f in fields(cls):
            env = os.environ.get(f"SECONDBRAIN_{f.name.upper()}")
            if env is not None and f.name != "home":
                values[f.name] = env
        values.update({k: v for k, v in overrides.items() if v is not None})
        for f in fields(cls):
            if f.name in values and f.name != "home":
                setattr(s, f.name, _coerce(values[f.name], type(getattr(s, f.name))))
        s.ensure_dirs()
        return s


def _coerce(value, typ):
    if typ is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return typ(value)
