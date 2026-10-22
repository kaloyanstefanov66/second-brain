# Second Brain OS

**Search everything you know.** A local-first app that indexes your notes, PDFs, bookmarks and
documents into a searchable knowledge graph. Nothing leaves your machine.

- **Ask questions across your files, with citations.** Answers link to the exact passage (and PDF page) they came from.
- **Discover connections automatically.** It finds pairs of notes that talk about the same thing but that you never linked, and explains each pair with the bridging terms and the two passages that match best.
- **See how your ideas connect.** An interactive graph of your explicit links and semantic similarity, clustered and labelled by topic.
- **Private by design.** Local embeddings, a local LLM (optional), one SQLite file, and a server bound to `127.0.0.1`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings]"      # drop [embeddings] for the zero-download mode

brain add ~/Notes ~/Papers ~/Downloads/bookmarks.html
brain serve --watch --open          # http://localhost:7777
```

To try it on the bundled demo vault: `brain add sample_vault && brain serve --open`.

For generated answers (instead of quoted sentences), install [Ollama](https://ollama.com) and run
`ollama pull llama3.2`. Second Brain detects it automatically. Any OpenAI-compatible local server
(LM Studio, llama.cpp, vLLM) on `localhost:1234` also works.

## What it indexes

| Source | Notes |
|---|---|
| Markdown / text (`.md .txt .org .rst`) | Frontmatter (`title`, `tags`), `#tags`, `[[wikilinks]]` and relative links become graph edges. Obsidian vaults work as-is. |
| PDF | Text per page, so citations carry page numbers. Scanned PDFs need OCR first. |
| DOCX | Paragraphs and headings. |
| HTML pages | Readable text of saved pages. |
| Browser bookmarks | Netscape `bookmarks.html` export (every browser), or Chrome/Edge `Bookmarks` JSON. Folders become tags. Set `fetch_bookmarks = true` to also download and index each page; this is the only feature that touches the network, and it's off by default. |

Indexing is incremental. Unchanged files are skipped by mtime/size, then by content hash.
`--watch` re-indexes files a couple of seconds after they change.

## How it works

```
files ──► loaders ──► chunker ──► embedder ──► SQLite (brain.db)
                                               ├─ documents + chunks + vectors
                                               ├─ FTS5 full-text index (BM25)
                                               └─ edges: link | semantic
query ──► hybrid retrieval (vectors + BM25, reciprocal-rank fusion) ──► local LLM or extractive answer, cited [n]
```

- **Embeddings** (`--embedder`, default `auto`): `fastembed` runs `BAAI/bge-small-en-v1.5` on CPU via ONNX
  (about 70 MB, downloaded once, then offline). `ollama` uses `nomic-embed-text` from your local Ollama.
  `hashing` is a built-in lexical embedder that needs no model at all, weighted by corpus IDF at query time.
  `auto` uses the first one available and keeps using what the index was built with. If you switch backends,
  the index is re-embedded automatically. So if your first run fell back to `hashing` (for example, you were
  offline), upgrade later with `brain --embedder fastembed reindex`.
- **Semantic edges.** Each document is the mean of its chunk vectors. Edges are its top-k neighbours, kept
  relative to *this corpus's* similarity distribution (mean + 2.5σ for strong edges, and each document's best
  two neighbours if they clear mean + 0.75σ), so thresholds work across embedding models.
- **Discovered connections.** A semantic edge where the two documents share no tags and are not within two
  hops of each other through links you wrote.
- **Clusters.** Weighted label propagation over all edges, labelled with each cluster's distinctive terms.

## CLI

```
brain add PATH...          index files/folders and remember them as sources
brain reindex              re-scan all sources
brain remove PATH          stop tracking a source (your files are never modified)
brain search "query"       hybrid search
brain ask "question"       cited answer (--no-llm for extractive)
brain connections          list discovered connections
brain stats                what's indexed, which models are in use
brain serve [--watch] [--open] [--port 7777]
```

Global options: `--home DIR` (data directory, default `~/.secondbrain`, or `$SECONDBRAIN_HOME`) and `--embedder`.

## Configuration

Settings are read from `~/.secondbrain/config.toml`, then `SECONDBRAIN_*` environment variables:

```toml
embedder = "auto"                    # auto | fastembed | ollama | hashing
fastembed_model = "BAAI/bge-small-en-v1.5"
ollama_url = "http://127.0.0.1:11434"
ollama_embed_model = "nomic-embed-text"
llm = "auto"                         # auto | ollama | openai | none
llm_model = "llama3.2"
openai_url = "http://127.0.0.1:1234/v1"
chunk_chars = 1000
neighbors_per_doc = 6
min_similarity = 0.0                 # 0 = calibrated per embedder
fetch_bookmarks = false
```

## Privacy and security

- All data lives in `~/.secondbrain/brain.db`. Files you upload through the web UI are copied to `~/.secondbrain/inbox/`.
- The server listens on `127.0.0.1` only. It rejects requests whose `Host` header isn't local (to block DNS-rebinding attacks)
  and cross-site `Origin`s, so web pages you visit can't read or change your index.
- Embedding and LLM calls go only to the local endpoints you configure.
- Note content is rendered with an escaping Markdown renderer; the UI loads no third-party scripts, fonts or CDNs.

## API

The web UI is a thin client over a JSON API (interactive docs at `/api/docs`): `GET /api/search`,
`POST /api/ask` (NDJSON stream of `sources`, `meta`, `delta`, `done` events), `GET /api/graph`, `GET /api/connections`,
`GET /api/documents/{id}`, `POST /api/sources`, `POST /api/upload`, `POST /api/reindex`, `GET /api/status`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover loaders, chunking, incremental indexing, graph and discovery rules, embedder switching,
extractive answers, LLM streaming against a fake local Ollama, and the HTTP API with its local-only guard.

## Limits and next steps

- The vector index is an in-memory NumPy matrix: instant up to a few hundred thousand chunks. Beyond that, swap in an ANN index.
- No OCR for scanned PDFs yet.
- Roadmap ideas: an import for Notion and Readwise exports, time-decay ranking for notes you never revisit, and per-cluster summaries.
