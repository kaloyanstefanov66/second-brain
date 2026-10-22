# Second Brain

**Search everything you know.** A local-first app that indexes your notes, PDFs, bookmarks and
documents into a searchable knowledge graph. Nothing leaves your machine.

Open it and it finds your notes for you: Obsidian and Logseq vaults, note folders, Documents, Desktop,
cloud drives and your browsers' bookmarks. One click indexes everything, and Home shows every note
sorted into topics it works out on its own.

- **Ask questions across your files, with citations.** Answers link to the exact passage (and PDF page) they came from.
- **Discover connections automatically.** It finds pairs of notes that talk about the same thing but that you never linked, and explains each pair with the bridging terms and the two passages that match best.
- **See how your ideas connect.** An interactive graph of your explicit links and semantic similarity, clustered and labelled by topic.
- **Private by design.** Local embeddings, a local LLM (optional), one SQLite file, and a server bound to `127.0.0.1`.
- **Or bring your own API key.** Claude, OpenAI, Google Gemini or GitHub Models (Copilot) can write the answers instead, if you choose to.

## Download the app

Get the latest release from the [Releases page](https://github.com/kaloyanstefanov66/second-brain/releases/latest):

| | File | Install |
|---|---|---|
| macOS (Apple silicon) | `Second-Brain-macOS.dmg` | Open it, drag **Second Brain** to Applications. The app isn't notarized yet, so the first time right-click it → **Open** → **Open**. |
| Windows 10/11 | `Second-Brain-Setup.exe` | Run it (no admin needed). If SmartScreen appears: **More info → Run anyway**. |
| Linux | `Second-Brain-Linux.tar.gz` | Extract, run `./install.sh`, then open *Second Brain* from your app menu. Opens in your browser unless GTK or Qt Python bindings are installed. |

Double-click it and a window opens. It finds your notes, and closing the window quits it. Your library lives in
`~/.secondbrain` and is shared with the command-line version. The first launch downloads the local embedding
model once (about 70 MB). Logs are in `~/.secondbrain/app.log`.

## Quick start (from source)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings]"      # drop [embeddings] for the zero-download mode

brain                               # opens http://localhost:7777
```

That's it. On first launch the app looks for your notes, shows what it found (pre-selected), and indexes
it when you click **Index**. From then on it stays in sync by itself: folders are watched for changes, and
browser bookmarks are re-read every 15 minutes.

Prefer the terminal? `brain discover --add` does the same discovery, and `brain add PATH` adds anything it missed.
To try the bundled demo instead: `brain add sample_vault && brain`.

For generated answers (instead of quoted sentences), either run a local model: install [Ollama](https://ollama.com)
and `ollama pull llama3.2`, or any OpenAI-compatible server (LM Studio, llama.cpp, vLLM) on `localhost:1234`,
both detected automatically. Or add a cloud API key (see below).

## How it finds your notes

**Discovery** (first launch, *Find more notes* on Home, or `brain discover`) looks at folder structure and file
names in your home folder, not file contents. It's time-limited to about 12 seconds:

| Finds | How |
|---|---|
| Obsidian vaults, Logseq graphs | folders containing `.obsidian/`, or `logseq/` + `pages/`, anywhere under your home folder, plus Obsidian's iCloud folder |
| Note folders | top-level folders (Documents, Desktop, Notes, Dropbox, OneDrive, Google Drive, iCloud Drive, ...) that contain notes or documents |
| Browser bookmarks | read in place from Chrome, Edge, Brave, Vivaldi, Arc, Chromium, Opera, Firefox and Safari profiles, no export needed |

It skips system and media folders (`Library`, `AppData`, Pictures, Music, Movies...), hidden folders, and code
projects. Downloads and very large folders are listed but not pre-selected. Nothing is indexed until you confirm,
and you can untick anything. (Safari's bookmarks need Full Disk Access on macOS. Apple Notes, Bear and Notion
keep notes in their own databases or the cloud, so export them to Markdown into a folder first.)

**Indexing** then reads only the confirmed sources, plus folders you add yourself (Library page or `brain add`)
and files you upload. Inside a source it walks subfolders, but only opens supported files
(`.md .markdown .txt .org .rst .pdf .docx .html .htm`, plus bookmark files). It skips hidden folders, `.git`,
`node_modules`, virtualenvs, build output, **code projects** (a git checkout with a `package.json`,
`pyproject.toml`, `Cargo.toml`...), and anything over 50 MB.
Re-indexing re-reads only files whose size or modification time changed, and the watcher watches only your
source folders. Your files are never modified; their text is copied into `~/.secondbrain/brain.db`, and
`brain remove PATH` deletes those copies, not the files.

## Answer engines and API keys

| Engine | Where it runs | Key |
|---|---|---|
| Auto (default) | Ollama or a local server if running, otherwise quoted sentences | none |
| Ollama / Local server | your machine | none |
| Claude | Anthropic API, via the official `anthropic` SDK (default model `claude-opus-5-5`) | `ANTHROPIC_API_KEY` |
| OpenAI | OpenAI API (default `gpt-5-mini`) | `OPENAI_API_KEY` |
| Google Gemini | Gemini API (default `gemini-2.5-flash`) | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| GitHub Models (Copilot) | GitHub's model catalog (default `openai/gpt-4.1-mini`) | `GITHUB_TOKEN`, a token with the *Models* permission |

Add a key in **Settings** in the web app, or from the terminal:

```bash
brain keys set anthropic        # prompts for the key (hidden input), verifies it, saves it
brain keys use anthropic        # answer with Claude (or: auto, none, ollama, local, openai, gemini, github)
brain keys list                 # shows masked keys and the active engine
brain keys remove anthropic
```

- Keys are verified (by listing the provider's models) before they are saved, and stored in
  `~/.secondbrain/providers.json` with file mode `0600`. Keys already in your environment variables are picked up automatically.
  The API and UI only ever show masked keys.
- **Auto never uses the cloud.** A cloud provider answers only after you select it.
- **What a cloud provider sees:** your question, the ~8 passages retrieved for it, and the last few chat turns.
  Not your library, your index or your embeddings, which always stay local.
- GitHub Copilot has no general-purpose API for third-party apps. The supported route is **GitHub Models**, which serves
  the Copilot model catalog with a GitHub token and bills against your GitHub Models / Copilot allowance.
- For Claude Opus 5.5 / Sonnet 5.5 requests, Anthropic's server-side refusal fallback (`fallbacks: "default"`) is enabled.

## What it indexes

| Source | Notes |
|---|---|
| Markdown / text (`.md .txt .org .rst`) | Frontmatter (`title`, `tags`), `#tags`, `[[wikilinks]]` and relative links become graph edges. Obsidian vaults work as-is. |
| PDF | Text per page, so citations carry page numbers. Scanned PDFs need OCR first. |
| DOCX | Paragraphs and headings. |
| HTML pages | Readable text of saved pages. |
| Browser bookmarks | Netscape `bookmarks.html` export (every browser), or Chrome/Edge `Bookmarks` JSON. Folders become tags. Set `fetch_bookmarks = true` to also download and index each page; this is the only feature that touches the network, and it's off by default. |

Indexing is incremental. Unchanged files are skipped by mtime/size, then by content hash.
The first index is the heavy part: it runs the embedding model over every passage. To keep the computer responsive,
the app runs at background priority and the model uses half your CPU cores (`index_threads`). You can **pause**
indexing from Home or the Library at any time: it stays paused across restarts until you resume, and search
keeps working with what's indexed so far. When the app starts, it finishes an interrupted index and picks up
changes made while it was closed.
The server re-indexes changed files a couple of seconds after they change (`--no-watch` turns this off).

## Home: everything, organised

Home lists every document exactly once: grouped into **topics**, which are clusters of notes that link to or resemble
each other, each named after its most distinctive words. Above them sit your **recently edited** notes, and below them
whatever doesn't belong to a topic yet. Topics update automatically as you write.

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
brain keys list|set|remove|use [PROVIDER]   manage API keys and the answer engine
brain                      start the app and open it in your browser
brain discover [--add]     find vaults, note folders and browser bookmarks (and index them)
brain serve [--no-watch] [--open] [--port 7777]
```

Global options: `--home DIR` (data directory, default `~/.secondbrain`, or `$SECONDBRAIN_HOME`) and `--embedder`.

## Configuration

Settings are read from `~/.secondbrain/config.toml`, then `SECONDBRAIN_*` environment variables:

```toml
embedder = "auto"                    # auto | fastembed | ollama | hashing
fastembed_model = "BAAI/bge-small-en-v1.5"
ollama_url = "http://127.0.0.1:11434"
ollama_embed_model = "nomic-embed-text"
llm = "auto"                         # auto | none | ollama | local | anthropic | openai | gemini | github
llm_model = "llama3.2"               # default local model
local_url = "http://127.0.0.1:1234/v1"  # OpenAI-compatible local server
chunk_chars = 1000
index_threads = 0                    # CPU threads for the embedding model; 0 = half your cores
neighbors_per_doc = 6
min_similarity = 0.0                 # 0 = calibrated per embedder
fetch_bookmarks = false
```

## Privacy and security

- All data lives in `~/.secondbrain/brain.db`. Files you upload through the web UI are copied to `~/.secondbrain/inbox/`.
- The server listens on `127.0.0.1` only. It rejects requests whose `Host` header isn't local (to block DNS-rebinding attacks)
  and cross-site `Origin`s, so web pages you visit can't read or change your index.
- Embeddings are always computed locally. Answer generation is local unless you select a cloud provider.
- Note content is rendered with an escaping Markdown renderer; the UI loads no third-party scripts, fonts or CDNs.

## API

The web UI is a thin client over a JSON API (interactive docs at `/api/docs`): `GET /api/search`,
`POST /api/ask` (NDJSON stream of `sources`, `meta`, `delta`, `done` events), `GET /api/graph`, `GET /api/connections`,
`GET /api/documents/{id}`, `POST /api/sources`, `POST /api/upload`, `POST /api/reindex`, `GET /api/status`,
`GET/PUT /api/providers…` (keys are write-only).

## Building the desktop app

The app is the same Python code in a native window ([pywebview](https://pywebview.flowrl.com/): WebKit on macOS,
Edge WebView2 on Windows), bundled with PyInstaller. Each OS builds its own app:

```bash
pip install -e ".[desktop,embeddings]" pyinstaller
pyinstaller packaging/secondbrain.spec --noconfirm   # → dist/Second Brain.app or dist/SecondBrain/
second-brain                                         # or run the windowed app straight from source
```

`.github/workflows/release.yml` builds all three on GitHub's macOS, Windows and Linux machines, packages them
(`.dmg`, an Inno Setup installer, and a `.tar.gz` with `install.sh`), and publishes them as a GitHub Release
when you push a version tag:

```bash
git tag v0.1.0 && git push origin v0.1.0     # or: git push origin main:release (creates the tag from pyproject's version)
```

It can also be run by hand from the Actions tab to get the builds without a release. The icons come from
`packaging/make_icons.py`. Builds are unsigned: removing the macOS and Windows warnings needs an Apple
Developer ID (notarization) and a Windows code-signing certificate.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover loaders, chunking, incremental indexing, graph and discovery rules, embedder switching,
extractive answers, LLM streaming against a fake local Ollama, the HTTP API with its local-only guard,
and the cloud providers against fake servers (key validation, masking, file permissions, "auto never uses the cloud",
streaming through the Anthropic SDK and the OpenAI-compatible APIs).

## Limits and next steps

- The vector index is an in-memory NumPy matrix: instant up to a few hundred thousand chunks. Beyond that, swap in an ANN index.
- No OCR for scanned PDFs yet.
- Roadmap ideas: an import for Notion and Readwise exports, time-decay ranking for notes you never revisit, and per-cluster summaries.
