"""Command line: `brain add ~/notes`, `brain ask "..."`, `brain serve`."""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import webbrowser

from .config import Settings


def _brain(args):
    from .brain import Brain

    overrides = {"embedder": getattr(args, "embedder", None)}
    brain = Brain(Settings.load(args.home, **overrides))
    if brain.ensure_ready():
        print(f"Re-embedded index with {brain.embedder.describe()}", file=sys.stderr)
    return brain


def _progress(p: dict) -> None:
    if not sys.stderr.isatty():
        return
    if p.get("stage") == "indexing" and p.get("total"):
        name = p.get("file", "")[-60:]
        print(f"\r  [{p['current'] + 1}/{p['total']}] {name:<60}", end="", file=sys.stderr, flush=True)
    elif p.get("stage") == "building graph":
        print("\r  building graph…" + " " * 70, end="", file=sys.stderr, flush=True)


def cmd_add(args):
    brain = _brain(args)
    brain.indexer.on_progress = _progress
    print(f"Indexing with {brain.embedder.describe()}", file=sys.stderr)
    report = brain.add(args.paths)
    print("\r" + " " * 90 + "\r", end="", file=sys.stderr)
    print(report.summary())
    for f in report.failed[:20]:
        print(f"  ! {f['path']}: {f['error']}", file=sys.stderr)


def cmd_reindex(args):
    brain = _brain(args)
    brain.indexer.on_progress = _progress
    report = brain.reindex()
    print("\r" + " " * 90 + "\r", end="", file=sys.stderr)
    print(report.summary())


def cmd_remove(args):
    brain = _brain(args)
    print(f"Removed {brain.remove(args.path)} documents")


def cmd_search(args):
    brain = _brain(args)
    hits = brain.search(args.query, k=args.k)
    if not hits:
        print("No results.")
    for i, h in enumerate(hits, 1):
        loc = " · ".join(x for x in (h.heading, f"p. {h.page}" if h.page else "") if x)
        print(f"{i:>2}. {h.title}  [{h.kind}]" + (f"  — {loc}" if loc else ""))
        print(textwrap.indent(textwrap.fill(h.snippet, 96), "    "))
        print(f"    {h.url or h.path}\n")


def cmd_ask(args):
    brain = _brain(args)
    sources = []
    for ev in brain.ask_stream(args.question, use_llm=not args.no_llm):
        if ev["type"] == "sources":
            sources = ev["sources"]
        elif ev["type"] == "meta" and ev.get("mode") == "llm":
            print(f"({ev['model']})\n", file=sys.stderr)
        elif ev["type"] == "delta":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "error":
            print(f"\nerror: {ev['message']}", file=sys.stderr)
    print("\n\nSources:")
    for s in sources:
        loc = " · ".join(x for x in (s["heading"], f"p. {s['page']}" if s["page"] else "") if x)
        print(f"  [{s['n']}] {s['title']}" + (f" — {loc}" if loc else ""))


def cmd_connections(args):
    brain = _brain(args)
    conns = brain.connections(args.limit)
    if not conns:
        print("No discovered connections yet (add more documents, or lower min_similarity).")
    for c in conns:
        print(f"{c['similarity']:.2f}  {c['a']['title']}  ↔  {c['b']['title']}")
        if c["terms"]:
            print(f"      bridging: {', '.join(c['terms'])}")


def cmd_stats(args):
    brain = _brain(args)
    s = brain.status()
    print(f"Home:        {s['home']}")
    print(f"Documents:   {s['documents']}  " + "  ".join(f"{k}={v}" for k, v in s["by_kind"].items()))
    print(f"Chunks:      {s['chunks']}   Words: {s['words']:,}")
    print(f"Graph:       {s['links']} links, {s['semantic']} semantic edges, {s['discovered']} discovered")
    print(f"Embeddings:  {s['embedder']['description']}")
    print(f"Answers:     {s['llm'] or 'extractive (no local LLM found: install Ollama for generated answers)'}")
    print("Sources:     " + ("\n             ".join(s["roots"]) or "none (run `brain add <folder>`)"))


def cmd_serve(args):
    import uvicorn

    from .server import create_app

    brain = _brain(args)
    if args.watch:
        from .watcher import FolderWatcher

        FolderWatcher(brain).start()
    host, port = args.host or brain.settings.host, args.port or brain.settings.port
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"Second Brain OS running at {url}  (embeddings: {brain.embedder.describe()})")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(create_app(brain), host=host, port=port, log_level="warning")


def main(argv=None):
    p = argparse.ArgumentParser(prog="brain", description="Second Brain OS: search everything you know, locally.")
    p.add_argument("--home", help="data directory (default ~/.secondbrain or $SECONDBRAIN_HOME)")
    p.add_argument("--embedder", choices=["auto", "fastembed", "ollama", "hashing"], help="embedding backend")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="index files or folders (and keep them as sources)")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("reindex", help="re-scan all sources for changes")
    s.set_defaults(fn=cmd_reindex)

    s = sub.add_parser("remove", help="stop tracking a source and drop its documents")
    s.add_argument("path")
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("search", help="hybrid semantic + keyword search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="ask a question, answered with citations")
    s.add_argument("question")
    s.add_argument("--no-llm", action="store_true", help="force the extractive answerer")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("connections", help="list automatically discovered connections")
    s.add_argument("-n", "--limit", type=int, default=20)
    s.set_defaults(fn=cmd_connections)

    s = sub.add_parser("stats", help="show what's indexed and which models are in use")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("serve", help="start the local web app")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--watch", action="store_true", help="re-index sources automatically when files change")
    s.add_argument("--open", action="store_true", help="open the browser")
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
