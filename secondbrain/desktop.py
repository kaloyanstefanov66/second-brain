"""Second Brain as a desktop app: a native window around the local server.

Double-click the app: a window opens straight away with a loading screen, the server starts in the
background (first run may download the local embedding model), then the app appears. Closing the
window quits. Launching it again while it runs just opens the running instance.

Without a native web view (pywebview missing, or no GUI toolkit on Linux) it falls back to your
default browser and keeps running until you choose Quit in the app.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from .config import Settings

log = logging.getLogger("secondbrain")

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
:root { color-scheme: light dark; }
body { margin: 0; height: 100vh; display: grid; place-items: center; background: #f6f6f4; color: #52514e;
       font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (prefers-color-scheme: dark) { body { background: #121211; color: #c3c2b7; } }
.box { text-align: center; max-width: 360px; padding: 0 24px; }
.spin { width: 22px; height: 22px; border: 3px solid #cfcdc6; border-top-color: #6250d6; border-radius: 50%;
        animation: r .8s linear infinite; margin: 0 auto 16px; }
@keyframes r { to { transform: rotate(360deg); } }
h1 { font-size: 17px; margin: 0 0 4px; color: inherit; font-weight: 600; }
.err .spin { display: none; } .err #msg { color: #c93b3a; }
</style></head><body><div class="box" id="box"><div class="spin"></div><h1>Second Brain</h1>
<div id="msg">Starting…</div></div></body></html>"""


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def running_instance(lock: Path) -> str | None:
    """URL of an already running Second Brain, if the lock file points at a live server."""
    try:
        url = json.loads(lock.read_text())["url"]
        with urllib.request.urlopen(f"{url}/api/status", timeout=1.5) as r:
            return url if r.status == 200 else None
    except (OSError, ValueError, KeyError):
        return None


def setup_logging(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(home / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    if sys.stderr and sys.stderr.isatty():
        root.addHandler(logging.StreamHandler())


class Runtime:
    """Owns the Brain, the folder watcher and the uvicorn server for one app session."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = settings.home / "desktop.json"
        self.server = None
        self.url: str | None = None
        self.quit = lambda: None  # replaced by the UI layer: close the window / stop waiting

    def start(self, say=lambda msg: None) -> str:
        import uvicorn

        from .brain import Brain
        from .embeddings import _model_cached, lower_priority
        from .server import create_app
        from .watcher import FolderWatcher

        lower_priority()
        msg = "Opening your library…"
        if self.settings.embedder in ("auto", "fastembed") and not _model_cached(self.settings):
            try:
                import fastembed  # noqa: F401

                # the download happens while the Brain loads its embedder, so say so up front
                msg = "Downloading the local language model (one time, about 70 MB). This can take a minute…"
            except ImportError:
                pass
        say(msg)
        brain = Brain(self.settings)
        if brain.ensure_ready():
            say("Updating your library for the new model…")
        brain.on_quit = lambda: self.quit()
        FolderWatcher(brain).start()
        brain.catch_up()

        port = free_port(self.settings.port)
        config = uvicorn.Config(create_app(brain), host="127.0.0.1", port=port, log_level="warning",
                                log_config=None)
        self.server = uvicorn.Server(config)
        threading.Thread(target=self.server.run, name="server", daemon=True).start()
        deadline = time.time() + 30
        while not self.server.started:
            if time.time() > deadline or not threading.main_thread().is_alive():
                raise RuntimeError("the local server did not start")
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{port}"
        self.lock.write_text(json.dumps({"url": self.url, "pid": os.getpid()}))
        log.info("Second Brain running at %s", self.url)
        return self.url

    def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        try:
            if json.loads(self.lock.read_text()).get("pid") == os.getpid():
                self.lock.unlink()
        except (OSError, ValueError):
            pass


def run_window(rt: Runtime) -> bool:
    """Run inside a native window. Returns False if no native web view is available."""
    if sys.platform.startswith("linux"):
        import importlib.util

        if not any(importlib.util.find_spec(m) for m in ("gi", "qtpy")):
            return False  # no GTK or Qt bindings: pywebview can't open a window here
    try:
        import webview
    except ImportError:
        return False

    window = webview.create_window("Second Brain", html=LOADING_HTML, width=1280, height=840, min_size=(420, 560),
                                   background_color="#f6f6f4", text_select=True)
    started = threading.Event()

    def say(msg: str) -> None:
        try:
            window.evaluate_js(f"document.getElementById('msg').textContent = {json.dumps(msg)}")
        except Exception:
            pass

    def boot():
        started.set()
        try:
            url = rt.start(say)
        except Exception as e:
            log.exception("startup failed")
            say(f"Couldn't start: {e}. Details are in {rt.settings.home / 'app.log'}")
            try:
                window.evaluate_js("document.getElementById('box').className = 'box err'")
            except Exception:
                pass
            return
        window.load_url(url)

    rt.quit = window.destroy
    try:
        webview.start(boot, private_mode=False, storage_path=str(rt.settings.home / "webview"))
    except Exception:
        if started.is_set():
            raise
        log.warning("no native web view available, using the browser instead", exc_info=True)
        return False
    return True


def run_browser(rt: Runtime) -> None:
    url = rt.start(lambda msg: print(msg, file=sys.stderr))
    print(f"Second Brain is running at {url}. Use Quit in the app (or Ctrl+C) to stop it.", file=sys.stderr)
    webbrowser.open(url)
    stop = threading.Event()
    rt.quit = stop.set
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="second-brain", description="Second Brain desktop app")
    p.add_argument("--home", help="data directory (default ~/.secondbrain)")
    p.add_argument("--browser", action="store_true", help="use your web browser instead of an app window")
    args, _ = p.parse_known_args(argv)  # ignore e.g. macOS -psn_* arguments

    settings = Settings.load(args.home)
    setup_logging(settings.home)
    rt = Runtime(settings)
    existing = running_instance(rt.lock)
    if existing:
        log.info("already running at %s", existing)
        webbrowser.open(existing)
        return
    try:
        if args.browser or os.environ.get("SECONDBRAIN_BROWSER") or not run_window(rt):
            run_browser(rt)
    finally:
        rt.stop()


if __name__ == "__main__":
    main()
