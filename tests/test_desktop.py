import json
import socket
import threading
import urllib.request

from secondbrain.config import Settings
from secondbrain.desktop import Runtime, free_port, running_instance


def test_free_port_falls_back_when_taken():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        taken = s.getsockname()[1]
        assert free_port(taken) != taken


def test_runtime_serves_detects_itself_and_quits(tmp_path):
    settings = Settings.load(tmp_path / "home", embedder="hashing", llm="none", port=free_port(7790))
    rt = Runtime(settings)
    assert running_instance(rt.lock) is None
    quit_called = threading.Event()
    rt.quit = quit_called.set
    url = rt.start()
    try:
        assert json.loads(rt.lock.read_text())["url"] == url
        assert running_instance(rt.lock) == url  # a second launch would hand over to this one
        with urllib.request.urlopen(f"{url}/api/status") as r:
            status = json.loads(r.read())
        assert status["desktop"] is True and status["documents"] == 0
        req = urllib.request.Request(f"{url}/api/quit", data=b"", method="POST", headers={"Origin": url})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read()) == {"ok": True}
        assert quit_called.wait(3)
    finally:
        rt.stop()
    assert not rt.lock.exists()
