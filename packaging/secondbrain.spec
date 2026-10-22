# PyInstaller recipe for the Second Brain desktop app. Build on the target OS:
#     pyinstaller packaging/secondbrain.spec --noconfirm
# Output: dist/Second Brain.app (macOS), dist/SecondBrain/ (Windows, Linux).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = Path(SPECPATH).parent
ICONS = ROOT / "packaging" / "icons"
version = next(l.split('"')[1] for l in (ROOT / "pyproject.toml").read_text().splitlines() if l.startswith("version"))

datas = collect_data_files("secondbrain")  # the web UI in secondbrain/static
binaries, hiddenimports = [], []
# Packages with native libraries, data files or plugins that static analysis misses.
for pkg in ("fastembed", "onnxruntime", "tokenizers", "webview", "watchdog", "clr_loader", "pythonnet"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass  # not installed for this platform's build
for pkg in ("fastembed", "anthropic", "pypdf", "fastapi", "uvicorn"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
hiddenimports += ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                  "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on", "multipart"]

a = Analysis([str(ROOT / "packaging" / "entry.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, excludes=["tkinter", "matplotlib", "IPython", "pytest"], noarchive=False)
pyz = PYZ(a.pure)
mac = sys.platform == "darwin"
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Second Brain" if mac else "SecondBrain",
          console=False, icon=str(ICONS / ("icon.icns" if mac else "icon.ico")),
          target_arch=None, codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, name="SecondBrain")

if mac:
    app = BUNDLE(coll, name="Second Brain.app", icon=str(ICONS / "icon.icns"),
                 bundle_identifier="io.github.kaloyanstefanov66.secondbrain", version=version,
                 info_plist={"CFBundleShortVersionString": version, "CFBundleVersion": version,
                             "NSHighResolutionCapable": True, "LSApplicationCategoryType": "public.app-category.productivity",
                             "NSHumanReadableCopyright": "Your notes stay on your Mac."})
