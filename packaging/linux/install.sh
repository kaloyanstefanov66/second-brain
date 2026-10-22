#!/usr/bin/env sh
# Installs Second Brain for the current user: app in ~/.local/share/second-brain,
# a `second-brain` command in ~/.local/bin, and an entry in your applications menu.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/second-brain"
BIN="$HOME/.local/bin"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

rm -rf "$DEST"
mkdir -p "$DEST" "$BIN" "$APPS"
cp -R "$HERE/SecondBrain/." "$DEST/"
cp "$HERE/icon.png" "$DEST/icon.png"
ln -sf "$DEST/SecondBrain" "$BIN/second-brain"

cat > "$APPS/second-brain.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Second Brain
Comment=Search everything you know
Exec=$DEST/SecondBrain
Icon=$DEST/icon.png
Terminal=false
Categories=Office;Utility;
DESKTOP

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
echo "Installed. Open \"Second Brain\" from your applications menu, or run: second-brain"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "(Add $BIN to your PATH to use the second-brain command.)";; esac
