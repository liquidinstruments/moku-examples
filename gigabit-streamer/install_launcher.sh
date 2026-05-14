#!/bin/bash
# install_launcher.sh — installs Moku:Delta Streamer GUI as a desktop application
# Run once after copying the GigabitStreamerCode folder to your machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICON_DIR="$HOME/.local/share/icons"
APP_DIR="$HOME/.local/share/applications"
WRAPPER="$SCRIPT_DIR/moku_streamer_launch.sh"

echo "============================================"
echo "  Moku:Delta Streamer — Desktop Installer"
echo "============================================"
echo ""

# ── 1. Check Python 3 ──────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install it with:"
    echo "  sudo apt install python3"
    exit 1
fi

# ── 2. Check / install tkinter ─────────────────────────────────────────────
python3 -c "import tkinter" 2>/dev/null || {
    echo "tkinter not found. Installing python3-tk..."
    sudo apt install -y python3-tk
}

# ── 3. Check / install Pillow ──────────────────────────────────────────────
python3 -c "import PIL" 2>/dev/null || {
    echo "Pillow not found. Installing..."
    pip3 install pillow --break-system-packages 2>/dev/null \
        || pip3 install pillow
}

# ── 4. Create local dirs ───────────────────────────────────────────────────
mkdir -p "$ICON_DIR" "$APP_DIR"

# ── 5. Copy icon ───────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/moku_streamer.png" ]; then
    cp "$SCRIPT_DIR/moku_streamer.png" "$ICON_DIR/moku-streamer.png"
    echo "Icon installed  →  $ICON_DIR/moku-streamer.png"
else
    echo "WARNING: moku_streamer.png not found — icon will be blank."
fi

# ── 6. Create wrapper launcher script ─────────────────────────────────────
# The .desktop file uses this wrapper so the working directory is always
# the GigabitStreamerCode folder regardless of how the app is launched.
cat > "$WRAPPER" << WRAPPER_EOF
#!/bin/bash
cd "$SCRIPT_DIR"
exec python3 "$SCRIPT_DIR/moku_gui.py" "\$@"
WRAPPER_EOF
chmod +x "$WRAPPER"
echo "Launcher script  →  $WRAPPER"

# ── 7. Write .desktop entry ────────────────────────────────────────────────
cat > "$APP_DIR/moku-streamer.desktop" << DESKTOP_EOF
[Desktop Entry]
Version=1.0
Name=Moku Streamer
GenericName=Gigabit Streamer
Comment=Moku:Delta Gigabit Streamer — capture and transmit DIFI/UDP streams
Exec=$WRAPPER
Icon=moku-streamer
Terminal=false
Type=Application
Categories=Science;Engineering;Utility;
Keywords=moku;delta;streamer;capture;DIFI;UDP;
StartupNotify=true
DESKTOP_EOF

chmod +x "$APP_DIR/moku-streamer.desktop"
echo ".desktop entry  →  $APP_DIR/moku-streamer.desktop"

# ── 8. Refresh application database ───────────────────────────────────────
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$APP_DIR" 2>/dev/null || true
fi

# ── 9. (Optional) pin to GNOME favourites ─────────────────────────────────
# Uncomment the block below if you use GNOME Shell and want the app pinned
# to the dock automatically.
#
# if command -v gsettings &>/dev/null; then
#     CURRENT=$(gsettings get org.gnome.shell favorite-apps)
#     if [[ "$CURRENT" != *"moku-streamer"* ]]; then
#         NEW="${CURRENT%]*}, 'moku-streamer.desktop']"
#         gsettings set org.gnome.shell favorite-apps "$NEW"
#         echo "Pinned to GNOME dock."
#     fi
# fi

echo ""
echo "✔  Installation complete."
echo ""
echo "You can now launch Moku Streamer from your application menu"
echo "(look under Science / Utilities, or just search 'Moku')."
echo ""
echo "Alternatively, double-click the launcher:"
echo "  $APP_DIR/moku-streamer.desktop"
echo ""
echo "Or run directly any time with:"
echo "  python3 $SCRIPT_DIR/moku_gui.py"
echo ""
