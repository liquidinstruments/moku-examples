#!/usr/bin/env python3
"""
Moku:Delta Gigabit Streamer GUI
================================
Graphical front-end for capture_it, send_it, and validate_capture.py.

Launch:
    python3 moku_gui.py

Desktop icon launcher is created by install_launcher.sh.
"""

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

# ── locate tool directory (same folder as this script) ───────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour scheme ─────────────────────────────────────────────────────────
BG          = "#1e2a38"   # main background
BG2         = "#253447"   # panel background
ACCENT      = "#2E75B6"   # blue accent
ACCENT2     = "#1F4E79"   # darker blue
FG          = "#e8edf2"   # main text
FG2         = "#a0b4c8"   # dimmed text
GREEN       = "#4caf50"
AMBER       = "#ff9800"
RED         = "#ef5350"
BTN_START   = "#2e7d32"   # green start button
BTN_STOP    = "#c62828"   # red stop button
BTN_FG      = "#ffffff"
ENTRY_BG    = "#16202d"
ENTRY_FG    = "#e8edf2"

FONT_LABEL  = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 9, "bold")
FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_MONO   = ("Courier New", 8)


# ══════════════════════════════════════════════════════════════════════════════
#  Password helper
# ══════════════════════════════════════════════════════════════════════════════
def ask_password(parent, cached):
    """Return cached password or prompt the user.  Returns None if cancelled."""
    if cached:
        return cached
    pw = simpledialog.askstring(
        "sudo password",
        "Enter your sudo password to enable real-time scheduling:",
        show="*", parent=parent)
    return pw   # None if the user cancelled


def build_command(binary_path, args_dict, use_sudo, password):
    """Return (cmd_list, stdin_bytes_or_None)."""
    cmd = []
    stdin_data = None
    if use_sudo and password:
        cmd = ["sudo", "-S", "-k"]   # -k: ignore cached credentials
        stdin_data = (password + "\n").encode()
    cmd.append(binary_path)
    for k, v in args_dict.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(k)
        else:
            cmd.extend([k, str(v)])
    return cmd, stdin_data


# ══════════════════════════════════════════════════════════════════════════════
#  Subprocess panel (shared logic for capture and send)
# ══════════════════════════════════════════════════════════════════════════════
class ProcessPanel:
    """Manages one subprocess: start, stop, stdout/stderr streaming."""

    def __init__(self):
        self.proc        = None
        self.out_queue   = queue.Queue()
        self._reader     = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, stdin_data=None):
        if self.running():
            return
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE  if stdin_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if stdin_data:
            try:
                self.proc.stdin.write(stdin_data.decode())
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    def _read_output(self):
        try:
            for line in self.proc.stdout:
                self.out_queue.put(line.rstrip())
        except Exception:
            pass
        self.out_queue.put(None)   # sentinel


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers for building form rows
# ══════════════════════════════════════════════════════════════════════════════
def make_frame(parent, **kw):
    return tk.Frame(parent, bg=BG2, **kw)

def label(parent, text, **kw):
    return tk.Label(parent, text=text, bg=BG2, fg=FG, font=FONT_LABEL, **kw)

def entry(parent, textvariable, width=14):
    e = tk.Entry(parent, textvariable=textvariable,
                 bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
                 relief="flat", bd=4, font=FONT_LABEL, width=width)
    return e

def check(parent, text, variable):
    return tk.Checkbutton(parent, text=text, variable=variable,
                          bg=BG2, fg=FG2, selectcolor=ENTRY_BG,
                          activebackground=BG2, activeforeground=FG,
                          font=FONT_LABEL)

def section_label(parent, text):
    f = tk.Frame(parent, bg=ACCENT2)
    tk.Label(f, text=text, bg=ACCENT2, fg=FG, font=FONT_BOLD,
             padx=8, pady=3).pack(side="left")
    return f

def button(parent, text, command, color=ACCENT, width=18):
    return tk.Button(parent, text=text, command=command,
                     bg=color, fg=BTN_FG, font=FONT_BOLD,
                     relief="flat", bd=0, padx=8, pady=5,
                     activebackground=color, activeforeground=BTN_FG,
                     cursor="hand2", width=width)

def log_box(parent):
    box = scrolledtext.ScrolledText(
        parent, bg="#0d1520", fg=FG, font=FONT_MONO,
        relief="flat", bd=4, wrap="word", state="disabled",
        height=12)
    box.tag_config("info",  foreground=FG)
    box.tag_config("stat",  foreground="#4fc3f7")
    box.tag_config("good",  foreground=GREEN)
    box.tag_config("warn",  foreground=AMBER)
    box.tag_config("error", foreground=RED)
    return box

def log_append(box, text, tag="info"):
    box.config(state="normal")
    box.insert("end", text + "\n", tag)
    box.see("end")
    box.config(state="disabled")

def classify_line(line):
    """Pick a colour tag for a log line."""
    lo = line.lower()
    if any(w in lo for w in ("pass", "no packets lost", "memory locked",
                              "pre-faulted", "listening", "transmitting")):
        return "good"
    if any(w in lo for w in ("gap", "fail", "error", "drop", "lost", "warn")):
        return "error"
    if any(w in lo for w in ("pkts", "mb/s", "kpkt", "written", "loop",
                              "elapsed", "duration", "progress", "scanning",
                              "int16s scanned", "samples/ch")):
        return "stat"
    return "info"


# ══════════════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════════════
class MokuGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Moku:Delta Gigabit Streamer")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(820, 660)

        # Set window icon
        icon_path = os.path.join(SCRIPT_DIR, "moku_streamer.png")
        if os.path.exists(icon_path):
            try:
                ico = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, ico)
                self._icon_ref = ico   # keep reference
            except Exception:
                pass

        self.sudo_password = None   # cached for the session

        self._build_ui()
        self.root.after(100, self._poll_capture)
        self.root.after(100, self._poll_send)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self.root, bg=ACCENT2, height=48)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Moku:Delta  Gigabit Streamer",
                 bg=ACCENT2, fg=FG, font=("Segoe UI", 13, "bold"),
                 pady=10).pack(side="left")

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",        background=BG,  borderwidth=0)
        style.configure("TNotebook.Tab",    background=BG2, foreground=FG2,
                        padding=[14, 6],    font=FONT_BOLD)
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT2)],
                  foreground=[("selected", FG)])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        cap_tab  = tk.Frame(nb, bg=BG)
        send_tab = tk.Frame(nb, bg=BG)
        cfg_tab  = tk.Frame(nb, bg=BG)

        nb.add(cap_tab,  text="  📥  Capture  ")
        nb.add(send_tab, text="  📤  Transmit  ")
        nb.add(cfg_tab,  text="  ⚙   Settings  ")

        self._build_capture_tab(cap_tab)
        self._build_send_tab(send_tab)
        self._build_settings_tab(cfg_tab)

    # ── Capture tab ───────────────────────────────────────────────────────
    def _build_capture_tab(self, parent):
        self.cap = ProcessPanel()

        left  = make_frame(parent)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        right = make_frame(parent)
        right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

        section_label(left, "capture_it  —  settings").pack(fill="x", pady=(0,8))

        # Variables
        self.cap_outfile      = tk.StringVar(value="capture.bin")
        self.cap_max_pkts     = tk.StringVar(value="")
        self.cap_seconds      = tk.StringVar(value="")
        self.cap_wait_timeout = tk.StringVar(value="60")
        self.cap_socket_buf   = tk.StringVar(value="268435456")
        self.cap_ram_buf      = tk.StringVar(value="512")
        self.cap_depth        = tk.StringVar(value="16")
        self.cap_cpu          = tk.StringVar(value="3")
        self.cap_rt           = tk.BooleanVar(value=True)
        self.cap_verbose      = tk.BooleanVar(value=True)
        self.cap_use_sudo     = tk.BooleanVar(value=True)
        self.cap_vita49       = tk.BooleanVar(value=False)

        rows = [
            ("Output file",        self.cap_outfile,      True),
            ("Max packets (0=∞)",  self.cap_max_pkts,     False),
            ("Max duration (s)",   self.cap_seconds,      False),
            ("Wait timeout (s)",   self.cap_wait_timeout, False),
            ("Socket buffer (B)",  self.cap_socket_buf,   False),
            ("RAM buffer (MiB)",   self.cap_ram_buf,       False),
            ("Queue depth",        self.cap_depth,         False),
            ("CPU core pin",       self.cap_cpu,           False),
        ]
        for lbl, var, has_browse in rows:
            row = tk.Frame(left, bg=BG2)
            row.pack(fill="x", pady=2)
            label(row, lbl + ":", width=18, anchor="w").pack(side="left")
            entry(row, var).pack(side="left", padx=(4,2))
            if has_browse:
                button(row, "Browse", lambda v=var: self._browse_save(v),
                       color=ACCENT, width=6).pack(side="left", padx=2)

        opts = tk.Frame(left, bg=BG2)
        opts.pack(fill="x", pady=6)
        check(opts, "RT scheduling (SCHED_FIFO)", self.cap_rt).pack(anchor="w")
        check(opts, "Verbose output",             self.cap_verbose).pack(anchor="w")
        check(opts, "Use sudo",                   self.cap_use_sudo).pack(anchor="w")
        check(opts, "Preserve VITA-49.2 headers", self.cap_vita49).pack(anchor="w")

        # Buttons
        btns = tk.Frame(left, bg=BG2)
        btns.pack(fill="x", pady=8)
        self.cap_start_btn = button(btns, "▶  Start Capture",
                                    self._start_capture, BTN_START, width=16)
        self.cap_start_btn.pack(side="left", padx=(0,4))
        self.cap_stop_btn  = button(btns, "■  Stop",
                                    self._stop_capture, BTN_STOP, width=8)
        self.cap_stop_btn.pack(side="left")
        self.cap_stop_btn.config(state="disabled")

        # Stats strip
        self.cap_status = tk.StringVar(value="Idle")
        tk.Label(left, textvariable=self.cap_status,
                 bg=BG2, fg=FG2, font=FONT_LABEL, anchor="w").pack(fill="x")

        # Log
        section_label(right, "Live output").pack(fill="x", pady=(0,6))
        self.cap_log = log_box(right)
        self.cap_log.pack(fill="both", expand=True)

    # ── Send tab ──────────────────────────────────────────────────────────
    def _build_send_tab(self, parent):
        self.snd = ProcessPanel()

        left  = make_frame(parent)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        right = make_frame(parent)
        right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

        section_label(left, "send_it  —  settings").pack(fill="x", pady=(0,8))

        self.snd_file     = tk.StringVar(value="capture.bin")
        self.snd_fs       = tk.StringVar(value="10080433")
        self.snd_dest     = tk.StringVar(value="10.10.10.1")
        self.snd_port     = tk.StringVar(value="4991")
        self.snd_loops    = tk.StringVar(value="1")
        self.snd_channels = tk.StringVar(value="2")
        self.snd_bits     = tk.StringVar(value="16")
        self.snd_cpu      = tk.StringVar(value="3")
        self.snd_rt       = tk.BooleanVar(value=True)
        self.snd_verbose  = tk.BooleanVar(value=True)
        self.snd_use_sudo = tk.BooleanVar(value=True)
        self.snd_vita49   = tk.BooleanVar(value=False)

        rows = [
            ("Input file",     self.snd_file,     True),
            ("Sample rate (Hz)",self.snd_fs,      False),
            ("Destination IP", self.snd_dest,     False),
            ("Port",           self.snd_port,     False),
            ("Loops (0=∞)",    self.snd_loops,    False),
            ("Channels",       self.snd_channels, False),
            ("Bit depth",      self.snd_bits,     False),
            ("CPU core pin",   self.snd_cpu,      False),
        ]
        for lbl, var, has_browse in rows:
            row = tk.Frame(left, bg=BG2)
            row.pack(fill="x", pady=2)
            label(row, lbl + ":", width=18, anchor="w").pack(side="left")
            entry(row, var).pack(side="left", padx=(4,2))
            if has_browse:
                button(row, "Browse", lambda v=var: self._browse_open(v),
                       color=ACCENT, width=6).pack(side="left", padx=2)

        opts = tk.Frame(left, bg=BG2)
        opts.pack(fill="x", pady=6)
        check(opts, "Input has VITA-49.2 headers", self.snd_vita49).pack(anchor="w")
        check(opts, "RT scheduling (SCHED_FIFO)",  self.snd_rt).pack(anchor="w")
        check(opts, "Verbose output",              self.snd_verbose).pack(anchor="w")
        check(opts, "Use sudo",                    self.snd_use_sudo).pack(anchor="w")

        btns = tk.Frame(left, bg=BG2)
        btns.pack(fill="x", pady=8)
        self.snd_start_btn = button(btns, "▶  Start Transmit",
                                    self._start_send, BTN_START, width=16)
        self.snd_start_btn.pack(side="left", padx=(0,4))
        self.snd_stop_btn  = button(btns, "■  Stop",
                                    self._stop_send, BTN_STOP, width=8)
        self.snd_stop_btn.pack(side="left")
        self.snd_stop_btn.config(state="disabled")

        self.snd_status = tk.StringVar(value="Idle")
        tk.Label(left, textvariable=self.snd_status,
                 bg=BG2, fg=FG2, font=FONT_LABEL, anchor="w").pack(fill="x")

        section_label(right, "Live output").pack(fill="x", pady=(0,6))
        self.snd_log = log_box(right)
        self.snd_log.pack(fill="both", expand=True)

    # ── Settings tab ──────────────────────────────────────────────────────
    def _build_settings_tab(self, parent):
        f = make_frame(parent)
        f.pack(fill="both", expand=True, padx=12, pady=12)

        # ── Tool locations ────────────────────────────────────────────────
        section_label(f, "Tool locations").pack(fill="x", pady=(0,10))

        self.cfg_tools_dir = tk.StringVar(value=SCRIPT_DIR)

        row = tk.Frame(f, bg=BG2)
        row.pack(fill="x", pady=3)
        label(row, "Tools directory:", width=20, anchor="w").pack(side="left")
        entry(row, self.cfg_tools_dir, width=40).pack(side="left", padx=4)
        button(row, "Browse",
               lambda: self._browse_dir(self.cfg_tools_dir),
               ACCENT, 6).pack(side="left")

        # ── Sudo password ─────────────────────────────────────────────────
        section_label(f, "Sudo password").pack(fill="x", pady=(16,10))

        pw_row = tk.Frame(f, bg=BG2)
        pw_row.pack(fill="x", pady=3)
        label(pw_row, "Cached password:", width=20, anchor="w").pack(side="left")
        self.pw_status = tk.StringVar(value="None")
        tk.Label(pw_row, textvariable=self.pw_status,
                 bg=BG2, fg=FG2, font=FONT_LABEL).pack(side="left", padx=8)
        button(pw_row, "Enter password",
               self._enter_password, ACCENT, 14).pack(side="left", padx=4)
        button(pw_row, "Clear",
               self._clear_password, ACCENT2, 6).pack(side="left")

        # ── Moku network / ARP neighbour ──────────────────────────────────
        section_label(f, "Moku Network  —  ARP Neighbour").pack(fill="x", pady=(16,10))

        self.cfg_moku_mac   = tk.StringVar(value="70:69:79:b2:01:41")
        self.cfg_moku_ip    = tk.StringVar(value="10.10.10.1")
        self.cfg_moku_iface = tk.StringVar(value="enp5s0f0np0")

        net_rows = [
            ("Moku MAC address:", self.cfg_moku_mac,   28),
            ("Moku IP address:",  self.cfg_moku_ip,    28),
            ("Network interface:",self.cfg_moku_iface, 28),
        ]
        for lbl_text, var, w in net_rows:
            r = tk.Frame(f, bg=BG2)
            r.pack(fill="x", pady=2)
            label(r, lbl_text, width=20, anchor="w").pack(side="left")
            entry(r, var, width=w).pack(side="left", padx=4)

        btn_row = tk.Frame(f, bg=BG2)
        btn_row.pack(fill="x", pady=8)
        button(btn_row, "Add Neighbour",
               self._add_neighbour, BTN_START, width=16).pack(side="left", padx=(0,6))
        button(btn_row, "Verify Neighbour",
               self._verify_neighbour, ACCENT, width=16).pack(side="left")

        section_label(f, "Neighbour status").pack(fill="x", pady=(8,4))
        self.neigh_log = log_box(f)
        self.neigh_log.config(height=6)
        self.neigh_log.pack(fill="x")

        # ── About ─────────────────────────────────────────────────────────
        section_label(f, "About").pack(fill="x", pady=(16,10))
        about = (
            "Moku:Delta Gigabit Streamer GUI\n"
            "Front-end for capture_it and send_it\n\n"
            "Requires:  python3-tk  (sudo apt install python3-tk)\n"
            "           capture_it and send_it built with:  make all\n"
            "           Optional RT caps set with:  sudo make install-rt\n"
        )
        tk.Label(f, text=about, bg=BG2, fg=FG2, font=FONT_LABEL,
                 justify="left", padx=8, pady=8).pack(anchor="w")

    # ──────────────────────────────────────────────────────────────────────
    #  Browse helpers
    # ──────────────────────────────────────────────────────────────────────
    def _browse_save(self, var):
        p = filedialog.asksaveasfilename(
            parent=self.root, title="Choose output file",
            defaultextension=".bin",
            filetypes=[("Binary", "*.bin"), ("All", "*")])
        if p:
            var.set(p)

    def _browse_open(self, var):
        p = filedialog.askopenfilename(
            parent=self.root, title="Choose file",
            filetypes=[("Binary", "*.bin"), ("All", "*")])
        if p:
            var.set(p)

    def _browse_dir(self, var):
        p = filedialog.askdirectory(parent=self.root, title="Select tools directory")
        if p:
            var.set(p)

    # ──────────────────────────────────────────────────────────────────────
    #  Password management
    # ──────────────────────────────────────────────────────────────────────
    def _enter_password(self):
        pw = simpledialog.askstring(
            "sudo password", "Enter sudo password:", show="*",
            parent=self.root)
        if pw is not None:
            self.sudo_password = pw
            self.pw_status.set("Set  ✓")

    def _clear_password(self):
        self.sudo_password = None
        self.pw_status.set("None")

    def _get_password(self, use_sudo):
        if not use_sudo:
            return ""
        if not self.sudo_password:
            pw = simpledialog.askstring(
                "sudo password",
                "Real-time scheduling requires sudo.\n"
                "Enter your password (will be cached for this session):",
                show="*", parent=self.root)
            if pw is None:
                return None   # user cancelled
            self.sudo_password = pw
            self.pw_status.set("Set  ✓")
        return self.sudo_password

    # ──────────────────────────────────────────────────────────────────────
    #  Capture control
    # ──────────────────────────────────────────────────────────────────────
    def _start_capture(self):
        if self.cap.running():
            return

        tools = self.cfg_tools_dir.get()
        binary = os.path.join(tools, "capture_it")
        if not os.path.exists(binary):
            messagebox.showerror("Not found",
                f"capture_it not found in:\n{tools}\n\n"
                "Check the Tools directory in Settings, then run:  make all",
                parent=self.root)
            return

        use_sudo = self.cap_use_sudo.get()
        password = self._get_password(use_sudo)
        if password is None:
            return   # user cancelled password dialog

        args = {
            "--outfile":            self.cap_outfile.get(),
            "--max-packets":        self.cap_max_pkts.get() or None,
            "--seconds":            self.cap_seconds.get() or None,
            "--wait-timeout":       self.cap_wait_timeout.get() or None,
            "--socket-buffer":      self.cap_socket_buf.get() or None,
            "--ram-buffer":         self.cap_ram_buf.get() or None,
            "--write-queue-depth":  self.cap_depth.get() or None,
            "--cpu":                self.cap_cpu.get() or None,
            "--rt":                 self.cap_rt.get(),
            "--verbose":            self.cap_verbose.get(),
            "--vita49":             self.cap_vita49.get(),
        }
        cmd, stdin_data = build_command(binary, args, use_sudo, password)

        log_append(self.cap_log, "$ " + " ".join(cmd), "info")
        self.cap.start(cmd, stdin_data)

        self.cap_start_btn.config(state="disabled")
        self.cap_stop_btn.config(state="normal")
        self.cap_status.set("● Running")

    def _stop_capture(self):
        self.cap.stop()

    def _poll_capture(self):
        try:
            while True:
                line = self.cap.out_queue.get_nowait()
                if line is None:   # process finished
                    self.cap_start_btn.config(state="normal")
                    self.cap_stop_btn.config(state="disabled")
                    rc = self.cap.proc.returncode if self.cap.proc else -1
                    status = "✓ Finished (PASS)" if rc == 0 else f"✗ Finished (rc={rc})"
                    self.cap_status.set(status)
                    log_append(self.cap_log, f"--- Process exited: {status} ---",
                               "good" if rc == 0 else "error")
                else:
                    log_append(self.cap_log, line, classify_line(line))
                    # update status strip with the most recent stat line
                    if "pkts" in line.lower() or "mb/s" in line.lower():
                        self.cap_status.set("● " + line.strip()[:80])
        except queue.Empty:
            pass
        self.root.after(80, self._poll_capture)

    # ──────────────────────────────────────────────────────────────────────
    #  Send control
    # ──────────────────────────────────────────────────────────────────────
    def _start_send(self):
        if self.snd.running():
            return

        tools = self.cfg_tools_dir.get()
        binary = os.path.join(tools, "send_it")
        if not os.path.exists(binary):
            messagebox.showerror("Not found",
                f"send_it not found in:\n{tools}\n\nRun:  make all",
                parent=self.root)
            return

        use_sudo = self.snd_use_sudo.get()
        password = self._get_password(use_sudo)
        if password is None:
            return

        args = {
            "--file":     self.snd_file.get(),
            "--fs":       self.snd_fs.get(),
            "--dest":     self.snd_dest.get(),
            "--port":     self.snd_port.get(),
            "--loops":    self.snd_loops.get() or None,
            "--channels": self.snd_channels.get() or None,
            "--bits":     self.snd_bits.get() or None,
            "--cpu":      self.snd_cpu.get() or None,
            "--vita49":   self.snd_vita49.get(),
            "--rt":       self.snd_rt.get(),
            "--verbose":  self.snd_verbose.get(),
        }
        cmd, stdin_data = build_command(binary, args, use_sudo, password)

        log_append(self.snd_log, "$ " + " ".join(cmd), "info")
        self.snd.start(cmd, stdin_data)

        self.snd_start_btn.config(state="disabled")
        self.snd_stop_btn.config(state="normal")
        self.snd_status.set("● Running")

    def _stop_send(self):
        self.snd.stop()

    def _poll_send(self):
        try:
            while True:
                line = self.snd.out_queue.get_nowait()
                if line is None:
                    self.snd_start_btn.config(state="normal")
                    self.snd_stop_btn.config(state="disabled")
                    rc = self.snd.proc.returncode if self.snd.proc else -1
                    status = "✓ Done" if rc == 0 else f"✗ Exited (rc={rc})"
                    self.snd_status.set(status)
                    log_append(self.snd_log, f"--- Process exited: {status} ---",
                               "good" if rc == 0 else "error")
                else:
                    log_append(self.snd_log, line, classify_line(line))
                    if "loop" in line.lower() or "pkts" in line.lower():
                        self.snd_status.set("● " + line.strip()[:80])
        except queue.Empty:
            pass
        self.root.after(80, self._poll_send)

    # ──────────────────────────────────────────────────────────────────────
    #  Validate control
    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    #  Moku neighbour helpers
    # ──────────────────────────────────────────────────────────────────────
    def _run_neigh_cmd(self, cmd, stdin_data=None):
        """Run a neighbour command, stream output to neigh_log."""
        log_append(self.neigh_log, "$ " + " ".join(cmd), "info")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if stdin_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            input_str = stdin_data.decode() if stdin_data else None
            out, _ = proc.communicate(input=input_str, timeout=10)
            for line in out.splitlines():
                tag = "error" if proc.returncode != 0 else classify_line(line)
                log_append(self.neigh_log, line, tag)
            if proc.returncode == 0:
                log_append(self.neigh_log, "✓  Done", "good")
            else:
                log_append(self.neigh_log, f"✗  Exited rc={proc.returncode}", "error")
        except subprocess.TimeoutExpired:
            log_append(self.neigh_log, "ERROR: command timed out", "error")
        except Exception as e:
            log_append(self.neigh_log, f"ERROR: {e}", "error")

    def _add_neighbour(self):
        ip    = self.cfg_moku_ip.get().strip()
        mac   = self.cfg_moku_mac.get().strip()
        iface = self.cfg_moku_iface.get().strip()
        if not ip or not mac or not iface:
            messagebox.showerror("Missing values",
                "Please fill in Moku IP, MAC, and interface.", parent=self.root)
            return
        password = self._get_password(True)
        if password is None:
            return
        cmd = ["sudo", "-S", "-k", "ip", "neigh", "replace",
               ip, "lladdr", mac, "dev", iface]
        self._run_neigh_cmd(cmd, stdin_data=(password + "\n").encode())

    def _verify_neighbour(self):
        iface = self.cfg_moku_iface.get().strip()
        if not iface:
            messagebox.showerror("Missing value",
                "Please enter the network interface.", parent=self.root)
            return
        self._run_neigh_cmd(["ip", "neigh", "show", "dev", iface])

    # ──────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def _on_close(self):
        if self.cap.running() or self.snd.running():
            if not messagebox.askyesno(
                    "Quit", "A process is still running. Stop it and quit?",
                    parent=self.root):
                return
        self.cap.stop()
        self.snd.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MokuGUI().run()
