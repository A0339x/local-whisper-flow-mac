#!/usr/bin/env python3
"""
Voice-To-Text — a local, free, Wispr-Flow-style dictation app.

Pipeline:  toggle hotkey ▸ record mic ▸ Whisper (MLX) transcribe ▸
           Ollama smart-format/correct ▸ paste into focused app.

A floating "recording" HUD (waveform pill with ✕ / ✓) appears while you talk.
Everything runs locally on Apple Silicon. No cloud, no subscription.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import threading
import time
import tomllib
from collections import deque
from pathlib import Path

import numpy as np
import requests
import rumps
import sounddevice as sd
from pynput import keyboard, mouse

# AppKit for the floating recording HUD (pulled in by rumps/pyobjc).
import objc
from AppKit import (
    NSBezierPath,
    NSColor,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSTimer,
    NSView,
    NSWindow,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSBackingStoreBuffered,
    NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSLineCapStyleRound,
    NSApplication,
    NSPopUpButton,
    NSButton,
    NSButtonTypeSwitch,
    NSTextField,
    NSScrollView,
    NSTableView,
    NSTableColumn,
    NSFont,
    NSProgressIndicator,
)
from Foundation import NSObject
from PyObjCTools import AppHelper

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ── Dictation history ────────────────────────────────────────────────────────

HISTORY_PATH = CONFIG_PATH.parent / "history.jsonl"
HISTORY_KEEP = 500  # rows kept on disk
ONBOARDED_PATH = CONFIG_PATH.parent / ".onboarded"


def history_append(text: str) -> None:
    if not text.strip():
        return
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text}
    try:
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log(f"  history write failed: {e}")


def history_load(limit: int = 300) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    out: list[dict] = []
    try:
        lines = HISTORY_PATH.read_text().splitlines()[-limit:]
    except Exception:
        return []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    out.reverse()  # newest first
    return out


def history_clear() -> None:
    try:
        HISTORY_PATH.unlink()
    except FileNotFoundError:
        pass


# ── States / UI glyphs ───────────────────────────────────────────────────────

IDLE = "idle"
RECORDING = "recording"
PROCESSING = "processing"

GLYPH = {IDLE: "🎤", RECORDING: "🔴", PROCESSING: "⏳"}

KEY_LABELS = {
    "alt_r": "Right Option",
    "alt_l": "Left Option",
    "cmd_r": "Right Command",
    "cmd_l": "Left Command",
    "ctrl_r": "Right Control",
    "ctrl_l": "Left Control",
}

SAMPLE_RATE = 16_000

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_CANCEL = "/System/Library/Sounds/Bottle.aiff"
SOUND_ERROR = "/System/Library/Sounds/Basso.aiff"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def play(sound_path: str) -> None:
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ── Audio capture ──────────────────────────────────────────────────────────────

def resolve_input_device(spec):  # noqa: ANN001
    """Map a config value to a sounddevice device index (or None = default)."""
    if spec in (None, "", "default"):
        return None
    if isinstance(spec, int):
        return spec
    devices = sd.query_devices()
    if spec == "builtin":
        patterns = ("macbook", "built-in", "built in", "imac", "mac mini", "mac studio")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and any(p in d["name"].lower() for p in patterns):
                return i
        return None  # fall back to default input
    for i, d in enumerate(devices):  # treat as a name substring
        if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower():
            return i
    return None


class AudioRecorder:
    """Captures mono float32 audio at 16 kHz; exposes a live input level.

    When ``warm`` is True the input stream stays open continuously and a small
    pre-roll ring buffer is kept, so pressing the hotkey captures instantly and
    never clips the start of speech.
    """

    def __init__(self, device=None, preroll_seconds: float = 0.5, warm: bool = True) -> None:  # noqa: ANN001
        self._device = device
        self._warm = warm
        self._preroll_max = int(SAMPLE_RATE * max(0.0, preroll_seconds))
        self._ring: deque[np.ndarray] = deque()
        self._ring_len = 0
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.level: float = 0.0  # 0..1, smoothed mic loudness for the HUD
        if warm:
            self._open_stream()

    def _open_stream(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        chunk = indata.copy().reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-9))
        inst = min(1.0, rms * 14.0)
        self.level = max(inst, self.level * 0.82)
        with self._lock:
            if self._recording:
                self._frames.append(chunk)
            elif self._preroll_max > 0:
                self._ring.append(chunk)
                self._ring_len += chunk.shape[0]
                while self._ring_len > self._preroll_max and len(self._ring) > 1:
                    self._ring_len -= self._ring.popleft().shape[0]

    def start(self) -> None:
        with self._lock:
            # Seed with the pre-roll so the moment before the press isn't lost.
            self._frames = list(self._ring) if self._warm else []
            self._recording = True
        if not self._warm:
            self._open_stream()

    def stop(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            frames = self._frames
            self._frames = []
            self._ring.clear()
            self._ring_len = 0
        self.level = 0.0
        if not self._warm and self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(frames, axis=0).astype("float32")

    def set_device(self, device) -> None:  # noqa: ANN001
        """Switch the input device live. Call only while idle."""
        with self._lock:
            self._device = device
            self._ring.clear()
            self._ring_len = 0
            self._frames = []
            self._recording = False
        if self._warm:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._open_stream()

    def set_warm(self, on: bool) -> None:
        if on == self._warm:
            return
        self._warm = on
        if on:
            if self._stream is None:
                self._open_stream()
        else:
            with self._lock:
                recording = self._recording
            if not recording and self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None


# ── Recording HUD (floating waveform pill) ───────────────────────────────────

PILL_W, PILL_H = 138.0, 34.0
BTN_R = 11.0          # button radius
BTN_INSET = 18.0      # button center inset from each end
N_BARS = 11


class PillView(NSView):
    """Custom-drawn capsule: ✕ button · live waveform · ✓ button."""

    def initWithFrame_(self, frame):  # noqa: N802
        self = objc.super(PillView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._level_provider = lambda: 0.0
        self._on_cancel = lambda: None
        self._on_confirm = lambda: None
        self._t0 = 0.0
        # Stable per-bar phase offsets so the waveform looks organic.
        self._phase = [random.uniform(0, math.tau) for _ in range(N_BARS)]
        return self

    # configuration from Python
    def configure_(self, info):  # passed a dict
        self._level_provider = info["level"]
        self._on_cancel = info["cancel"]
        self._on_confirm = info["confirm"]

    def animate(self):  # NSTimer target
        self.setNeedsDisplay_(True)

    def isFlipped(self):  # noqa: N802
        return False

    def _left_center(self):
        return (BTN_INSET, PILL_H / 2.0)

    def _right_center(self):
        return (PILL_W - BTN_INSET, PILL_H / 2.0)

    def drawRect_(self, rect):  # noqa: N802
        b = self.bounds()

        # Capsule background.
        capsule = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            b, PILL_H / 2.0, PILL_H / 2.0
        )
        NSColor.colorWithCalibratedWhite_alpha_(0.11, 0.97).set()
        capsule.fill()
        NSColor.colorWithCalibratedWhite_alpha_(0.30, 1.0).set()
        capsule.setLineWidth_(1.0)
        capsule.stroke()

        # Waveform bars.
        try:
            level = float(self._level_provider())
        except Exception:
            level = 0.0
        s = BTN_R / 18.0  # glyph scale relative to the original size
        t = time.time()
        area_x0 = BTN_INSET + BTN_R + 6.0
        area_x1 = PILL_W - BTN_INSET - BTN_R - 6.0
        area_w = area_x1 - area_x0
        bar_w = 2.0
        gap = (area_w - N_BARS * bar_w) / (N_BARS - 1)
        cy = PILL_H / 2.0
        max_h = PILL_H * 0.5
        NSColor.whiteColor().set()
        for i in range(N_BARS):
            wobble = 0.5 + 0.5 * math.sin(t * 6.0 + self._phase[i])
            amp = (0.18 + 0.82 * level) * wobble
            h = max(3.0, amp * max_h)
            x = area_x0 + i * (bar_w + gap)
            bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, cy - h / 2.0, bar_w, h), bar_w / 2.0, bar_w / 2.0
            )
            bar.fill()

        # Left button: gray circle + white ✕.
        lx, ly = self._left_center()
        NSColor.colorWithCalibratedWhite_alpha_(0.42, 1.0).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(lx - BTN_R, ly - BTN_R, 2 * BTN_R, 2 * BTN_R)
        ).fill()
        d = 5.5 * s
        x_path = NSBezierPath.bezierPath()
        x_path.moveToPoint_((lx - d, ly - d))
        x_path.lineToPoint_((lx + d, ly + d))
        x_path.moveToPoint_((lx - d, ly + d))
        x_path.lineToPoint_((lx + d, ly - d))
        x_path.setLineWidth_(2.0 * s)
        x_path.setLineCapStyle_(NSLineCapStyleRound)
        NSColor.whiteColor().set()
        x_path.stroke()

        # Right button: white circle + dark ✓.
        rx, ry = self._right_center()
        NSColor.whiteColor().set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(rx - BTN_R, ry - BTN_R, 2 * BTN_R, 2 * BTN_R)
        ).fill()
        chk = NSBezierPath.bezierPath()
        chk.moveToPoint_((rx - 6.0 * s, ry - 0.5 * s))
        chk.lineToPoint_((rx - 1.5 * s, ry - 5.0 * s))
        chk.lineToPoint_((rx + 6.5 * s, ry + 5.5 * s))
        chk.setLineWidth_(2.4 * s)
        chk.setLineCapStyle_(NSLineCapStyleRound)
        NSColor.colorWithCalibratedWhite_alpha_(0.11, 1.0).set()
        chk.stroke()

    def mouseDown_(self, event):  # noqa: N802
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        lx, ly = self._left_center()
        rx, ry = self._right_center()
        if math.hypot(p.x - lx, p.y - ly) <= BTN_R + 4:
            self._on_cancel()
        elif math.hypot(p.x - rx, p.y - ry) <= BTN_R + 4:
            self._on_confirm()


class PillPanel(NSPanel):
    def canBecomeKeyWindow(self):  # noqa: N802
        return True  # receive clicks…

    def canBecomeMainWindow(self):  # noqa: N802
        return False  # …but never steal the active app


class RecordingHUD:
    """Owns the floating panel. All methods must run on the main thread."""

    def __init__(self, level_provider, on_cancel, on_confirm) -> None:
        self._info = {
            "level": level_provider,
            "cancel": on_cancel,
            "confirm": on_confirm,
        }
        self._panel: PillPanel | None = None
        self._view: PillView | None = None
        self._timer: NSTimer | None = None

    def _build(self) -> None:
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = PillPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PILL_W, PILL_H), style, NSBackingStoreBuffered, False
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setFloatingPanel_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
        )
        view = PillView.alloc().initWithFrame_(
            NSMakeRect(0, 0, PILL_W, PILL_H)
        )
        view.configure_(self._info)
        panel.setContentView_(view)
        self._panel, self._view = panel, view

    def _reposition(self) -> None:
        scr = NSScreen.mainScreen().frame()
        x = scr.origin.x + (scr.size.width - PILL_W) / 2.0
        y = scr.origin.y + 130.0
        self._panel.setFrameOrigin_((x, y))

    def show(self) -> None:
        if self._panel is None:
            self._build()
        self._reposition()
        self._panel.orderFrontRegardless()
        if self._timer is None:
            self._timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    1.0 / 30.0, self._view, "animate", None, True
                )
            )

    def hide(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        if self._panel is not None:
            self._panel.orderOut_(None)


# ── Settings window ──────────────────────────────────────────────────────────

class SettingsController(NSObject):
    """A small titled window with a microphone picker + warm-mic toggle."""

    def initWithApp_(self, app):  # noqa: N802
        self = objc.super(SettingsController, self).init()
        if self is None:
            return None
        self._app = app
        self._window = None
        self._popup = None
        self._warm_btn = None
        self._specs = []
        return self

    def _build(self) -> None:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
        )
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 400, 246), style, NSBackingStoreBuffered, False
        )
        win.setTitle_("Voice To Text — Settings")
        win.setReleasedWhenClosed_(False)
        # Normal level + auto-hide when you switch apps, so it never floats on
        # top of or gets in the way of whatever you're working in.
        win.setLevel_(0)
        win.setHidesOnDeactivate_(True)
        cv = win.contentView()

        def label(text, frame, secondary=False):
            f = NSTextField.alloc().initWithFrame_(frame)
            f.setStringValue_(text)
            f.setBezeled_(False)
            f.setDrawsBackground_(False)
            f.setEditable_(False)
            f.setSelectable_(False)
            if secondary:
                f.setTextColor_(NSColor.secondaryLabelColor())
            cv.addSubview_(f)
            return f

        label("Microphone:", NSMakeRect(20, 198, 360, 18))
        popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(20, 166, 360, 28), False
        )
        popup.setTarget_(self)
        popup.setAction_("micChanged:")
        cv.addSubview_(popup)
        self._popup = popup

        warm = NSButton.alloc().initWithFrame_(NSMakeRect(20, 130, 360, 22))
        warm.setButtonType_(NSButtonTypeSwitch)
        warm.setTitle_("Keep mic warm (instant capture; orange mic dot stays on)")
        warm.setTarget_(self)
        warm.setAction_("warmToggled:")
        cv.addSubview_(warm)
        self._warm_btn = warm

        hist = NSButton.alloc().initWithFrame_(NSMakeRect(20, 86, 200, 30))
        hist.setTitle_("Dictation History…")
        hist.setBezelStyle_(1)  # rounded
        hist.setTarget_(self)
        hist.setAction_("openHistory:")
        cv.addSubview_(hist)

        label(
            "Switching to a Bluetooth mic drops it to call-quality while warm.\n"
            "Open this window anytime with ⌃⌥⌘M.",
            NSMakeRect(20, 18, 360, 44),
            secondary=True,
        )
        self._window = win

    def openHistory_(self, sender):  # noqa: N802
        self._app.open_history()

    def _refresh(self) -> None:
        self._popup.removeAllItems()
        self._specs = []
        items = [("System Default", "default"), ("Built-in (Mac mic)", "builtin")]
        for d in sd.query_devices():
            if d["max_input_channels"] > 0:
                items.append((d["name"], d["name"]))
        current = str(self._app.cfg["audio"].get("input_device", "builtin"))
        selected = 0
        for i, (lbl, spec) in enumerate(items):
            self._popup.addItemWithTitle_(lbl)
            self._specs.append(spec)
            if spec == current:
                selected = i
        self._popup.selectItemAtIndex_(selected)
        self._warm_btn.setState_(
            1 if self._app.cfg["audio"].get("warm_mic", True) else 0
        )

    def show(self) -> None:
        if self._window is None:
            self._build()
        self._refresh()
        if self._window.isMiniaturized():
            self._window.deminiaturize_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._window.center()
        self._window.makeKeyAndOrderFront_(None)
        # Force the window visible even when the app is in a background state
        # (e.g. started by launchd at login), otherwise it opens behind others.
        self._window.orderFrontRegardless()

    def micChanged_(self, sender):  # noqa: N802
        idx = sender.indexOfSelectedItem()
        if 0 <= idx < len(self._specs):
            self._app.apply_mic(self._specs[idx])

    def warmToggled_(self, sender):  # noqa: N802
        self._app.apply_warm(bool(sender.state()))


class HistoryController(NSObject):
    """A scrollable list of past dictations; click a row to re-copy it."""

    def initWithApp_(self, app):  # noqa: N802
        self = objc.super(HistoryController, self).init()
        if self is None:
            return None
        self._app = app
        self._window = None
        self._table = None
        self._entries = []
        return self

    def _build(self) -> None:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
        )
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 540, 460), style, NSBackingStoreBuffered, False
        )
        win.setTitle_("Dictation History")
        win.setReleasedWhenClosed_(False)
        win.setLevel_(0)
        win.setHidesOnDeactivate_(True)
        cv = win.contentView()

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, 60, 508, 384))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(2)  # NSBezelBorder

        table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, 508, 384))
        col = NSTableColumn.alloc().initWithIdentifier_("entry")
        col.setWidth_(490)
        table.addTableColumn_(col)
        table.setHeaderView_(None)
        table.setRowHeight_(20)
        table.setUsesAlternatingRowBackgroundColors_(True)
        table.setDataSource_(self)
        table.setTarget_(self)
        table.setDoubleAction_("copySelected:")
        scroll.setDocumentView_(table)
        cv.addSubview_(scroll)
        self._table = table

        copy_btn = NSButton.alloc().initWithFrame_(NSMakeRect(16, 16, 150, 30))
        copy_btn.setTitle_("Copy selected")
        copy_btn.setBezelStyle_(1)
        copy_btn.setTarget_(self)
        copy_btn.setAction_("copySelected:")
        cv.addSubview_(copy_btn)

        clear_btn = NSButton.alloc().initWithFrame_(NSMakeRect(174, 16, 110, 30))
        clear_btn.setTitle_("Clear")
        clear_btn.setBezelStyle_(1)
        clear_btn.setTarget_(self)
        clear_btn.setAction_("clearAll:")
        cv.addSubview_(clear_btn)

        hint = NSTextField.alloc().initWithFrame_(NSMakeRect(300, 20, 224, 18))
        hint.setStringValue_("Double-click a row to copy it")
        hint.setBezeled_(False)
        hint.setDrawsBackground_(False)
        hint.setEditable_(False)
        hint.setSelectable_(False)
        hint.setTextColor_(NSColor.secondaryLabelColor())
        cv.addSubview_(hint)

        self._window = win

    def _reload(self) -> None:
        self._entries = history_load()
        if self._table is not None:
            self._table.reloadData()

    def show(self) -> None:
        if self._window is None:
            self._build()
        self._reload()
        if self._window.isMiniaturized():
            self._window.deminiaturize_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._window.center()
        self._window.makeKeyAndOrderFront_(None)
        # Force the window visible even when the app is in a background state
        # (e.g. started by launchd at login), otherwise it opens behind others.
        self._window.orderFrontRegardless()

    # NSTableView data source
    def numberOfRowsInTableView_(self, tv):  # noqa: N802
        return len(self._entries)

    def tableView_objectValueForTableColumn_row_(self, tv, col, row):  # noqa: N802
        e = self._entries[row]
        text = (e.get("text", "") or "").replace("\n", " ⏎ ")
        return f"{e.get('ts', '')[5:]}   {text}"  # drop the year

    def copySelected_(self, sender):  # noqa: N802
        r = self._table.selectedRow()
        if 0 <= r < len(self._entries):
            txt = self._entries[r].get("text", "")
            clipboard_set(txt)
            rumps.notification("Voice-To-Text", "Copied to clipboard", txt[:60])

    def clearAll_(self, sender):  # noqa: N802
        history_clear()
        self._reload()


# ── Onboarding ───────────────────────────────────────────────────────────────

OB_W, OB_H = 580.0, 480.0
OB_STEPS = ["welcome", "permissions", "calibrate_normal", "calibrate_excited", "download", "done"]
CALIB_SENTENCE = "“The quick brown fox jumps over the lazy dog.”"


class OnboardingController(NSObject):
    """A first-run wizard: explains permissions + warm mic, and calibrates voice."""

    def initWithApp_(self, app):  # noqa: N802
        self = objc.super(OnboardingController, self).init()
        if self is None:
            return None
        self._app = app
        self._window = None
        self._step = 0
        self._normal_feat = None
        self._excited_feat = None
        self._status_label = None
        self._next_btn = None
        self._progress = None
        self._dl_status = None
        self._dl_btn = None
        return self

    # ── infra ──
    def show(self) -> None:
        if self._window is None:
            style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, OB_W, OB_H), style, NSBackingStoreBuffered, False
            )
            win.setTitle_("Voice To Text — Setup")
            win.setReleasedWhenClosed_(False)
            win.setLevel_(0)
            self._window = win
        self._step = 0
        self._render()
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._window.center()
        self._window.makeKeyAndOrderFront_(None)
        self._window.orderFrontRegardless()

    @objc.python_method
    def _label(self, parent, text, frame, size=13, bold=False, secondary=False):
        f = NSTextField.alloc().initWithFrame_(frame)
        f.setStringValue_(text)
        f.setBezeled_(False)
        f.setDrawsBackground_(False)
        f.setEditable_(False)
        f.setSelectable_(False)
        f.setUsesSingleLineMode_(False)
        f.cell().setWraps_(True)
        f.setFont_(
            NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
        )
        if secondary:
            f.setTextColor_(NSColor.secondaryLabelColor())
        parent.addSubview_(f)
        return f

    @objc.python_method
    def _button(self, parent, title, frame, action):
        b = NSButton.alloc().initWithFrame_(frame)
        b.setTitle_(title)
        b.setBezelStyle_(1)  # rounded
        b.setTarget_(self)
        b.setAction_(action)
        parent.addSubview_(b)
        return b

    @objc.python_method
    def _render(self) -> None:
        cv = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, OB_W, OB_H))
        step = OB_STEPS[self._step]
        getattr(self, f"_step_{step}")(cv)

        # Footer navigation.
        if self._step > 0:
            self._button(cv, "Back", NSMakeRect(20, 20, 90, 32), "back:")
        last = self._step == len(OB_STEPS) - 1
        self._next_btn = self._button(
            cv,
            "Finish" if last else "Next",
            NSMakeRect(OB_W - 130, 20, 110, 32),
            "finish:" if last else "next:",
        )
        self._label(
            cv,
            f"Step {self._step + 1} of {len(OB_STEPS)}",
            NSMakeRect(OB_W / 2 - 60, 26, 120, 18),
            size=11,
            secondary=True,
        ).setAlignment_(2)  # center
        self._window.setContentView_(cv)

    # ── navigation ──
    def next_(self, sender):  # noqa: N802
        if self._step < len(OB_STEPS) - 1:
            self._step += 1
            self._render()

    def back_(self, sender):  # noqa: N802
        if self._step > 0:
            self._step -= 1
            self._render()

    def finish_(self, sender):  # noqa: N802
        self._apply_calibration()
        try:
            ONBOARDED_PATH.write_text("done\n")
        except Exception:
            pass
        self._window.orderOut_(None)

    # ── steps ──
    @objc.python_method
    def _step_welcome(self, cv):
        self._label(cv, "Welcome to Voice To Text", NSMakeRect(40, OB_H - 70, OB_W - 80, 30), size=20, bold=True)
        body = (
            "Free, on-device dictation — your voice never leaves this Mac.\n\n"
            "HOW IT WORKS\n"
            "•  Tap the Right Option (⌥) key, start talking, then tap it again to "
            "stop. Your words are cleaned up and pasted wherever your cursor is.\n"
            "•  A floating waveform pill appears while recording — press ✓ to "
            "finish or ✕ to cancel.\n"
            "•  It removes filler words, fixes punctuation, applies “no wait, I "
            "mean…” corrections, and adds “!” when you sound excited.\n\n"
            "ABOUT THE “WARM MIC”\n"
            "To capture instantly with no clipped words, the app keeps your "
            "built-in Mac microphone active the whole time it runs. That’s why "
            "you’ll see the orange mic dot in the menu bar — it’s expected and "
            "normal. Your headphones/AirPods are never used for input, so their "
            "audio quality stays perfect. You can change the mic or turn this off "
            "anytime in Settings (click the app icon)."
        )
        self._label(cv, body, NSMakeRect(40, 70, OB_W - 80, OB_H - 150))

    @objc.python_method
    def _step_permissions(self, cv):
        self._label(cv, "Permissions", NSMakeRect(40, OB_H - 70, OB_W - 80, 30), size=20, bold=True)
        self._label(
            cv,
            "Dictation needs three one-time macOS permissions. Click each button "
            "to open the right pane, then enable “Python” (or this app):",
            NSMakeRect(40, OB_H - 130, OB_W - 80, 48),
        )
        self._button(cv, "Open Microphone settings", NSMakeRect(40, OB_H - 180, 280, 32), "openMic:")
        self._label(cv, "Hear your voice.", NSMakeRect(330, OB_H - 176, 220, 22), secondary=True)
        self._button(cv, "Open Accessibility settings", NSMakeRect(40, OB_H - 222, 280, 32), "openAcc:")
        self._label(cv, "Paste with ⌘V.", NSMakeRect(330, OB_H - 218, 220, 22), secondary=True)
        self._button(cv, "Open Input Monitoring settings", NSMakeRect(40, OB_H - 264, 280, 32), "openInput:")
        self._label(cv, "Detect the Right Option key.", NSMakeRect(330, OB_H - 260, 220, 22), secondary=True)
        granted = False
        try:
            import HIServices

            granted = bool(HIServices.AXIsProcessTrusted())
        except Exception:
            pass
        self._label(
            cv,
            ("Accessibility is currently: " + ("✓ granted" if granted else "✗ not yet granted")
             + ".  After enabling permissions you may need to quit and relaunch the app."),
            NSMakeRect(40, 70, OB_W - 80, 40),
            secondary=True,
        )

    @objc.python_method
    def _calib_step(self, cv, title, instruction, action):
        self._label(cv, title, NSMakeRect(40, OB_H - 70, OB_W - 80, 30), size=20, bold=True)
        self._label(cv, instruction, NSMakeRect(40, OB_H - 140, OB_W - 80, 56))
        self._label(cv, CALIB_SENTENCE, NSMakeRect(40, OB_H - 196, OB_W - 80, 26), size=15, bold=True)
        self._button(cv, "● Record (3s)", NSMakeRect(40, OB_H - 250, 180, 34), action)
        self._status_label = self._label(
            cv, "Click Record, then read the sentence aloud.",
            NSMakeRect(40, 80, OB_W - 80, 60), secondary=True,
        )

    @objc.python_method
    def _step_calibrate_normal(self, cv):
        self._calib_step(
            cv, "Calibrate — your normal voice",
            "Let’s learn your normal speaking level so we can tell when you’re "
            "excited. Read this in your NORMAL, relaxed voice:",
            "recordNormal:",
        )
        if self._normal_feat:
            self._status_label.setStringValue_(
                f"✓ Captured your normal voice (loudness {self._normal_feat['rms']:.3f})."
            )

    @objc.python_method
    def _step_calibrate_excited(self, cv):
        self._calib_step(
            cv, "Calibrate — your excited voice",
            "Now read it again, but sound EXCITED — louder and more energetic, "
            "like you just got great news:",
            "recordExcited:",
        )
        if self._excited_feat:
            self._status_label.setStringValue_(
                f"✓ Captured your excited voice (loudness {self._excited_feat['rms']:.3f})."
            )

    @objc.python_method
    def _step_download(self, cv):
        self._label(cv, "Download the AI models", NSMakeRect(40, OB_H - 70, OB_W - 80, 30), size=20, bold=True)
        self._label(
            cv,
            "Optional but recommended: fetch the speech model (~3 GB) and the "
            "formatting model now, so your very first dictation is instant. "
            "Otherwise they download automatically the first time you use them.",
            NSMakeRect(40, OB_H - 150, OB_W - 80, 64),
        )
        self._dl_btn = self._button(cv, "Download models", NSMakeRect(40, OB_H - 206, 200, 34), "downloadModels:")
        self._progress = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(40, OB_H - 246, OB_W - 80, 18))
        self._progress.setIndeterminate_(False)
        self._progress.setMinValue_(0.0)
        self._progress.setMaxValue_(100.0)
        self._progress.setDoubleValue_(0.0)
        cv.addSubview_(self._progress)
        self._dl_status = self._label(
            cv, "You can also skip this and download later on first use.",
            NSMakeRect(40, 88, OB_W - 80, 60), secondary=True,
        )

    @objc.python_method
    def _step_done(self, cv):
        self._label(cv, "You’re all set!  🎤", NSMakeRect(40, OB_H - 80, OB_W - 80, 34), size=22, bold=True)
        sens = self._app.cfg.get("tone", {}).get("excitement_sensitivity", 1.35)
        tuned = ""
        if self._normal_feat and self._excited_feat:
            tuned = "We tuned excitement detection to your voice. "
        body = (
            "Tap Right Option (⌥) anytime to dictate — talk, then tap again to "
            "paste.\n\n"
            f"{tuned}Open Settings (microphone, history, options) by clicking the "
            "app icon in your Dock.\n\n"
            "If you haven’t granted the permissions yet, do that now (Back), then "
            "quit and relaunch the app.\n\n"
            "Tip: the first dictation downloads the speech model (~3 GB) once — "
            "give it a minute that first time."
        )
        self._label(cv, body, NSMakeRect(40, 80, OB_W - 80, OB_H - 190))

    # ── actions ──
    def openMic_(self, sender):  # noqa: N802
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])

    def openAcc_(self, sender):  # noqa: N802
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])

    def openInput_(self, sender):  # noqa: N802
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"])

    def recordNormal_(self, sender):  # noqa: N802
        self._record("normal", sender)

    def recordExcited_(self, sender):  # noqa: N802
        self._record("excited", sender)

    @objc.python_method
    def _record(self, which, btn):
        btn.setEnabled_(False)
        btn.setTitle_("● Recording… (3s)")
        if self._status_label:
            self._status_label.setStringValue_("Listening… read the sentence now.")

        def work():
            try:
                dev = resolve_input_device(
                    self._app.cfg.get("audio", {}).get("input_device", "builtin")
                )
                rec = sd.rec(int(3.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                             channels=1, dtype="float32", device=dev)
                sd.wait()
                feat = analyze_prosody(rec.reshape(-1))
            except Exception as e:
                feat = None
                log(f"onboarding capture error: {e}")
            AppHelper.callAfter(self._record_done, which, btn, feat)

        threading.Thread(target=work, daemon=True).start()

    @objc.python_method
    def _record_done(self, which, btn, feat):
        if which == "normal":
            self._normal_feat = feat
        else:
            self._excited_feat = feat
        btn.setEnabled_(True)
        btn.setTitle_("Re-record (3s)")
        if self._status_label:
            if feat:
                self._status_label.setStringValue_(
                    f"✓ Captured (loudness {feat['rms']:.3f}, pitch variation "
                    f"{feat['f0_std']:.1f}). Click Next, or Re-record."
                )
            else:
                self._status_label.setStringValue_(
                    "Couldn’t capture — check Microphone permission and try again."
                )

    @objc.python_method
    def _apply_calibration(self):
        n, e = self._normal_feat, self._excited_feat
        if n:
            self._app._tone_baseline = {"rms": n["rms"], "f0_std": n["f0_std"], "count": 5}
            self._app._save_tone_baseline()
            log(f"onboarding: baseline set rms={n['rms']:.3f} f0std={n['f0_std']:.2f}")
        if n and e:
            ratios = []
            if n["rms"] > 0:
                ratios.append(e["rms"] / n["rms"])
            if n["f0_std"] > 0:
                ratios.append(e["f0_std"] / n["f0_std"])
            if ratios:
                sens = round(max(1.2, min(2.2, 1 + 0.45 * (max(ratios) - 1))), 2)
                self._app.cfg.setdefault("tone", {})["excitement_sensitivity"] = sens
                self._app._persist("excitement_sensitivity", sens)
                log(f"onboarding: tuned excitement_sensitivity={sens}")

    # ── model download (with progress) ──
    def downloadModels_(self, sender):  # noqa: N802
        self._dl_btn.setEnabled_(False)
        self._dl_btn.setTitle_("Downloading…")
        threading.Thread(target=self._download_worker, daemon=True).start()

    @objc.python_method
    def _ui(self, fn, *args):
        AppHelper.callAfter(fn, *args)

    @objc.python_method
    def _set_status(self, text):
        if self._dl_status is not None:
            self._dl_status.setStringValue_(text)

    @objc.python_method
    def _set_progress(self, pct):
        if self._progress is not None:
            self._progress.setDoubleValue_(max(0.0, min(100.0, pct)))

    @objc.python_method
    def _set_done(self):
        if self._progress is not None:
            self._progress.setDoubleValue_(100.0)
        if self._dl_status is not None:
            self._dl_status.setStringValue_("✓ Models ready — your first dictation will be instant.")
        if self._dl_btn is not None:
            self._dl_btn.setEnabled_(True)
            self._dl_btn.setTitle_("Re-check / Download")

    @objc.python_method
    def _download_worker(self):
        cfg = self._app.cfg
        # Formatting model (Ollama).
        try:
            fmt = cfg.get("formatting", {})
            if fmt.get("enabled", True):
                url, model = fmt["ollama_url"], fmt["model"]
                self._ui(self._set_status, f"Formatting model: {model}…")
                if not self._ollama_has(url, model):
                    self._ollama_pull(url, model)
                self._ui(self._set_progress, 100.0)
        except Exception as e:
            log(f"onboarding ollama download error: {e}")
            self._ui(self._set_status, f"Formatting model issue: {e}")
        # Speech model (Whisper via Hugging Face).
        try:
            repo = cfg["transcription"]["model"]
            self._ui(self._set_status, f"Speech model: {repo.split('/')[-1]}…")
            self._ui(self._set_progress, 0.0)
            self._download_whisper(repo)
        except Exception as e:
            log(f"onboarding whisper download error: {e}")
            self._ui(self._set_status, f"Speech model issue: {e}")
        self._ui(self._set_done)

    @objc.python_method
    def _ollama_has(self, url, model):
        try:
            r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return model in names
        except Exception:
            return False

    @objc.python_method
    def _ollama_pull(self, url, model):
        with requests.post(
            f"{url.rstrip('/')}/api/pull", json={"name": model}, stream=True, timeout=3600
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                total, completed = d.get("total"), d.get("completed")
                if total and completed:
                    self._ui(self._set_progress, completed * 100.0 / total)
                if d.get("status"):
                    self._ui(self._set_status, f"{model}: {d['status']}")

    @objc.python_method
    def _hf_total_bytes(self, repo):
        try:
            r = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=10)
            return sum((s.get("size") or 0) for s in r.json().get("siblings", []))
        except Exception:
            return 0

    @objc.python_method
    def _dir_size(self, path):
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass
        return total

    @objc.python_method
    def _download_whisper(self, repo):
        cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
        total = self._hf_total_bytes(repo)
        err = {}

        def dl():
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo)
            except Exception as e:
                err["e"] = e

        t = threading.Thread(target=dl, daemon=True)
        t.start()
        while t.is_alive():
            if total:
                self._ui(self._set_progress, min(99.0, self._dir_size(cache) * 100.0 / total))
            time.sleep(0.5)
        t.join()
        if "e" in err:
            raise err["e"]
        self._ui(self._set_progress, 100.0)


# ── Transcription (Whisper via MLX) ──────────────────────────────────────────

def transcribe(audio: np.ndarray, model: str, language: str, vocabulary: str = "") -> dict:
    import mlx_whisper

    if audio.size == 0:
        return {"text": "", "segments": []}
    opts: dict = {}
    if language:
        opts["language"] = language
    if vocabulary:
        # Primes the decoder toward these spellings (names, jargon, acronyms).
        opts["initial_prompt"] = f"Glossary: {vocabulary}."
    return mlx_whisper.transcribe(audio, path_or_hf_repo=model, **opts)


def transcript_with_paragraphs(result: dict, pause_seconds: float) -> str:
    """Join Whisper segments, inserting a paragraph break on long spoken pauses."""
    segments = result.get("segments") or []
    if not segments or pause_seconds <= 0:
        return (result.get("text") or "").strip()
    parts: list[str] = []
    prev_end = None
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        if prev_end is not None and (seg.get("start", 0.0) - prev_end) >= pause_seconds:
            parts.append("\n\n")
        elif parts:
            parts.append(" ")
        parts.append(txt)
        prev_end = seg.get("end", prev_end)
    return "".join(parts).strip()


def apply_replacements(text: str, mapping: dict) -> str:
    """Deterministically fix mis-heard terms (case-insensitive, whole phrases).

    More-specific keys are applied first so a short key can't partially clobber
    a longer one.
    """
    if not text or not mapping:
        return text
    for wrong in sorted(mapping, key=len, reverse=True):
        right = mapping[wrong]
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


# ── Voice-tone (prosody) analysis ────────────────────────────────────────────

TONE_BASELINE_PATH = CONFIG_PATH.parent / "prosody_baseline.json"


def analyze_prosody(audio: np.ndarray) -> dict | None:
    """Loudness of voiced frames + pitch variability (semitones), via numpy."""
    sr = SAMPLE_RATE
    frame = int(0.03 * sr)
    hop = int(0.01 * sr)
    if audio.size < frame:
        return None
    # Frame energies → voicing threshold.
    energies = []
    for i in range(0, audio.size - frame, hop):
        fr = audio[i : i + frame]
        energies.append(float(np.sqrt(np.mean(fr * fr) + 1e-9)))
    energies = np.asarray(energies)
    if energies.size == 0:
        return None
    thresh = max(0.01, 0.4 * float(np.percentile(energies, 90)))
    voiced = energies[energies > thresh]
    rms = float(np.mean(voiced)) if voiced.size else float(np.mean(energies))
    # Pitch via autocorrelation on voiced frames (80–400 Hz).
    lag_min, lag_max = int(sr / 400), int(sr / 80)
    f0s = []
    for i in range(0, audio.size - frame, hop * 3):
        fr = audio[i : i + frame]
        if np.sqrt(np.mean(fr * fr) + 1e-9) <= thresh:
            continue
        fr = fr - np.mean(fr)
        ac = np.correlate(fr, fr, "full")[frame - 1 :]
        if ac.size <= lag_max or ac[0] <= 0:
            continue
        seg = ac[lag_min:lag_max]
        if seg.size == 0:
            continue
        peak = int(np.argmax(seg)) + lag_min
        if ac[peak] > 0.3 * ac[0]:
            f0s.append(sr / peak)
    if len(f0s) >= 3:
        f0arr = np.asarray(f0s)
        semis = 12.0 * np.log2(f0arr / np.median(f0arr))
        f0_std = float(np.std(semis))
    else:
        f0_std = 0.0
    return {"rms": rms, "f0_std": f0_std}


# ── Smart formatting (Ollama) ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a text-cleanup engine for a dictation app. Your ONLY \
job is to rewrite a raw speech-to-text transcript into clean written text.

⚠️ CRITICAL: You are NOT a chatbot or assistant. You must NEVER answer, reply \
to, respond to, or have a conversation with the text. If the transcript is a \
question, you output the cleaned-up question — you do NOT answer it. If it is a \
greeting like "hey how's it going", you output the cleaned-up greeting — you do \
NOT greet back. You only ever rewrite the input; you never produce new content.

Stay as close to VERBATIM as possible. Your edits are STRICTLY limited to:
1. Fixing punctuation, capitalization, spacing, and obvious transcription errors \
— including inserting a SMALL missing function word (a, an, the, it, to, is, \
of, that) ONLY when the sentence is clearly ungrammatical without it. Never \
insert content words (nouns, verbs, adjectives) and never change the meaning.
2. Choosing end punctuation that fits the wording's intent: a question mark for \
questions, and an exclamation mark when the phrasing is clearly excited, \
emphatic, or celebratory (e.g. "this is amazing", "let's go", "we did it", "I \
can't wait", "no way", "yes finally"). Use "!" SPARINGLY — only when the words \
genuinely convey excitement, at most one per sentence; otherwise a period. \
Do not add excitement that isn't in the wording — UNLESS a [Voice tone: ...] \
note says the speaker sounded excited, in which case you MAY use exclamation \
marks for emphatic sentences even when the wording alone is neutral. A question \
always ends with a single "?" — never "?!".
3. Removing ONLY non-lexical fillers: "um", "uh", "er", "ah", "hmm", "mm", and \
stuttered repetitions / false starts (e.g. "the the" → "the", "I-I went" → "I went").
4. Applying explicit spoken self-corrections. If the speaker corrects themselves \
(e.g. "the red one, sorry I mean the blue one", "no wait", "scratch that", "I \
didn't mean that, I meant..."), keep ONLY the corrected intent and drop the \
retracted words.
5. Formatting a list when the speaker clearly enumerates items ("first... \
second...", "one... two...").
6. Honoring spoken formatting commands ("new line", "new paragraph", "bullet \
point", "period", "comma", "question mark", "exclamation point/mark") by \
APPLYING them, not writing the words literally.
7. Preserving any paragraph breaks (blank lines) already present in the input — \
do NOT merge separate paragraphs back together.

KEEP EVERY REAL WORD THE SPEAKER SAID. Do NOT delete, shorten, paraphrase, or \
"tidy up" actual words — especially leading acknowledgments and discourse markers \
like "sure", "yeah", "yes", "no", "okay", "alright", "cool", "so", "well", \
"actually", "like", "you know", "right", "I mean". These are NOT filler — keep \
them exactly. The ONLY words you may drop are non-lexical fillers (um, uh, er, ah, \
hmm) and stutters. If in doubt, keep it. \
Do NOT add information, summarize, translate, or explain. Output ONLY the \
rewritten text — no preamble, no quotes, no commentary."""

# Few-shot pairs framed as a transform task. Note the greeting/thanks examples:
# they teach the model to CLEAN, never to reply.
FEWSHOT_PAIRS = [
    # Drops only "um"/"uh", keeps "so"/"and then", applies the milk→oat milk fix.
    (
        "um so i went to the store and i bought uh apples and then milk no wait "
        "i mean oat milk and some bread",
        "So I went to the store and I bought apples and then oat milk and some bread.",
    ),
    (
        "for the trip we need to pack first sunscreen second the passports and "
        "third uh the chargers",
        "For the trip we need to pack:\n\n1. Sunscreen\n2. The passports\n3. The chargers",
    ),
    # Discourse markers preserved verbatim — only punctuation/casing added.
    ("yeah that's a bit better", "Yeah, that's a bit better."),
    # Leading acknowledgments are kept, never dropped.
    ("sure here's the link", "Sure, here's the link."),
    ("okay no problem i'll send it over", "Okay, no problem, I'll send it over."),
    # Excited / celebratory wording → exclamation marks.
    ("wow this actually works that's incredible", "Wow, this actually works. That's incredible!"),
    ("let's go we finally shipped it", "Let's go! We finally shipped it!"),
    # Neutral wording → stays a period (don't over-exclaim).
    ("okay i finished the report", "Okay, I finished the report."),
    # Inserts only the clearly-missing article "the" — no other changes.
    ("i went to store and grabbed milk", "I went to the store and grabbed milk."),
    ("hey so how's it going", "Hey, so how's it going?"),
    ("okay well i think that works", "Okay, well, I think that works."),
    ("thank you", "Thank you."),
]

_INSTRUCTION = (
    "Rewrite this dictation transcript as clean written text per the rules. "
    "Output ONLY the rewritten text — never a reply.\n\nTranscript:\n"
)


def format_text(text: str, url: str, model: str, tone: str | None = None) -> str:
    if not text:
        return text
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for raw, clean in FEWSHOT_PAIRS:
        messages.append({"role": "user", "content": _INSTRUCTION + raw})
        messages.append({"role": "assistant", "content": clean})
    user_content = _INSTRUCTION + text
    if tone == "excited":
        user_content = (
            "[Voice tone: the speaker sounded a bit energetic. You MAY end ONE "
            "clearly emphatic sentence with '!' if it genuinely fits — but keep "
            "questions ending in '?' (NEVER '?!'), keep neutral statements ending "
            "in '.', never add or change words, and never exclaim more than one "
            "sentence.]\n\n"
        ) + user_content
    messages.append({"role": "user", "content": user_content})
    resp = requests.post(
        f"{url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    out = resp.json()["message"]["content"].strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    return out or text


# ── Clipboard + paste ──────────────────────────────────────────────────────────

def clipboard_get() -> str:
    try:
        return subprocess.run(
            ["pbpaste"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return ""


def clipboard_set(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True, timeout=5)


_kbd = keyboard.Controller()


def paste_into_focused_app() -> None:
    _kbd.press(keyboard.Key.cmd)
    _kbd.press("v")
    _kbd.release("v")
    _kbd.release(keyboard.Key.cmd)


def deliver_text(text: str, cfg: dict) -> None:
    if not text:
        return
    if not cfg["paste"]["auto_paste"]:
        clipboard_set(text)
        return
    previous = clipboard_get() if cfg["paste"]["restore_clipboard"] else None
    clipboard_set(text)
    time.sleep(0.05)
    paste_into_focused_app()
    if previous is not None:
        def _restore() -> None:
            time.sleep(0.6)
            clipboard_set(previous)

        threading.Thread(target=_restore, daemon=True).start()


# ── The app ──────────────────────────────────────────────────────────────────

class FlowApp(rumps.App):
    def __init__(self, cfg: dict) -> None:
        super().__init__(GLYPH[IDLE], quit_button=None)
        self.cfg = cfg
        self.state = IDLE
        audio_cfg = cfg.get("audio", {})
        device = resolve_input_device(audio_cfg.get("input_device", "builtin"))
        try:
            dev_name = sd.query_devices(device)["name"] if device is not None else "default"
        except Exception:
            dev_name = str(device)
        self.recorder = AudioRecorder(
            device=device,
            preroll_seconds=audio_cfg.get("preroll_seconds", 0.5),
            warm=audio_cfg.get("warm_mic", True),
        )
        log(f"mic: {dev_name} (warm={audio_cfg.get('warm_mic', True)}, "
            f"preroll={audio_cfg.get('preroll_seconds', 0.5)}s)")
        self._lock = threading.Lock()
        self._last_paste_ts = 0.0
        self._paste_done_ts = 0.0  # to ignore our own synthetic Cmd+V events
        self._context_changed = False  # set when you click/type elsewhere
        self._tone_baseline = self._load_tone_baseline()
        self.hud = RecordingHUD(
            level_provider=lambda: self.recorder.level,
            on_cancel=self.cancel,
            on_confirm=self.confirm,
        )

        self.status_item = rumps.MenuItem("Idle")
        self.fmt_item = rumps.MenuItem(
            "Smart formatting", callback=self.toggle_formatting
        )
        self.fmt_item.state = bool(cfg["formatting"]["enabled"])
        self.mic_menu = rumps.MenuItem("Microphone")
        self.settings = SettingsController.alloc().initWithApp_(self)
        self.history = HistoryController.alloc().initWithApp_(self)
        self.onboarding = OnboardingController.alloc().initWithApp_(self)

        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("Toggle dictation", callback=lambda _: self.toggle()),
            rumps.MenuItem("Settings…", callback=self.open_settings),
            rumps.MenuItem("Dictation History…", callback=self.open_history),
            rumps.MenuItem("Setup / Onboarding…", callback=self.open_onboarding),
            self.mic_menu,
            self.fmt_item,
            None,
            rumps.MenuItem(f"Hotkey: {KEY_LABELS.get(cfg['hotkey']['key'], cfg['hotkey']['key'])}", callback=None),
            rumps.MenuItem(
                f"Whisper: {cfg['transcription']['model'].split('/')[-1]}",
                callback=None,
            ),
            rumps.MenuItem(f"Formatter: {cfg['formatting']['model']}", callback=None),
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]
        self._populate_mic_menu()

        # The companion "Settings" app signals us by creating this file.
        self._settings_trigger = CONFIG_PATH.parent / ".show_settings"
        try:
            self._settings_trigger.unlink()  # clear any stale trigger
        except FileNotFoundError:
            pass
        self._settings_watch = rumps.Timer(self._check_settings_trigger, 0.4)
        self._settings_watch.start()

        self._start_hotkey_listener()

        # First run → show the onboarding wizard once the app loop is up.
        if not ONBOARDED_PATH.exists():
            AppHelper.callAfter(self.onboarding.show)

    def open_onboarding(self, _=None) -> None:
        AppHelper.callAfter(self.onboarding.show)

    def _check_settings_trigger(self, _timer) -> None:  # noqa: ANN001
        try:
            if self._settings_trigger.exists():
                self._settings_trigger.unlink()
                self.settings.show()
        except Exception as e:
            log(f"  settings trigger error: {e}")

    # ── Microphone picker ──
    def _populate_mic_menu(self) -> None:
        if self.mic_menu._menu is not None:  # submenu exists only after first add
            self.mic_menu.clear()
        current = str(self.cfg["audio"].get("input_device", "builtin"))

        def add(label: str, spec: str) -> None:
            item = rumps.MenuItem(label, callback=self._select_mic)
            item.spec = spec
            item.state = current == spec
            self.mic_menu.add(item)

        add("System Default", "default")
        add("Built-in (Mac mic)", "builtin")
        self.mic_menu.add(rumps.separator)
        for d in sd.query_devices():
            if d["max_input_channels"] > 0:
                add(d["name"], d["name"])
        self.mic_menu.add(rumps.separator)
        self.mic_menu.add(rumps.MenuItem("Rescan devices", callback=lambda _: self._populate_mic_menu()))

    def _select_mic(self, sender: rumps.MenuItem) -> None:
        self.apply_mic(sender.spec)

    def open_settings(self, _=None) -> None:
        AppHelper.callAfter(self.settings.show)

    def open_history(self, _=None) -> None:
        log("open_history requested")
        AppHelper.callAfter(self._show_history_safe)

    def _show_history_safe(self) -> None:
        try:
            self.history.show()
            log("history window shown")
        except Exception as e:
            log(f"history show error: {e!r}")

    def apply_mic(self, spec: str) -> None:
        if self.state != IDLE:
            rumps.notification("Voice-To-Text", "Busy", "Finish the current dictation first.")
            return
        device = resolve_input_device(spec)
        try:
            name = sd.query_devices(device)["name"] if device is not None else "System Default"
        except Exception:
            name = str(spec)
        try:
            self.recorder.set_device(device)
        except Exception as e:
            rumps.notification("Voice-To-Text", "Could not open that mic", str(e))
            return
        self.cfg["audio"]["input_device"] = spec
        self._persist("input_device", spec)
        self._populate_mic_menu()
        log(f"mic switched -> {name} ({spec})")
        rumps.notification("Voice-To-Text", "Microphone set", name)

    def apply_warm(self, on: bool) -> None:
        try:
            self.recorder.set_warm(on)
        except Exception as e:
            rumps.notification("Voice-To-Text", "Could not change mic mode", str(e))
            return
        self.cfg["audio"]["warm_mic"] = on
        self._persist("warm_mic", on)
        log(f"warm mic -> {on}")

    def _persist(self, key: str, value) -> None:  # noqa: ANN001
        if isinstance(value, bool):
            v = "true" if value else "false"
        elif isinstance(value, str):
            v = f'"{value}"'
        else:
            v = str(value)
        try:
            text = CONFIG_PATH.read_text()
            new = re.sub(
                rf"^(\s*{re.escape(key)}\s*=).*$", rf"\1 {v}", text, count=1, flags=re.M
            )
            CONFIG_PATH.write_text(new)
        except Exception as e:
            log(f"  could not persist {key}: {e}")

    # ── UI helpers ──
    def set_state(self, state: str, status: str | None = None) -> None:
        self.state = state
        self.title = GLYPH[state]
        self.status_item.title = status or state.capitalize()

    # ── Hotkey ──
    def _resolve_trigger(self):
        name = self.cfg["hotkey"]["key"]
        if hasattr(keyboard.Key, name):
            return getattr(keyboard.Key, name)
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        return keyboard.Key.alt_r  # fallback: Right Option

    def _start_hotkey_listener(self) -> None:
        # Pre-warm pyobjc's lazy lookup of AXIsProcessTrusted on the main thread.
        # Two pynput listeners (toggle + settings combo) otherwise race to load
        # it concurrently, and pyobjc's loader isn't thread-safe (KeyError).
        try:
            import HIServices

            HIServices.AXIsProcessTrusted()
        except Exception as e:
            log(f"  (AXIsProcessTrusted warm failed: {e})")

        self._trigger = self._resolve_trigger()
        self._trigger_down = False  # debounce auto-repeat while held
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

        combo = self.cfg["hotkey"].get("settings_combo", "")
        if combo:
            self._settings_hk = keyboard.GlobalHotKeys({combo: self.open_settings})
            self._settings_hk.daemon = True
            self._settings_hk.start()

        # Watch for clicks so we know when you've moved to a new spot, and
        # shouldn't auto-prepend a space to the next dictation.
        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    def _on_click(self, x, y, button, pressed) -> None:  # noqa: ANN001
        if pressed:
            self._context_changed = True

    def _on_press(self, key) -> None:  # noqa: ANN001
        if key == self._trigger:
            if not self._trigger_down:
                self._trigger_down = True
                self.toggle()
        elif time.time() - self._paste_done_ts > 0.5:
            # A real keystroke (not our own synthetic Cmd+V right after a paste)
            # means you've typed/moved — don't auto-space the next dictation.
            self._context_changed = True

    def _on_release(self, key) -> None:  # noqa: ANN001
        if key == self._trigger:
            self._trigger_down = False

    def toggle_formatting(self, sender: rumps.MenuItem) -> None:
        sender.state = not sender.state
        self.cfg["formatting"]["enabled"] = bool(sender.state)

    # ── Core flow ──
    def toggle(self) -> None:
        with self._lock:
            if self.state == PROCESSING:
                return
            if self.state == IDLE:
                self._begin_recording()
            elif self.state == RECORDING:
                self._end_recording_and_process()

    def confirm(self) -> None:
        """✓ button — same as stopping the hotkey."""
        with self._lock:
            if self.state == RECORDING:
                self._end_recording_and_process()

    def cancel(self) -> None:
        """✕ button — discard the recording, paste nothing."""
        with self._lock:
            if self.state != RECORDING:
                return
            self.recorder.stop()
            AppHelper.callAfter(self.hud.hide)
            if self.cfg["sounds"]["enabled"]:
                play(SOUND_CANCEL)
            self.set_state(IDLE, "Cancelled")

    def _begin_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as e:
            play(SOUND_ERROR)
            rumps.notification("Voice-To-Text", "Could not start recording", str(e))
            return
        if self.cfg["sounds"]["enabled"]:
            play(SOUND_START)
        AppHelper.callAfter(self.hud.show)
        self.set_state(RECORDING, "Recording… (tap hotkey or ✓ to stop)")
        log("● recording started")

    def _end_recording_and_process(self) -> None:
        audio = self.recorder.stop()
        AppHelper.callAfter(self.hud.hide)
        if self.cfg["sounds"]["enabled"]:
            play(SOUND_STOP)
        self.set_state(PROCESSING, "Transcribing…")
        log(f"■ stopped — {audio.size / SAMPLE_RATE:.1f}s captured, transcribing…")
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def _maybe_prepend_space(self, text: str) -> str:
        """Add a leading space only when continuing in the same spot — i.e. a
        recent previous paste and no click/typing since."""
        window = self.cfg["paste"].get("space_between_seconds", 0)
        if window and self._last_paste_ts and not self._context_changed:
            gap = time.time() - self._last_paste_ts
            if 0 < gap <= window and text[:1] not in (" ", "\n", "\t"):
                return " " + text
        return text

    # ── Voice-tone assessment ──
    def _load_tone_baseline(self) -> dict:
        try:
            return json.loads(TONE_BASELINE_PATH.read_text())
        except Exception:
            return {"rms": 0.0, "f0_std": 0.0, "count": 0}

    def _save_tone_baseline(self) -> None:
        try:
            TONE_BASELINE_PATH.write_text(json.dumps(self._tone_baseline))
        except Exception:
            pass

    def _assess_tone(self, audio: np.ndarray) -> str | None:
        feat = analyze_prosody(audio)
        if feat is None:
            return None
        b = self._tone_baseline
        sens = self.cfg.get("tone", {}).get("excitement_sensitivity", 1.5)
        excited = None
        if b["count"] >= 4:
            rms_ratio = feat["rms"] / max(1e-6, b["rms"])
            f0_ratio = feat["f0_std"] / b["f0_std"] if b["f0_std"] > 0 else 1.0
            # Loudness is the primary, reliable signal; pitch only reinforces a
            # clip that is ALSO at least a bit louder than usual.
            if rms_ratio >= sens or (rms_ratio >= 1.2 and f0_ratio >= sens):
                excited = "excited"
        # Adapt to your TYPICAL level on EVERY clip (slow EMA) so the baseline
        # tracks your normal voice and can never get stuck flagging everything.
        a = 0.1
        if b["count"] == 0:
            b["rms"], b["f0_std"] = feat["rms"], feat["f0_std"]
        else:
            b["rms"] = (1 - a) * b["rms"] + a * feat["rms"]
            b["f0_std"] = (1 - a) * b["f0_std"] + a * feat["f0_std"]
        b["count"] += 1
        self._save_tone_baseline()
        log(
            f"  tone: rms={feat['rms']:.3f} f0std={feat['f0_std']:.2f} → "
            f"{excited or 'neutral'} (baseline rms={b['rms']:.3f} "
            f"f0std={b['f0_std']:.2f} n={b['count']})"
        )
        return excited

    def _process(self, audio: np.ndarray) -> None:
        try:
            result = transcribe(
                audio,
                self.cfg["transcription"]["model"],
                self.cfg["transcription"]["language"],
                self.cfg["transcription"].get("vocabulary", ""),
            )
            tone_cfg = self.cfg.get("tone", {})
            text = transcript_with_paragraphs(
                result, tone_cfg.get("paragraph_pause_seconds", 0)
            )
            text = apply_replacements(text, self.cfg.get("replacements", {}))
            log(f"  transcript: {text!r}")
            if not text:
                self.set_state(IDLE, "Heard nothing")
                return
            tone = self._assess_tone(audio) if tone_cfg.get("detect_excitement", False) else None
            if self.cfg["formatting"]["enabled"]:
                self.status_item.title = "Formatting…"
                try:
                    text = format_text(
                        text,
                        self.cfg["formatting"]["ollama_url"],
                        self.cfg["formatting"]["model"],
                        tone=tone,
                    )
                    log(f"  formatted : {text!r}")
                except Exception as e:
                    log(f"  formatting skipped (Ollama error): {e}")
                    rumps.notification(
                        "Voice-To-Text",
                        "Formatting skipped (Ollama error)",
                        str(e),
                    )
            history_append(text)
            text = self._maybe_prepend_space(text)
            deliver_text(text, self.cfg)
            now = time.time()
            self._last_paste_ts = now
            self._paste_done_ts = now
            self._context_changed = False  # fresh baseline after pasting
            self.set_state(IDLE, "Pasted ✓")
            log(f"✓ pasted {text!r}")
        except Exception as e:
            play(SOUND_ERROR)
            self.set_state(IDLE, "Error")
            rumps.notification("Voice-To-Text", "Something went wrong", str(e))


def main() -> None:
    cfg = load_config()
    FlowApp(cfg).run()


if __name__ == "__main__":
    main()
