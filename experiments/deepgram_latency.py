"""Deepgram streaming latency prototype — proves the "hot socket, gated door"
design before we wire it into the app.

Architecture (per the plan):
  • The WebSocket to Deepgram stays HOT the whole time (a KeepAlive ping every
    few seconds keeps it alive with zero audio flowing).
  • Audio is GATED: mic frames are only forwarded to Deepgram while the "door"
    is open — i.e. between your first key press (start) and second (stop). The
    rest of the time your voice never leaves the machine.
  • Because the socket is already live, starting a dictation has ZERO handshake
    cost, and stopping returns the final transcript in ~one network round-trip.

It prints interim results live (the streaming feel) and, on stop, the
stop-to-final latency — the number that actually matters for "feels instant".

SETUP
  1. Get a Deepgram API key (deepgram.com — free tier includes credit).
  2. printf 'YOUR_DG_KEY' > ~/.config/voice-to-text/deepgram_key
  3. .venv/bin/python experiments/deepgram_latency.py
  4. Press ENTER to open the door and talk; ENTER again to stop. Ctrl-C quits.

Dependency: websocket-client (pip install websocket-client).
"""
import json
import os
import queue
import threading
import time
from pathlib import Path

import sounddevice as sd
import websocket  # websocket-client

SR = 16000
KEY_FILE = Path(os.path.expanduser("~/.config/voice-to-text/deepgram_key"))
# nova-2 = Deepgram's fast streaming model; endpointing=300ms decides "they
# stopped"; interim_results streams partial text as you talk.
URL = (
    "wss://api.deepgram.com/v1/listen?"
    "model=nova-2&encoding=linear16&sample_rate=%d&channels=1"
    "&interim_results=true&smart_format=true&endpointing=300" % SR
)


def ts() -> str:
    return time.strftime("%H:%M:%S") + ".%03d" % int((time.time() % 1) * 1000)


class DeepgramStreamer:
    def __init__(self, key: str):
        self._key = key
        self.ws = None
        self.connected = False
        self.door = False
        self.audioq: queue.Queue = queue.Queue()
        self.running = True
        self._t_close = None
        self._t_open = None
        self._got_first_interim = False
        self.final_parts: list[str] = []

    # ── hot connection ──
    def connect(self) -> None:
        t = time.time()
        self.ws = websocket.create_connection(
            URL, header=["Authorization: Token %s" % self._key])
        self.connected = True
        print("[%s] connected (hot) in %.0fms — socket stays open between dictations"
              % (ts(), (time.time() - t) * 1000))
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._keepalive_loop, daemon=True).start()
        threading.Thread(target=self._sender_loop, daemon=True).start()

    def _keepalive_loop(self) -> None:
        # Deepgram drops an idle socket after ~10s; KeepAlive holds it open while
        # the door is closed so the next dictation has no reconnect cost.
        while self.running:
            time.sleep(5)
            if self.connected and not self.door:
                self._safe_send(json.dumps({"type": "KeepAlive"}))

    def _sender_loop(self) -> None:
        while self.running:
            try:
                data = self.audioq.get(timeout=0.3)
            except queue.Empty:
                continue
            if self.connected:
                try:
                    self.ws.send_binary(data)
                except Exception as e:
                    print("[%s] send error: %r" % (ts(), e))

    def _recv_loop(self) -> None:
        while self.running and self.connected:
            try:
                msg = self.ws.recv()
            except Exception:
                break
            if not msg:
                continue
            try:
                d = json.loads(msg)
            except Exception:
                continue
            alt = (d.get("channel", {}).get("alternatives") or [{}])[0]
            txt = (alt.get("transcript") or "").strip()
            if not txt:
                continue
            if d.get("is_final"):
                self.final_parts.append(txt)
                if self._t_close is not None:
                    lat = (time.time() - self._t_close) * 1000
                    print("\n[%s] ✅ FINAL  (+%.0fms after you stopped): %s"
                          % (ts(), lat, txt))
                    self._t_close = None
                else:
                    print("\n[%s] final: %s" % (ts(), txt))
            else:
                if not self._got_first_interim and self._t_open is not None:
                    lat = (time.time() - self._t_open) * 1000
                    print("[%s] first interim (+%.0fms after first word): %s"
                          % (ts(), lat, txt))
                    self._got_first_interim = True
                else:
                    print("   … %s" % txt[-80:], end="\r", flush=True)

    def _safe_send(self, text: str) -> None:
        try:
            self.ws.send(text)
        except Exception:
            pass

    # ── the gated door ──
    def push(self, data: bytes) -> None:
        if self.door:
            self.audioq.put(data)

    def open_door(self) -> None:
        self.final_parts = []
        self._got_first_interim = False
        self._t_open = time.time()
        self.door = True
        print("\n[%s] ▶ DOOR OPEN — streaming your voice to Deepgram. Speak…" % ts())

    def close_door(self) -> None:
        self.door = False
        self._t_close = time.time()
        # Finalize flushes whatever's buffered and returns the final transcript.
        self._safe_send(json.dumps({"type": "Finalize"}))
        print("[%s] ⏹ DOOR CLOSED — voice no longer leaving the machine; finalizing…"
              % ts())

    def stop(self) -> None:
        self.running = False
        self.connected = False
        self._safe_send(json.dumps({"type": "CloseStream"}))
        try:
            self.ws.close()
        except Exception:
            pass


def main() -> None:
    if not KEY_FILE.exists() or not KEY_FILE.read_text().strip():
        print("No Deepgram key. Run:\n  printf 'YOUR_DG_KEY' > %s" % KEY_FILE)
        return
    streamer = DeepgramStreamer(KEY_FILE.read_text().strip())
    streamer.connect()

    def audio_cb(indata, frames, t, status):  # noqa: ANN001
        streamer.push(bytes(indata))

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                            blocksize=int(SR * 0.05), callback=audio_cb)
    stream.start()
    print("\nMic warm. Press ENTER to start dictating, ENTER again to stop. Ctrl-C to quit.\n")
    door_open = False
    try:
        while True:
            input()
            door_open = not door_open
            (streamer.open_door if door_open else streamer.close_door)()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stream.stop()
        streamer.stop()
        print("\nbye")


if __name__ == "__main__":
    main()
