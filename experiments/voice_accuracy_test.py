#!/usr/bin/env python3
"""Real-voice accuracy test: AssemblyAI streaming vs Groq Whisper vs local Whisper.

Records YOU reading 8 known passages, then transcribes each with all three
engines and reports word error rate (WER) against the exact text you read.
This is the definitive accuracy test — synthetic `say` audio is too clean to
separate the engines; your real voice (mic, room, accent) is what matters.

Run it from the project root, in a real Terminal window (it needs mic access):

    .venv/bin/python experiments/voice_accuracy_test.py

For each passage: press Enter, read the sentence aloud, press Enter again.
Recordings are saved under experiments/voice_test_recordings/ so you can re-score
later with:  .venv/bin/python experiments/voice_accuracy_test.py --score-only
"""
import json, os, sys, time, threading, wave, re, queue
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow

REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_test_recordings")
os.makedirs(REC_DIR, exist_ok=True)
SR = 16000

AAI_KEY = flow._resolve_api_key("ASSEMBLYAI_API_KEY", "~/.config/voice-to-text/assemblyai_key")
GROQ_KEY = flow._resolve_api_key("GROQ_API_KEY", "~/.config/voice-to-text/groq_key")
AAI_URL = ("wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&encoding=pcm_s16le"
           "&format_turns=true&speech_model=universal-streaming-english")
WMODEL = "mlx-community/whisper-large-v3-mlx"

PASSAGES = {
 "casual": "Hey, are we still on for dinner tonight? I was thinking we could try that new ramen place downtown around seven, but let me know if that works for you.",
 "email": "Hi Sarah, thanks for sending over the quarterly report. I've reviewed the numbers and everything looks good. Could we schedule a call on Thursday to discuss the budget for next quarter?",
 "jargon": "I deployed the new model using Ollama and the local Whisper pipeline on my Mac. Anthropic's Claude Code helped me wire up the MLX backend, and the latency dropped below half a second.",
 "names": "Gregory met with Priya and Mateo at the AssemblyAI office in San Francisco to discuss the Groq integration and the new streaming API.",
 "narrative": "When I first started building this app, I wanted something that respected privacy and ran entirely on my own machine. Over time it grew into a full dictation system with smart formatting and voice commands.",
 "questions": "Did you call the dentist? What time is the appointment? I think it's at three, but I'm not totally sure. Can you double check and text me back?",
 "disfluent": "So basically what I'm trying to say is that we should probably push the launch back a week or two just to be safe, because the testing isn't finished yet.",
 "mixed": "Let's meet at the coffee shop on Fifth Avenue at nine in the morning. Bring your laptop and the signed contract, and we'll go over the final details before the client arrives.",
}

# ── WER ────────────────────────────────────────────────────────────────────
def norm(s):
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]", " ", s).split()

def edits(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r)+1, len(h)+1), int)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)], len(r)

# ── record ─────────────────────────────────────────────────────────────────
def record_one(path):
    import sounddevice as sd
    q = queue.Queue()
    def cb(indata, frames, t, status): q.put(indata.copy())
    input("    ▶︎  Press Enter, then read it aloud… ")
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", callback=cb)
    frames = []
    with stream:
        print("    ●  recording… press Enter when you finish.")
        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        while not stop.is_set():
            try: frames.append(q.get(timeout=0.1))
            except queue.Empty: pass
        time.sleep(0.35)  # capture the natural trailing pause
        while not q.empty(): frames.append(q.get())
    pcm = np.concatenate(frames).astype(np.int16).tobytes() if frames else b""
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
    return len(pcm) // 2 / SR

# ── engines ────────────────────────────────────────────────────────────────
def load(path):
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())

def aai(pcm):
    import websocket
    turns, st = {}, {"b": threading.Event(), "d": threading.Event(), "t0": None}
    ws = websocket.create_connection(AAI_URL, header=[f"Authorization: {AAI_KEY}"], timeout=20)
    def rd():
        ws.settimeout(15)
        while True:
            try: m = ws.recv()
            except Exception: break
            if not m: break
            x = json.loads(m)
            if x.get("type") == "Begin": st["b"].set()
            elif "turn_order" in x:
                turns[x["turn_order"]] = x.get("transcript", "")
                if x.get("end_of_turn") and st["t0"]: st["d"].set()
            elif x.get("type") == "Termination": st["d"].set(); break
    threading.Thread(target=rd, daemon=True).start(); st["b"].wait(10)
    fr = int(SR*0.05)*2
    for i in range(0, len(pcm), fr):
        try: ws.send(pcm[i:i+fr], opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception: break
        time.sleep(0.05)
    st["t0"] = time.perf_counter()
    try: ws.send(json.dumps({"type": "ForceEndpoint"}))
    except Exception: pass
    st["d"].wait(6)
    try: ws.close()
    except Exception: pass
    return " ".join(turns[k] for k in sorted(turns)).strip()

def groq(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)/32768.0
    return flow.transcribe_remote(a, "https://api.groq.com/openai/v1", "whisper-large-v3",
                                  GROQ_KEY, "en", "").get("text", "").strip()

def local(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)/32768.0
    return flow.transcribe(a, WMODEL, "en", "").get("text", "").strip()

# ── flow ───────────────────────────────────────────────────────────────────
def main():
    score_only = "--score-only" in sys.argv
    if not score_only:
        print("\n" + "="*64)
        print("  REAL-VOICE ACCURACY TEST  —  read each line in your normal voice")
        print("="*64)
        for i, (label, text) in enumerate(PASSAGES.items(), 1):
            print(f"\n[{i}/{len(PASSAGES)}]  ({label})\n    “{text}”")
            dur = record_one(os.path.join(REC_DIR, f"{label}.wav"))
            print(f"    ✓ saved ({dur:.1f}s)")
        print("\nAll recorded. Scoring with AssemblyAI, Groq, and local Whisper…\n")

    engines = []
    if AAI_KEY: engines.append(("AssemblyAI", aai))
    else: print("  (no AssemblyAI key — skipping)")
    if GROQ_KEY: engines.append(("Groq", groq))
    else: print("  (no Groq key — skipping)")
    engines.append(("Local", local))

    agg = {n: [0, 0] for n, _ in engines}
    hdr = "passage".ljust(11) + "".join(n[:10].rjust(13) for n, _ in engines)
    print(hdr); print("-"*len(hdr))
    for label, text in PASSAGES.items():
        p = os.path.join(REC_DIR, f"{label}.wav")
        if not os.path.exists(p):
            print(f"{label:<11}  (no recording)"); continue
        pcm = load(p); row = label.ljust(11); details = []
        for name, fn in engines:
            try:
                h = fn(); e, n = edits(text, h); agg[name][0] += e; agg[name][1] += n
                row += f"{e/n*100:>11.1f}%"
                details.append(f"    {name:<11} {h[:88]!r}")
            except Exception as ex:
                row += f"{'ERR':>12}"; details.append(f"    {name:<11} ERR {ex}")
            time.sleep(0.4)
        print(row)
        for d in details: print(d)
    print("-"*len(hdr))
    for name, _ in engines:
        e, n = agg[name]
        if n: print(f"OVERALL {name:<12} WER {e/n*100:.2f}%   ({e} edits / {n} words)")
    print("\nRecordings kept in experiments/voice_test_recordings/  (delete anytime).")

if __name__ == "__main__":
    main()
