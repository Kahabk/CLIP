#!/usr/bin/env python3


import os
import json
import hashlib
import threading
import re
import time
from multiprocessing import Process, Queue, Event
import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from vosk import Model, KaldiRecognizer
import scipy.signal
device = "cuda" if torch.cuda.is_avlable() else "cpu"
# ============================= CONFIG =============================
OLLAMA_MODEL = "qwen3-vl:4b"
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# VOSK models are normally trained at 16000 Hz. We'll resample input audio to this rate for recognition.
REC_SR = 16000

# Auto-detect a working device sample rate for recording/playback (fixes many ALSA issues)
def find_working_rate():
    for rate in [48000, 44100, 32000, 22050, 16000]:
        try:
            sd.check_input_settings(samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    return 44100  # fallback

SR = find_working_rate()
print(f"Using working device sample rate: {SR} Hz (will resample to {REC_SR} Hz for VOSK)")

# =================================================================

def cache_path(text, speaker="default", model_tag="xtts_v2"):
    """Deterministic cache path for generated TTS files."""
    key = f"{model_tag}|{speaker}|{text}"
    h = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.wav")

# ===================== TTS WORKER =====================
def tts_worker(in_q: Queue, out_q: Queue, stop: Event):
    try:
        from TTS.api import TTS
        print("[TTS] Loading XTTS model... (first time may take ~30-60s)")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False, progress_bar=True)
    except Exception as e:
        print("[TTS] Failed to load TTS model:", e)
        # Signal failure by putting empty string and exiting
        while not stop.is_set():
            try:
                txt = in_q.get(timeout=1)
                out_q.put("")
            except Exception:
                break
        return

    while not stop.is_set():
        try:
            text = in_q.get(timeout=1)
        except Exception:
            continue
        if not text or stop.is_set():
            break

        speaker = "Claribel Dervla"
        path = cache_path(text, speaker=speaker)
        if os.path.exists(path):
            out_q.put(path)
            continue

        try:
            # Use tts_to_file which reliably writes audio to disk
            tmp_path = path + ".tmp.wav"
            tts.tts_to_file(text=text, speaker=speaker, file_path=tmp_path, language="en")
            # Ensure final file exists
            if os.path.exists(tmp_path):
                os.replace(tmp_path, path)
                out_q.put(path)
            else:
                print("[TTS] tts_to_file did not produce output file")
                out_q.put("")
        except Exception as e:
            print("[TTS] Error generating audio:", e)
            out_q.put("")

# ===================== PLAYBACK =====================
def player(play_q: queue.Queue, stop: Event):
    while not stop.is_set():
        try:
            path = play_q.get(timeout=2)
        except Exception:
            continue
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[Player] File missing: {path}")
            continue
        try:
            data, file_sr = sf.read(path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            # Resample to device SR if needed
            if file_sr != SR:
                data = scipy.signal.resample_poly(data, SR, file_sr)
            sd.play(data, samplerate=SR, blocking=True)
            sd.wait()
        except Exception as e:
            print("[Player] Playback error:", e)

# ===================== OLLAMA (streaming helper) =====================
def ollama_stream(text, callback):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": text}],
        "stream": True
    }
    try:
        with requests.post("http://localhost:11434/api/chat", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            buffer = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    # If it's not JSON (some servers send plain text), treat as raw
                    data = {"message": {"content": line}}
                if data.get("done"):
                    break
                delta = data.get("message", {}).get("content", "")
                buffer += delta
                # Emit in readable chunks
                if len(buffer) > 80 or buffer.strip().endswith(('.', '!', '?')):
                    part = buffer.strip()
                    if part:
                        callback(part)
                    buffer = ""
            if buffer.strip():
                callback(buffer.strip())
    except Exception as e:
        print("[Ollama] Error:", e)
        callback("Sorry boss, Ollama is down.")

# ===================== MAIN =====================
def main():
    if not os.path.exists(VOSK_MODEL_PATH):
        print("Vosk model missing! Please download and unpack the VOSK model at:", VOSK_MODEL_PATH)
        return

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, REC_SR)
    rec.SetWords(True)

    # Multiprocessing queues/events for TTS worker
    tts_in = Queue()
    tts_out = Queue()
    tts_stop = Event()
    tts_proc = Process(target=tts_worker, args=(tts_in, tts_out, tts_stop), daemon=True)
    tts_proc.start()

    play_q = queue.Queue()
    play_stop = Event()
    threading.Thread(target=player, args=(play_q, play_stop,), daemon=True).start()

    last_sent = 0

    def speak(text, block_until_ready=True, timeout=30):
        """Send text to TTS; optionally wait for the generated wav path before queuing playback.
        This enforces "first step complete" semantics before next step.
        """
        text = text.strip()
        if not text:
            return
        speaker = "Claribel Dervla"
        path = cache_path(text, speaker=speaker)
        if os.path.exists(path):
            play_q.put(path)
            return

        # Ask the TTS worker to generate the file
        tts_in.put(text)
        if not block_until_ready:
            return
        try:
            # Wait for path from tts_out (the worker will put the path or empty string on failure)
            res = tts_out.get(timeout=timeout)
            if res:
                play_q.put(res)
            else:
                print("[Speak] TTS worker returned no path")
        except Exception:
            print("[Speak] Timeout waiting for TTS generation")

    def on_chunk(chunk):
        # Break long chunks into sentences and speak each fully
        for s in re.split(r'(?<=[.!?])\s+', chunk):
            s = s.strip()
            if s:
                speak(s, block_until_ready=True)

    def process(text):
        nonlocal last_sent
        if time.time() - last_sent < 1.2:
            return
        last_sent = time.time()
        print(f"\nYou: {text}")
        threading.Thread(target=ollama_stream, args=(text, on_chunk), daemon=True).start()

    def callback(indata, frames, timeinfo, status):
        # indata: float32 array at device SR. Resample to REC_SR for recognizer if needed.
        try:
            audio = indata[:, 0]
            if SR != REC_SR:
                # Resample from device SR -> REC_SR
                audio = scipy.signal.resample_poly(audio, REC_SR, SR)
            # Convert to 16-bit PCM bytes for VOSK
            pcm = (audio * 32767).astype(np.int16).tobytes()
            if rec.AcceptWaveform(pcm):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text and len(text.split()) >= 2:
                    process(text)
            else:
                # Partial result available if needed
                pass
        except Exception as e:
            print("[Callback] Error processing audio chunk:", e)

    print("\nROBOT IS ALIVE AND LISTENING – SPEAK NOW!\n")

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32', callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nGoodbye boss!")
    except Exception as e:
        print("[Main] Fatal error:", e)
    finally:
        # Signal TTS worker and join
        try:
            tts_stop.set()
        except Exception:
            pass
        try:
            play_stop.set()
        except Exception:
            pass
        # If process still alive, terminate
        try:
            if tts_proc.is_alive():
                tts_proc.terminate()
                tts_proc.join(timeout=1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
