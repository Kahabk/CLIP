"""
Realtime STT -> Ollama (stream) -> TTS pipeline.

Requirements:
  pip install vosk sounddevice soundfile requests TTS
  and a downloaded Vosk model folder (set VOSK_MODEL_PATH).

Run:
  python realtime_stt_ollama_tts.py
"""

import os
import json
import queue
import threading
import hashlib
import time
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf

# STT: Vosk
from vosk import Model, KaldiRecognizer

# TTS: Coqui TTS (same API you used)
from TTS.api import TTS

# ------------------ CONFIG ------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-vl:4b"

VOSK_MODEL_PATH = "./vosk-model-small-en-us-0.15"  # change to your downloaded model folder
AUDIO_SAMPLE_RATE = 16000  # Vosk works well at 16k
DEVICE = None  # None or device index for sounddevice

USE_GPU_FOR_TTS = True
TTS_SPEAKER = "Claribel Dervla"
CHUNK_CHAR = 220
CACHE_DIR = "./tts_cache"

# STT trigger: either continuous or "press Enter to send" style.
# Here: we send each recognized complete sentence (end of phrase) to LLM.
# You can alter behavior to stream partial STT results.
# --------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- initialize TTS (Coqui) ----------
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=USE_GPU_FOR_TTS)

# ---------- Playback queue and thread ----------
play_q = queue.Queue()

def playback_worker():
    while True:
        item = play_q.get()
        if item is None:
            break
        audio, sr = item
        # ensure mono float32
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        sd.play(audio, samplerate=sr, blocking=True)
        play_q.task_done()

pb_thread = threading.Thread(target=playback_worker, daemon=True)
pb_thread.start()

def enqueue_play(audio, sr):
    play_q.put((audio, sr))

# ---------- simple TTS cache ----------
def tts_cache_path(text):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.wav")

def synthesize_text(text):
    """
    Return (audio_array, samplerate). Use cache when available.
    """
    path = tts_cache_path(text)
    if os.path.exists(path):
        data, sr = sf.read(path, dtype='float32')
        return data, sr

    try:
        # prefer tts_to_file to directly save file
        tts.tts_to_file(text=text, speaker=TTS_SPEAKER, language="en", file_path=path)
        data, sr = sf.read(path, dtype='float32')
        return data, sr
    except Exception:
        # fallback to tts.tts() returning numpy array (24k)
        wav = tts.tts(text=text, speaker=TTS_SPEAKER, language="en")
        audio = np.array(wav, dtype=np.float32)
        sr = 24000
        try:
            sf.write(path, audio, sr, subtype='PCM_16')
        except Exception:
            pass
        return audio, sr

# ---------- text chunker ----------
def chunk_text(text, max_chars=CHUNK_CHAR):
    words = text.split()
    chunks = []
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}".strip()
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks

# ---------- Ollama streaming (robust parsing) ----------
def stream_ollama_reply(user_text, on_chunk_callback=None):
    """
    Sends `user_text` to Ollama and streams partial text.
    on_chunk_callback(partial_text) is called when a new text delta arrives.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Reply in short helpful sentences."},
            {"role": "user", "content": user_text}
        ],
        "stream": True
    }
    with requests.post(OLLAMA_URL, json=payload, headers=headers, stream=True) as r:
        r.raise_for_status()
        buffer_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # flexible extraction of delta/token
            delta = None
            if isinstance(obj, dict):
                if 'response' in obj and isinstance(obj['response'], str):
                    delta = obj['response']
                elif 'token' in obj:
                    delta = obj['token']
                elif 'message' in obj and isinstance(obj['message'], dict):
                    content = obj['message'].get('content')
                    if isinstance(content, str):
                        delta = content
                elif 'delta' in obj and isinstance(obj['delta'], dict):
                    delta = obj['delta'].get('content')
            if not delta:
                continue
            buffer_text += delta
            # when we have sentence end or long chunk -> callback
            if len(buffer_text) >= CHUNK_CHAR or buffer_text.strip().endswith(('.', '!', '?')):
                if on_chunk_callback:
                    on_chunk_callback(buffer_text)
                buffer_text = ""
        # finished
        if buffer_text.strip():
            if on_chunk_callback:
                on_chunk_callback(buffer_text)

# ---------- Vosk STT listening ----------
def start_vosk_recognizer(queue_out, model_path=VOSK_MODEL_PATH, samplerate=AUDIO_SAMPLE_RATE, device=DEVICE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}. Download and set VOSK_MODEL_PATH.")
    model = Model(model_path)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        # convert to bytes (16-bit) for Vosk
        data = (indata * 32768).astype('int16').tobytes()
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text = res.get("text", "").strip()
            if text:
                queue_out.put(text)
        else:
            # partial can be accessed via rec.PartialResult()
            # We won't forward partials to LLM to avoid spamming; only full results.
            pass

    stream = sd.InputStream(samplerate=samplerate, device=device, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    return stream  # keep alive

# ---------- High level orchestration ----------
def main_loop():
    stt_queue = queue.Queue()

    print("Starting Vosk recognizer... speak into your mic.")
    stream = start_vosk_recognizer(stt_queue)

    try:
        while True:
            # wait for recognized phrase from STT
            text = stt_queue.get()  # blocking
            if not text:
                continue
            print("STT recognized:", text)

            # optional: simple voice command to exit
            if text.lower().strip() in ("exit", "quit", "stop", "goodbye"):
                print("Exit command detected. Shutting down.")
                break

            # send to Ollama and stream reply -> synthesize progressively
            partial_buffer = {"acc": ""}

            def on_chunk(chunk_text):
                # chunk_text may contain multiple sentences; break into small chunks
                chunks = chunk_text.strip()
                if not chunks:
                    return
                # produce sub-chunks for better streaming
                for c in chunk_text.splitlines():
                    for sub in chunk_text.split('. '):
                        if not sub.strip():
                            continue
                # simpler: just chunk by chars
                pieces = chunk_text_wrap(chunk_text)
                for p in pieces:
                    # synthesize and enqueue for playback
                    audio, sr = synthesize_text(p)
                    enqueue_play(audio, sr)

            # Because we want reliable chunking, define helper:
            def chunk_text_wrap(t):
                return chunk_text(t, max_chars=CHUNK_CHAR)

            print("Sending to Ollama...")
            stream_thread = threading.Thread(target=stream_ollama_reply, args=(text,on_chunk), daemon=True)
            stream_thread.start()
            stream_thread.join()  # block until reply complete (you can make this non-blocking if desired)
            print("Reply finished.")
    finally:
        print("Cleaning up...")
        stream.stop()
        play_q.put(None)
        pb_thread.join(timeout=2)

if __name__ == "__main__":
    main_loop()
