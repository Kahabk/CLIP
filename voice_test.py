# safe_tts_worker.py  (launch this as a separate process)
# This file exposes a simple IPC: read lines from stdin -> synthesize -> write WAV path to stdout

import sys, os, hashlib
from TTS.api import TTS
import soundfile as sf

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER = "Claribel Dervla"
CACHE_DIR = "./tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

tts = TTS(model_name=MODEL, gpu=False)  # force cpu for safety
def cache_path(text):
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.wav")

for line in sys.stdin:
    text = line.strip()
    if not text:
        print("", flush=True)
        continue
    path = cache_path(text)
    if not os.path.exists(path):
        tts.tts_to_file(text=text, speaker=SPEAKER, language="en", file_path=path)
    print(path, flush=True)
