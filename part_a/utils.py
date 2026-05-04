import librosa
import re
import numpy as np


# Audio Preprocessing:
def load_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = audio / (max(abs(audio)) + 1e-9)
    return audio


# Text Normalization:
def normalize_text(text):
    text = text.lower()

    replacements = {
        "şə": "se",
        "ə": "e",
        "ı": "i",
        "ö": "o",
        "ü": "u",
        "ç": "c"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_avg_metric(values):
    return np.mean(values)