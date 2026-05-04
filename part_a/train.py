import os
import pandas as pd
import evaluate
import numpy as np

from part_a.model import transcribe
from part_a.utils import load_audio, normalize_text
from part_a.results import generate_results

# Data Loading:
BASE = "az_dataset"
AUDIO_DIR = f"{BASE}/clips"
TSV_PATH = f"{BASE}/validated.tsv"

df = pd.read_csv(TSV_PATH, sep="\t")
df = df[["path", "sentence"]].dropna()

df = df[df["sentence"].str.len() > 10].reset_index(drop=True)
df = df.sample(50, random_state=42).reset_index(drop=True)

df["audio_path"] = df["path"].apply(lambda x: os.path.join(AUDIO_DIR, x))


# Training loop:
predictions = []
references = []

for i in range(len(df)):

    audio_path = df.loc[i, "audio_path"]
    ref = df.loc[i, "sentence"]

    audio = load_audio(audio_path)

    pred = transcribe(audio)

    predictions.append(pred)
    references.append(ref)

    print(f"{i+1}/{len(df)}")


# Normalization:
preds = [normalize_text(x) for x in predictions]
refs = [normalize_text(x) for x in references]


# Results:
generate_results(preds, refs)