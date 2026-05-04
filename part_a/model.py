from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device
)

def transcribe(audio, language="azerbaijani"):
    return pipe(
        audio,
        generate_kwargs={
            "language": language,
            "num_beams": 5
        }
    )["text"]