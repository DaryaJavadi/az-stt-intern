Part A:

pip install -r requirements.txt
python download_data.py
python part_a/train.py

Run this to see the results:
<!-- 1. -->
python -m part_a.download_data
<!-- 2 -->
python -m part_a.train



# 🇦🇿 Azerbaijani Speech-to-Text (ASR) Project

## Project Overview

This project focuses on building an Automatic Speech Recognition (ASR) system for the Azerbaijani language using the Whisper model from OpenAI.  
The project includes:
- Data preprocessing and cleaning
- Baseline inference using pretrained Whisper
- Fine-tuning Whisper on Azerbaijani speech data
- Evaluation using WER (Word Error Rate) and CER (Character Error Rate)

The goal is not only performance improvement but also building a complete ASR pipeline.

---

## Model Details:

### Base Model:
- Model: `openai/whisper-medium`
- Task: Speech-to-Text (ASR)
- Framework: HuggingFace Transformers

### Fine-tuning Parameters:
- Learning rate: `1e-5`
- Batch size: `1`
- Gradient accumulation steps: `8`
- Epochs: `5`
- Optimizer: AdamW
- Mixed precision: FP16 (if CUDA available)
- Language: Azerbaijani

---

## Evaluation Results:

### 🔹 Baseline vs Fine-tuned Model

| Model                | WER     | CER     |
|---------------------|---------|---------|
| Base Whisper        | 0.4050  | 0.1050  |
| Fine-Tuned Whisper  | 0.2317  | 0.0577  |

---

## Setup Instructions:

### 1. Clone repository
```bash
git clone https://github.com/your_username/az-stt-intern.git
cd az-stt-intern
