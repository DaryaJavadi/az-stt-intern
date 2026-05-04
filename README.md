# 🇦🇿 Azərbaycan Dili üçün Nitqdən Mətnə (ASR) Layihəsi

## Layihə Haqqında

Bu layihə OpenAI Whisper modelindən istifadə edərək Azərbaycan dili üçün Automatic Speech Recognition (ASR) sistemi qurmağa yönəlib.  

Layihəyə aşağıdakılar daxildir:
- Dataset preprocessing və təmizləmə
- Pretrained Whisper modeli ilə baseline inference
- Whisper modelinin Azərbaycan dili üzərində fine-tune edilməsi
- WER (Word Error Rate) və CER (Character Error Rate) metrikaları ilə qiymətləndirmə

Layihənin məqsədi yalnız performans artırmaq deyil, eyni zamanda tam ASR pipeline qurmaqdır.

---

## Dataset Məlumatları

```text
https://mozilladatacollective.com/datasets/cmn29hqvk015ko107fblsr5ay
```

---

## Model Məlumatları

### Base Model
- Model: `openai/whisper-medium`
- Task: Speech-to-Text (ASR)
- Framework: HuggingFace Transformers

### Fine-tuning Parametrləri
- Model: `openai/whisper-large-v3`
- Learning rate: `1e-5`
- Batch size: `1`
- Gradient accumulation steps: `8`
- Epoch sayı: `5`
- Optimizer: AdamW
- Mixed precision: FP16 (CUDA mövcuddursa)
- Dil: Azərbaycan dili

---

## Qiymətləndirmə Nəticələri

### 🔹 Baseline və Fine-tuned Model Müqayisəsi

| Model | WER | CER |
|---|---|---|
| Base Whisper | 0.4050 | 0.1050 |
| Fine-Tuned Whisper | 0.2317 | 0.0577 |

Fine-tuned model base model ilə müqayisədə daha aşağı WER və CER nəticəsi göstərmişdir.

---

## Overfitting Analizi və Training Davranışı

Datasetin kiçik (təxminən 100–200 nümunə) olması səbəbilə modelin overfitting etməməsi əsas məqsədlərdən biri idi.

Training nəticələrinə baxdıqda:

- Training Loss azalmağa davam edir  
- Validation Loss isə müəyyən mərhələdən sonra sabit qalır (plateau)

Bu vəziyyət overfitting deyil, modelin optimal öyrənmə nöqtəsinə çatdığını göstərir.

### Hesabat üçün qeyd:

Dataset kiçik (200 nümunə) olduğu üçün modelin datanı əzbərləməməsi adına təlimi 5 epoch-da saxladım. Qrafiklərdən görünür ki, Validation Loss artıq düşmür, bu isə təlimin optimal nöqtədə dayandığını göstərir.

---

## Ideal Checkpoint Seçimi

Training nəticələrinə əsasən 5-ci epoch ən yaxşı nəticəni vermişdir:

- WER: 0.2317  
- CER: 0.0577  
- Minimum Validation Loss  

Bu səbəbdən 5-ci epoch ideal checkpoint kimi seçilmişdir.

Əgər `load_best_model_at_end=True` istifadə olunubsa, model avtomatik olaraq ən yaxşı checkpoint-i saxlayır.

---

## Setup Təlimatları

### 1. Repository-ni clone edin
```bash
git clone https://github.com/your_username/az-stt-intern.git
cd az-stt-intern
```

---

## 2. Virtual environment yaradın

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Lazımi kitabxanaları quraşdırın

```bash
pip install -r requirements.txt
```

---

## 4. `.env` faylı yaradın

Layihənin əsas qovluğunda `.env` faylı yaradın:

```env
MDC_API_KEY=your_mozilla_data_collective_key
HF_TOKEN=your_huggingface_token
```

### HF Token almaq üçün

1. Aşağıdakı linkə keçin:

```text
https://huggingface.co/settings/tokens
```

2. Yeni token yaradın

3. Token-i `.env` faylına əlavə edin

---

### Mozilla Data Collective API açarı

`MDC_API_KEY` dataset download etmək üçün istifadə olunur.

```text
https://mozilladatacollective.com/datasets/cmn29hqvk015ko107fblsr5ay
```

---

# Part A - Base ASR Pipeline

## Dataset download və hazırlıq

```bash
python -m part_a.download_dataset
```

Bu mərhələdə:
- dataset yüklənir
- extract olunur
- qovluq strukturu sadələşdirilir

---

## Base model evaluation

```bash
python -m part_a.train
```

Bu mərhələ:
- Whisper modelini inference üçün işə salır
- WER/CER hesablayır
- ən yaxşı və ən pis nümunələri göstərir
- nəticələri CSV faylı kimi saxlayır

---

# Part B - Fine-Tuning

## Fine-tuning prosesini işə salmaq

```bash
python -m part_b.train
```

Bu mərhələdə:
- dataset train/validation olaraq bölünür
- preprocessing tətbiq olunur
- Whisper modeli fine-tune edilir
- hər epoch üçün validation WER izlənir
- ən yaxşı checkpoint saxlanılır

---

# Vizualizasiya və Nəticələr

`results/` qovluğunda aşağıdakılar yerləşir:
- training loss qrafikləri
- validation WER qrafikləri
- comparison cədvəlləri
- CSV nəticələri

---

# Overfitting Qarşısının Alınması

Layihədə aşağıdakı metodlardan istifadə olunub:
- validation split
- low learning rate
- best checkpoint selection
- epoch-based evaluation
- audio normalization
- silence trimming
- text normalization

---

# GPU Tövsiyəsi

Fine-tuning üçün GPU istifadəsi tövsiyə olunur.

İstifadə edilə bilər:
- Google Colab
- Kaggle Notebooks

CPU ilə training olduqca yavaş işləyə bilər.

---

# Nəticə

Layihədə Azərbaycan dili üçün işləyən ASR pipeline qurulmuş, model qiymətləndirilmiş və fine-tuning cəhdi həyata keçirilmişdir. Fine-tuned model base model ilə müqayisədə daha yaxşı nəticə göstərmişdir.