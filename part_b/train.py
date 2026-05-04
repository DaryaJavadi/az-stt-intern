import evaluate
from transformers import WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from .config import Config
from .model import get_whisper_model
from .dataset import AzerbaijanSpeechDataset, DataCollatorSpeechSeq2SeqWithPadding

# from dotenv import load_dotenv
# import os

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# os.environ["HF_TOKEN"] = HF_TOKEN

from dotenv import load_dotenv
import os
from huggingface_hub import login

# 1. Load .env
load_dotenv()

# 2. Get token
HF_TOKEN = os.getenv("HF_TOKEN")

# 3. Login to HuggingFace (DO THIS EARLY)
login(token=HF_TOKEN)

# Metrikaların qurulması
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str)
    }

def main():
    cfg = Config()
    processor = WhisperProcessor.from_pretrained(cfg.MODEL_ID, language="azerbaijani", task="transcribe")
    
    # Dataset
    ds_helper = AzerbaijanSpeechDataset(cfg, processor)
    train_ds, test_ds = ds_helper.load_and_prepare()
    
    # Model
    model = get_whisper_model(cfg.MODEL_ID)
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=cfg.EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=lambda p: compute_metrics(p, processor),
        processing_class=processor
    )

    trainer.train()
    
    # Save
    trainer.save_model(cfg.FINAL_MODEL_DIR)
    processor.save_pretrained(cfg.FINAL_MODEL_DIR)

if __name__ == "__main__":
    main()