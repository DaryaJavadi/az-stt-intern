import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_whisper_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model