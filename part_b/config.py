class Config:
    BASE_DIR = "az_dataset"
    AUDIO_DIR = f"{BASE_DIR}/clips"
    TSV_PATH = f"{BASE_DIR}/validated.tsv"
    
    MODEL_ID = "openai/whisper-large-v3"
    OUTPUT_DIR = "./whisper-large-az-task"
    FINAL_MODEL_DIR = "./model/best_checkpoint"
    
    NUM_SAMPLES = 200
    EPOCHS = 5
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16