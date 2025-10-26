#1. DATA CONFIGURATIONS
# ------------------------------------------
DATA_CONFIG = {
    "TARGET_TEST_FILENAME": "balinesse_storiette.csv",
    "SOURCE_HF_DATASET": "brighter-dataset/BRIGHTER-emotion-categories",
    "TEST_DATA_DIR": "data/test_data",
    "CACHE_DIR": "data/hf_download_cache",
    "LABELS": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
    "TRAIN_TEXT_COLUMN": "text",
    "TEST_TEXT_COLUMN": "sentence",
    "NUM_LABELS": 6,
    "DATASET_LANGUAGES": ["ind", "jav", "sun"]
}

#2. MODEL CONFIGURATIONS
# ------------------------------------------
MODEL_PATHS = {
    "M-BERT": "bert-base-multilingual-cased",
    "LaBSE": "sentence-transformers/LaBSE",    
    "IndoBERT": "indolem/indobert-base-uncased",
    "RemBERT": "google/rembert"
}

#3. TRAINING CONFIGURATIONS
# ------------------------------------------
TRAINING_CONFIG = {
    "LEARNING_RATE": 2e-5,
    "BATCH_SIZE": 16,
    "EPOCHS": 100,
    "MAX_SEQ_LENGTH": 128,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_STEPS": 0,
    "SEED": 42

}