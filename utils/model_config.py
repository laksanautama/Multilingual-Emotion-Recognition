#1. DATA CONFIGURATIONS
# ------------------------------------------
DATA_CONFIG = {
    "TARGET_TEST_FILENAME": "balinesse_storiette_v1.csv",
    "SOURCE_HF_DATASET": "brighter-dataset/BRIGHTER-emotion-categories",
    "TEST_DATA_DIR": "data/test_data",
    "FAISS_INDEX_DIR": "data/faiss_index",
    "CACHE_DIR": "data/hf_download_cache",
    "LABELS": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
    "IND_LABELS": ["marah", "jijik", "takut", "senang", "sedih", "terkejut"],
    "BAL_LABELS": ["gedeg", "seneb", "takut", "sukha", "sedih", "makesiab"],
    "TRAIN_TEXT_COLUMN": "text",
    "TEST_TEXT_COLUMN": "sentence",
    "NUM_LABELS": 6,
    "LABEL_COLUMN": "labels",
    "DATASET_LANGUAGES": ["ind", "jav", "sun"],
    "TRAIN_SPLIT_SIZE": 0.5,
    "NUM_TRAIN_SAMPLES_PER_CLASS": 10,
    "NUM_VAL_SAMPLES": 10,
    "RAG_TOP_K": 5,
    "EMBEDDING_MODEL": "text-embedding-004"
}

#2. MODEL CONFIGURATIONS
# ------------------------------------------

MODEL_PATHS = {
    "mbert": "bert-base-multilingual-cased",
    "labse": "sentence-transformers/LaBSE",    
    "indobert": "indolem/indobert-base-uncased",
    "rembert": "google/rembert",
    "meta_llama": "meta-llama/Llama-3.1-8B-Instruct"
}

#3. TRAINING CONFIGURATIONS
# ------------------------------------------
TRAINING_CONFIG = {
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 8,
    "EPOCHS": 10,
    "MAX_SEQ_LENGTH": 256,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_STEPS": 0,
    "SEED": 42

}

#4. SAVED MODEL DIRECTORIES
# ------------------------------------------
MODEL_SAVE = {
    "indobert": "models/model_save/indobert",
    "labse": "models/model_save/labse",
    "mbert": "models/model_save/mbert",
    "rembert": "models/model_save/rembert"
}

#5. MODEL CHECKPOINTS
# ------------------------------------------
MODEL_CHECKPOINTS = {
    "indobert": "models/indobert_checkpoint",
    "labse": "models/labse_checkpoint",
    "mbert": "models/lmbert_checkpoint",
    "rembert": "models/rembert_checkpoint"
}