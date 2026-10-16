import os
from pathlib import Path

# Paths (Dynamic cross-platform root with environment variable override)
BASE_DIR = Path(os.getenv("URBANSOUND_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
DATASET_DIR = BASE_DIR / "UrbanSound8K"
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_CSV = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
OUTPUT_DIR = BASE_DIR / "pipeline_output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CACHE_DIR = OUTPUT_DIR / "cache"

# Audio Preprocessing Specs (Aligned with Modern Sound ML Standards)
SAMPLE_RATE = 22050
DURATION_SECONDS = 4.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)  # 88,200 samples

# Spectrogram Specs
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Class Taxonomy
CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

# Training Defaults
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 20
NUM_FOLDS = 10
