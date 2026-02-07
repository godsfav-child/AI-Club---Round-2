"""
Configuration file for Speech Emotion Recognition project
Python 3.12.2 compatible
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' / 'RAVDESS'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 22050  # Standard sample rate for librosa
DURATION = 3.0  # seconds
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000

# Spectrogram Shape
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 1

# Emotion Labels (RAVDESS encoding)
EMOTION_DICT = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
NUM_CLASSES = len(EMOTIONS)

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'noise_factor': 0.005,  # Noise injection level
    'pitch_shift_range': 2,  # ±2 semitones
    'time_stretch_range': (0.9, 1.1),  # 90% to 110% speed
    'augmentation_per_sample': 2  # Number of augmented versions per original
}

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model Parameters
DROPOUT_RATE = 0.4
L2_REGULARIZATION = 0.001

# Callbacks
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Model Save Path
MODEL_SAVE_PATH = MODEL_DIR / 'best_model.keras'  # Using .keras format for TF 2.16+

# Python version check
import sys
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"Running on Python {PYTHON_VERSION}")

if sys.version_info < (3, 12):
    print("Warning: This project is optimized for Python 3.12.2+")
