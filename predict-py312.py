"""
Live inference script for Speech Emotion Recognition
Python 3.12.2 compatible
Usage: python predict.py --audio path/to/audio.wav
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from preprocessing import process_audio_file


def load_model(model_path: str | Path | None = None) -> keras.Model | None:
    """
    Load trained model from file
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded Keras model or None if error
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    model_path = Path(model_path)
    
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_emotion(
    audio_path: str | Path, 
    model: keras.Model
) -> tuple[str | None, float | None, np.ndarray | None]:
    """
    Predict emotion from audio file
    
    Args:
        audio_path: Path to audio file (.wav)
        model: Trained Keras model
    
    Returns:
        Tuple of (predicted_emotion, confidence, all_probabilities)
    """
    print(f"\nProcessing audio: {audio_path}")
    
    # Process audio file
    spectrograms = process_audio_file(audio_path, augment=False)
    
    if len(spectrograms) == 0:
        print("Error: Could not process audio file")
        return None, None, None
    
    # Get spectrogram
    spectrogram = spectrograms[0]
    
    # Add batch and channel dimensions
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    
    # Normalize (using approximate values)
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
    
    # Predict
    predictions = model.predict(spectrogram, verbose=0)
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    predicted_emotion = config.EMOTIONS[predicted_idx]
    
    return predicted_emotion, confidence, predictions[0]


def print_all_probabilities(probabilities: np.ndarray) -> None:
    """
    Print probabilities for all emotions
    
    Args:
        probabilities: Array of probabilities for each class
    """
    print("\nProbabilities for all emotions:")
    print("-" * 40)
    for i, emotion in enumerate(config.EMOTIONS):
        prob = probabilities[i] * 100
        bar = '█' * int(prob / 2)
        print(f"{emotion:12s} | {bar:50s} {prob:5.2f}%")
    print("-" * 40)


def main() -> None:
    """
    Main prediction function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict emotion from audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--audio', 
        type=str, 
        required=True,
        help='Path to audio file (.wav)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=str(config.MODEL_SAVE_PATH),
        help='Path to trained model'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show probabilities for all emotions'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPEECH EMOTION RECOGNITION - LIVE INFERENCE")
    print("=" * 60)
    print(f"Python version: {config.PYTHON_VERSION}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Predict
    emotion, confidence, probabilities = predict_emotion(args.audio, model)
    
    if emotion is None:
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    
    if args.verbose or confidence < 50:
        print_all_probabilities(probabilities)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
