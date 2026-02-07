"""
CNN Model Architecture for Speech Emotion Recognition
Python 3.12.2 compatible with TensorFlow 2.16+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


def build_cnn_model(
    input_shape: tuple[int, int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS),
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    l2_reg: float = config.L2_REGULARIZATION
) -> keras.Model:
    """
    Build a 2D CNN model for emotion classification
    
    Architecture:
    - 4 Convolutional blocks with BatchNormalization and MaxPooling
    - Global Average Pooling to reduce parameters
    - Dropout for regularization
    - Dense layers for classification
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_classes: Number of emotion classes
        dropout_rate: Dropout probability
        l2_reg: L2 regularization factor
    
    Returns:
        Compiled Keras model
    """
    
    # Use Keras 3 compatible regularizer
    from keras.regularizers import L2
    regularizer = L2(l2_reg)
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Conv Block 2
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.6),
        
        # Conv Block 3
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.7),
        
        # Conv Block 4
        layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.8),
        
        # Global Average Pooling (reduces parameters, prevents overfitting)
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate * 0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model: keras.Model, learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_callbacks(model_save_path: str | None = None) -> list[keras.callbacks.Callback]:
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
    
    Returns:
        List of callbacks
    """
    if model_save_path is None:
        model_save_path = str(config.MODEL_SAVE_PATH)
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks


def create_model() -> keras.Model:
    """
    Convenience function to create and compile model
    
    Returns:
        Compiled model ready for training
    """
    model = build_cnn_model()
    model = compile_model(model)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    model = create_model()
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
