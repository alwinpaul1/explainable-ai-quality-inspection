"""
CNN models for quality inspection using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_simple_cnn(input_shape=(300, 300, 1), num_classes=1):
    """
    Create simple CNN model following optimized architecture.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(filters=32,
                     kernel_size=3,
                     strides=2,
                     activation='relu',
                     input_shape=input_shape),
        
        # First pooling layer
        layers.MaxPooling2D(pool_size=2,
                           strides=2),
        
        # Second convolutional layer
        layers.Conv2D(filters=16,
                     kernel_size=3,
                     strides=2,
                     activation='relu'),
        
        # Second pooling layer
        layers.MaxPooling2D(pool_size=2,
                           strides=2),
        
        # Flattening
        layers.Flatten(),
        
        # Fully-connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(rate=0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.2),
        
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def create_data_generators(train_path, test_path, image_size=(300, 300), batch_size=64, seed=123):
    """
    Create data generators for training.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        image_size: Target image size
        batch_size: Batch size
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, validation_dataset, test_dataset
    """
    # Training data generator with augmentation
    train_generator = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
        rescale=1./255,
        validation_split=0.2
    )
    
    # Test data generator (no augmentation)
    test_generator = ImageDataGenerator(rescale=1./255)
    
    gen_args = dict(
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="binary",
        classes={"ok": 0, "defective": 1},
        shuffle=True,
        seed=seed
    )
    
    train_dataset = train_generator.flow_from_directory(
        directory=train_path,
        subset="training",
        **gen_args
    )
    
    validation_dataset = train_generator.flow_from_directory(
        directory=train_path,
        subset="validation",
        **gen_args
    )
    
    test_dataset = test_generator.flow_from_directory(
        directory=test_path,
        **gen_args
    )
    
    return train_dataset, validation_dataset, test_dataset

class QualityInspectionModel:
    """Quality inspection model wrapper."""
    
    def __init__(self, input_shape=(300, 300, 1), num_classes=1):
        self.model = create_simple_cnn(input_shape, num_classes)
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def compile_model(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        """Compile the model."""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def get_model(self):
        """Get the Keras model."""
        return self.model

# Legacy function for backward compatibility
def create_model(model_type='simple', num_classes=1, pretrained=False, input_shape=(300, 300, 1)):
    """
    Create model with backward compatibility.
    
    Args:
        model_type: Model type (only 'simple' supported)
        num_classes: Number of classes (1 for binary)
        pretrained: Ignored (kept for compatibility)
        input_shape: Input shape
    
    Returns:
        Compiled Keras model
    """
    if model_type != 'simple':
        print(f"Warning: {model_type} not supported, using simple CNN")
    
    quality_model = QualityInspectionModel(input_shape, num_classes)
    quality_model.compile_model()
    return quality_model.get_model()