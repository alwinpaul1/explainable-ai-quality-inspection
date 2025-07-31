"""
CNN models for quality inspection following the notebook approach
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_simple_cnn(input_shape=(300, 300, 1), num_classes=1):
    """
    Create simple CNN model following the notebook architecture.
    
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
    Create data generators following the notebook approach.
    
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
        classes={"ok": 0, "defective": 1},  # Updated class names
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
    """Quality inspection model following notebook approach."""
    
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
        model_type: Model type (only 'simple' supported now)
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
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features for explainability."""
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        return features
    
    def get_feature_maps(self, x):
        """Get intermediate feature maps for visualization."""
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        # Register hooks for feature extraction
        hooks = []
        if self.backbone_name == 'resnet50':
            hooks.append(self.backbone.layer1.register_forward_hook(hook_fn))
            hooks.append(self.backbone.layer2.register_forward_hook(hook_fn))
            hooks.append(self.backbone.layer3.register_forward_hook(hook_fn))
            hooks.append(self.backbone.layer4.register_forward_hook(hook_fn))
        
        # Forward pass to collect feature maps
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return feature_maps

class SimpleQualityInspectionCNN(nn.Module):
    """Simple CNN for quick testing and prototyping."""
    
    def __init__(self, num_classes=2, input_channels=3):
        super(SimpleQualityInspectionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features for explainability."""
        with torch.no_grad():
            features = self.features(x)
            features = features.view(features.size(0), -1)
        return features

def create_model(model_type='resnet50', num_classes=2, pretrained=True):
    """Factory function to create models."""
    
    if model_type == 'simple':
        return SimpleQualityInspectionCNN(num_classes=num_classes)
    else:
        return QualityInspectionCNN(
            num_classes=num_classes,
            backbone=model_type,
            pretrained=pretrained
        )

class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models, voting='soft'):
        """
        Args:
            models: List of trained models
            voting: 'hard' or 'soft' voting
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        self.num_models = len(models)
    
    def forward(self, x):
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                if self.voting == 'soft':
                    output = F.softmax(output, dim=1)
                outputs.append(output)
        
        if self.voting == 'soft':
            # Average probabilities
            ensemble_output = torch.stack(outputs).mean(dim=0)
        else:
            # Hard voting - majority vote
            predictions = torch.stack([torch.argmax(out, dim=1) for out in outputs])
            ensemble_output = torch.mode(predictions, dim=0)[0]
        
        return ensemble_output