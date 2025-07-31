"""
CNN models for quality inspection with explainability support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class QualityInspectionCNN(nn.Module):
    """CNN model for quality inspection with feature extraction capabilities."""
    
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture ('resnet50', 'efficientnet', 'vgg16')
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super(QualityInspectionCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Initialize backbone
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
            
        elif backbone == 'efficientnet':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'vgg16':
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            self.feature_dim = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
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