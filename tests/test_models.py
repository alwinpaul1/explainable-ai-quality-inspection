"""
Unit tests for model architectures and functionality.
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn_model import QualityInspectionCNN, create_model


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 2
        self.image_size = 224
        
        # Create dummy input
        self.dummy_input = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size
        )
    
    def test_resnet50_model(self):
        """Test ResNet50 model creation and forward pass."""
        model = QualityInspectionCNN(
            num_classes=self.num_classes,
            backbone='resnet50',
            pretrained=False
        )
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test feature extraction
        features = model.get_features(self.dummy_input)
        self.assertIsInstance(features, torch.Tensor)
    
    def test_efficientnet_model(self):
        """Test EfficientNet model creation and forward pass."""
        model = QualityInspectionCNN(
            num_classes=self.num_classes,
            backbone='efficientnet',
            pretrained=False
        )
        
        with torch.no_grad():
            output = model(self.dummy_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_vgg16_model(self):
        """Test VGG16 model creation and forward pass."""
        model = QualityInspectionCNN(
            num_classes=self.num_classes,
            backbone='vgg16',
            pretrained=False
        )
        
        with torch.no_grad():
            output = model(self.dummy_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_create_model_function(self):
        """Test the create_model factory function."""
        model_types = ['resnet50', 'efficientnet', 'vgg16', 'simple']
        
        for model_type in model_types:
            model = create_model(
                model_type=model_type,
                num_classes=self.num_classes,
                pretrained=False
            )
            
            with torch.no_grad():
                output = model(self.dummy_input)
            
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_invalid_backbone(self):
        """Test that invalid backbone raises ValueError."""
        with self.assertRaises(ValueError):
            QualityInspectionCNN(
                num_classes=self.num_classes,
                backbone='invalid_backbone'
            )
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = create_model('resnet50', num_classes=self.num_classes, pretrained=False)
        
        # Check that model has parameters
        params = list(model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check that some parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)


class TestModelTraining(unittest.TestCase):
    """Test cases for model training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = create_model('simple', num_classes=2, pretrained=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create dummy data
        self.inputs = torch.randn(4, 3, 224, 224)
        self.targets = torch.randint(0, 2, (4,))
    
    def test_training_step(self):
        """Test a single training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check that loss is a scalar
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_evaluation_mode(self):
        """Test model in evaluation mode."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.inputs)
        
        # Check output shape and values
        self.assertEqual(outputs.shape, (4, 2))
        self.assertTrue(torch.all(torch.isfinite(outputs)))


if __name__ == '__main__':
    # Run tests
    unittest.main()