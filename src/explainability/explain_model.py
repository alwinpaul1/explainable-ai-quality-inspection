"""
Explainability methods for quality inspection models using TensorFlow/Keras
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Explainability libraries
from lime import lime_image
import shap

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import create_simple_cnn

class ModelExplainer:
    """Explainability wrapper for TensorFlow/Keras quality inspection models."""
    
    def __init__(self, model_path, model_type='simple', num_classes=1):
        """
        Args:
            model_path: Path to trained Keras model (.h5 file)
            model_type: Type of model architecture (only 'simple' supported)
            num_classes: Number of classes (1 for binary classification)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_shape = (300, 300, 1)  # Grayscale images
        
        # Load TensorFlow/Keras model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Loaded TensorFlow model from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Initialize explainability methods
        self._init_explainers()
        
        # Class names for binary classification
        self.class_names = ['OK', 'Defective']
    
    def _init_explainers(self):
        """Initialize different explainability methods for TensorFlow."""
        # LIME explainer
        self.lime_explainer = lime_image.LimeImageExplainer()
        
        # SHAP explainer (using DeepExplainer for neural networks)
        try:
            # Create background dataset for SHAP (small sample)
            background = np.zeros((10, *self.input_shape))
            self.shap_explainer = shap.DeepExplainer(self.model, background)
        except Exception as e:
            print(f"Warning: SHAP initialization failed: {e}")
            self.shap_explainer = None
    
    def predict_fn(self, images):
        """Prediction function for LIME (TensorFlow compatible)."""
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = images[np.newaxis, ...]
            
            # Ensure grayscale format (H, W, 1)
            if images.shape[-1] == 3:  # Convert RGB to grayscale
                images = np.mean(images, axis=-1, keepdims=True)
            elif len(images.shape) == 3:  # Add channel dimension if missing
                images = np.expand_dims(images, axis=-1)
            
            # Resize to model input size
            processed_images = []
            for img in images:
                if img.shape[:2] != self.input_shape[:2]:
                    img = cv2.resize(img.squeeze(), self.input_shape[:2])
                    img = np.expand_dims(img, axis=-1)
                processed_images.append(img)
            images = np.array(processed_images)
            
            # Normalize to [0, 1]
            if images.max() > 1.0:
                images = images / 255.0
        
        # Make predictions
        predictions = self.model.predict(images, verbose=0)
        
        # Convert to binary classification probabilities
        if self.num_classes == 1:  # Binary classification with sigmoid
            prob_defective = predictions.flatten()
            prob_ok = 1 - prob_defective
            return np.column_stack([prob_ok, prob_defective])
        else:
            return predictions
    
    def explain_with_lime(self, image, num_samples=1000, num_features=10):
        """Generate LIME explanation for TensorFlow model."""
        # Preprocess image
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('L'))  # Load as grayscale
        
        # Ensure proper format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image.squeeze(), self.input_shape[:2])
            image = np.expand_dims(image, axis=-1)
        
        # Ensure image is in [0, 255] range for LIME
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to RGB format for LIME (it expects 3 channels)
        if image.shape[-1] == 1:
            image_rgb = np.repeat(image, 3, axis=-1)
        else:
            image_rgb = image
        
        try:
            explanation = self.lime_explainer.explain_instance(
                image_rgb,
                self.predict_fn,
                top_labels=2,  # Binary classification
                hide_color=0,
                num_samples=num_samples,
                num_features=num_features
            )
            return explanation
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None
    
    def explain_with_shap(self, image):
        """Generate SHAP explanation for TensorFlow model."""
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return None
        
        # Preprocess image
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('L'))  # Load as grayscale
        
        # Ensure proper format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image.squeeze(), self.input_shape[:2])
            image = np.expand_dims(image, axis=-1)
        
        # Normalize
        if image.max() > 1.0:
            image = image / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        try:
            shap_values = self.shap_explainer.shap_values(image_batch)
            return shap_values, image_batch
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None, None
    
    def explain_with_gradcam(self, image, layer_name=None):
        """Generate Grad-CAM explanation for TensorFlow model."""
        # Preprocess image
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('L'))  # Load as grayscale
        
        # Ensure proper format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image.squeeze(), self.input_shape[:2])
            image = np.expand_dims(image, axis=-1)
        
        # Normalize
        if image.max() > 1.0:
            image = image / 255.0
        
        # Add batch dimension
        image_batch = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            print("No convolutional layer found for Grad-CAM")
            return None
        
        try:
            # Create a model that outputs both the original prediction and the feature maps
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_batch)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy(), predictions.numpy()[0]
            
        except Exception as e:
            print(f"Grad-CAM explanation failed: {e}")
            return None, None
    
    def visualize_explanations(self, image, explanations, save_path=None):
        """Visualize different explanations for TensorFlow models."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Preprocess original image for display
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('L'))
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            display_image = cv2.resize(image.squeeze(), self.input_shape[:2])
        else:
            display_image = image.squeeze()
        
        # Original image
        axes[0, 0].imshow(display_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Get prediction
        prediction = self.predict_fn(image)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        axes[0, 0].text(0.02, 0.98, 
                       f'Prediction: {self.class_names[predicted_class]}\nConfidence: {confidence:.3f}',
                       transform=axes[0, 0].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # LIME explanation
        if 'lime' in explanations and explanations['lime'] is not None:
            lime_exp = explanations['lime']
            temp, mask = lime_exp.get_image_and_mask(
                predicted_class, positive_only=False, num_features=10, hide_rest=False
            )
            # Convert to grayscale for display
            temp_gray = np.mean(temp, axis=-1) if len(temp.shape) == 3 else temp
            axes[0, 1].imshow(temp_gray, cmap='gray')
            axes[0, 1].set_title('LIME Explanation')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'LIME\nNot Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # SHAP explanation
        if 'shap' in explanations and explanations['shap'][0] is not None:
            shap_values, _ = explanations['shap']
            if isinstance(shap_values, list):
                shap_attr = shap_values[0][0].squeeze()  # First image, first class
            else:
                shap_attr = shap_values[0].squeeze()
                
            # Handle different shapes
            if len(shap_attr.shape) == 3:
                shap_attr = np.mean(shap_attr, axis=-1)
            
            im = axes[0, 2].imshow(shap_attr, cmap='RdBu_r')
            axes[0, 2].set_title('SHAP Explanation')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
        else:
            axes[0, 2].text(0.5, 0.5, 'SHAP\nNot Available', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
        
        # Grad-CAM
        if 'gradcam' in explanations and explanations['gradcam'][0] is not None:
            gradcam_heatmap, _ = explanations['gradcam']
            
            # Resize heatmap to match input image
            gradcam_resized = cv2.resize(gradcam_heatmap, self.input_shape[:2])
            
            # Overlay on original image
            axes[1, 0].imshow(display_image, cmap='gray', alpha=0.7)
            im = axes[1, 0].imshow(gradcam_resized, cmap='jet', alpha=0.5)
            axes[1, 0].set_title('Grad-CAM')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Prediction probabilities
        axes[1, 1].bar(range(len(prediction)), prediction)
        axes[1, 1].set_xticks(range(len(prediction)))
        axes[1, 1].set_xticklabels(self.class_names)
        axes[1, 1].set_title('Prediction Probabilities')
        axes[1, 1].set_ylabel('Probability')
        
        # CNN Architecture info with layer-by-layer details
        total_params = self.model.count_params()
        conv_layers = len([layer for layer in self.model.layers if 'conv' in layer.name.lower()])
        dense_layers = len([layer for layer in self.model.layers if 'dense' in layer.name.lower()])
        
        # Create a text-based architecture diagram
        arch_text = [
            'CNN Architecture:',
            '─────────────────',
            '300×300×1 (Input)',
            '     ↓',
            'Conv2D(32, 3×3, s=2)',
            '149×149×32',
            '     ↓',
            'MaxPool2D(2×2)',
            '74×74×32',
            '     ↓', 
            'Conv2D(16, 3×3, s=2)',
            '36×36×16',
            '     ↓',
            'MaxPool2D(2×2)',
            '18×18×16',
            '     ↓',
            'Flatten → 5,184',
            '     ↓',
            'Dense(128) + Dropout',
            '     ↓', 
            'Dense(64) + Dropout',
            '     ↓',
            'Dense(1, Sigmoid)',
            '',
            f'Total: {total_params:,} params',
            f'Size: {total_params*4/1024/1024:.1f} MB'
        ]
        
        # Display the architecture
        y_start = 0.98
        for i, line in enumerate(arch_text):
            y_pos = y_start - (i * 0.035)
            if y_pos < 0.02:  # Don't go below the plot area
                break
            fontweight = 'bold' if i < 2 or 'Total:' in line or 'Size:' in line else 'normal'
            fontsize = 10 if i < 2 else 8
            axes[1, 2].text(0.05, y_pos, line, fontweight=fontweight, fontsize=fontsize, 
                           transform=axes[1, 2].transAxes, family='monospace')
        
        axes[1, 2].set_title('CNN Architecture', fontweight='bold', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Explanation saved: {save_path}")
        
        plt.show()
    
    def explain_image(self, image_path, methods=['lime', 'shap'], save_path=None):
        """Comprehensive explanation of a single image using TensorFlow-compatible methods."""
        print(f"🔍 Explaining image: {image_path}")
        print(f"Methods: {methods}")
        
        explanations = {}
        
        # Generate explanations
        if 'lime' in methods:
            print("Generating LIME explanation...")
            try:
                explanations['lime'] = self.explain_with_lime(image_path)
            except Exception as e:
                print(f"LIME failed: {e}")
                explanations['lime'] = None
        
        if 'shap' in methods:
            print("Generating SHAP explanation...")
            try:
                explanations['shap'] = self.explain_with_shap(image_path)
            except Exception as e:
                print(f"SHAP failed: {e}")
                explanations['shap'] = (None, None)
        
        if 'gradcam' in methods:
            print("Generating Grad-CAM explanation...")
            try:
                explanations['gradcam'] = self.explain_with_gradcam(image_path)
            except Exception as e:
                print(f"Grad-CAM failed: {e}")
                explanations['gradcam'] = (None, None)
        
        # Visualize results
        self.visualize_explanations(image_path, explanations, save_path)
        
        return explanations

def main():
    """Example usage of TensorFlow model explainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain TensorFlow model predictions')
    parser.add_argument('--model-path', required=True, help='Path to trained Keras model (.h5)')
    parser.add_argument('--image-path', required=True, help='Path to image to explain')
    parser.add_argument('--model-type', default='simple', help='Model architecture (simple only)')
    parser.add_argument('--methods', nargs='+', 
                       default=['lime', 'shap', 'gradcam'],
                       help='Explanation methods to use')
    parser.add_argument('--save-path', help='Path to save explanation visualization')
    
    args = parser.parse_args()
    
    # Initialize explainer
    explainer = ModelExplainer(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # Generate explanations
    explainer.explain_image(
        args.image_path,
        methods=args.methods,
        save_path=args.save_path
    )
    
    print("✅ Explanation generation completed!")

if __name__ == "__main__":
    main()