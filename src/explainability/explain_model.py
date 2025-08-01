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
            model_path: Path to trained Keras model (.keras file)
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
        
        # SHAP explainer will be initialized lazily with actual data
        self.shap_explainer = None
        self.shap_background = None
    
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
    
    def _create_shap_background(self, sample_image):
        """Create a meaningful background dataset for SHAP."""
        try:
            # Create diverse background samples
            backgrounds = []
            
            # Add the sample image itself (normalized)
            backgrounds.append(sample_image)
            
            # Add variations: darker, lighter, blurred versions
            darker = np.clip(sample_image * 0.5, 0, 1)
            lighter = np.clip(sample_image * 1.5, 0, 1)
            backgrounds.extend([darker, lighter])
            
            # Add noise variations
            for i in range(3):
                noise = np.random.normal(0, 0.05, sample_image.shape)
                noisy = np.clip(sample_image + noise, 0, 1)
                backgrounds.append(noisy)
            
            # Add uniform backgrounds
            backgrounds.append(np.zeros_like(sample_image))  # Black
            backgrounds.append(np.ones_like(sample_image) * 0.5)  # Gray
            backgrounds.append(np.ones_like(sample_image))  # White
            
            # Add edge-enhanced version
            from scipy import ndimage
            edges = ndimage.sobel(sample_image.squeeze())
            edges = np.expand_dims(edges, axis=-1)
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
            backgrounds.append(edges)
            
            return np.array(backgrounds[:10])  # Use up to 10 backgrounds
            
        except Exception as e:
            print(f"Warning: Could not create enhanced background, using simple backgrounds: {e}")
            # Fallback to simple backgrounds
            backgrounds = []
            backgrounds.append(np.zeros_like(sample_image))
            backgrounds.append(np.ones_like(sample_image) * 0.5)
            backgrounds.append(sample_image)
            return np.array(backgrounds)
    
    def explain_with_shap(self, image):
        """Generate SHAP explanation for TensorFlow model with improved background."""
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
        
        # Initialize SHAP explainer lazily with meaningful background
        if self.shap_explainer is None:
            try:
                print("Initializing SHAP explainer with enhanced background...")
                background = self._create_shap_background(image)
                self.shap_explainer = shap.DeepExplainer(self.model, background)
                print(f"SHAP explainer initialized with {len(background)} background samples")
            except Exception as e:
                print(f"SHAP initialization failed: {e}")
                return None, None
        
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
        image_batch = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
        
        # Ensure model is built by doing a forward pass
        try:
            # Build the model if it hasn't been built yet
            if not hasattr(self.model, '_built') or not self.model.built:
                self.model.build(input_shape=(None, *self.input_shape))
            _ = self.model(image_batch)
        except Exception as e:
            print(f"Error building model: {e}")
            return None, None
        
        # Find last convolutional layer if not specified
        if layer_name is None:
            conv_layers = [layer.name for layer in self.model.layers if 'conv2d' in layer.name.lower()]
            if conv_layers:
                layer_name = conv_layers[-1]  # Use the last conv layer
            else:
                print("No convolutional layer found for Grad-CAM")
                return None, None
        
        try:
            # For Sequential models, use a simpler approach
            # Get intermediate output directly from the layer
            with tf.GradientTape() as tape:
                tape.watch(image_batch)
                
                # Forward pass through the model
                predictions = self.model(image_batch)
                
                # Get intermediate layer output
                intermediate_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(layer_name).output
                )
                conv_outputs = intermediate_model(image_batch)
                
                # For binary classification with sigmoid, use the raw prediction value
                if self.num_classes == 1:
                    loss = predictions[0, 0]  # Single output for binary classification
                else:
                    class_idx = tf.argmax(predictions[0])
                    loss = predictions[0, class_idx]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                print("No gradients computed for Grad-CAM")
                return None, None
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0)
            if tf.reduce_max(heatmap) > 0:
                heatmap = heatmap / tf.reduce_max(heatmap)
            
            return heatmap.numpy(), predictions.numpy()[0]
            
        except Exception as e:
            print(f"Grad-CAM explanation failed: {e}")
            return None, None
    
    @tf.function
    def _compute_gradients(self, images, target_class_idx=None):
        """Compute gradients for integrated gradients."""
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images)
            
            if self.num_classes == 1:  # Binary classification
                if target_class_idx is None or target_class_idx == 1:
                    loss = predictions[0, 0]  # Defective class
                else:
                    loss = 1 - predictions[0, 0]  # OK class
            else:
                if target_class_idx is None:
                    target_class_idx = tf.argmax(predictions[0])
                loss = predictions[0, target_class_idx]
        
        return tape.gradient(loss, images)
    
    def explain_with_integrated_gradients(self, image, target_class_idx=None, m_steps=50, batch_size=32):
        """Generate Integrated Gradients explanation for TensorFlow model."""
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
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
        
        # Create baseline (black image)
        baseline = tf.zeros_like(image_tensor)
        
        try:
            # Generate m_steps interpolated images between baseline and input
            alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
            
            # Initialize integrated gradients
            integrated_gradients = tf.zeros_like(image_tensor)
            
            # Process in batches to avoid memory issues
            for i in range(0, len(alphas), batch_size):
                alpha_batch = alphas[i:i+batch_size]
                
                # Create interpolated images
                interpolated_images = []
                for alpha in alpha_batch:
                    interpolated = baseline + alpha * (image_tensor - baseline)
                    interpolated_images.append(interpolated[0])  # Remove batch dimension
                
                if interpolated_images:
                    interpolated_batch = tf.stack(interpolated_images)
                    
                    # Compute gradients for this batch
                    gradients = self._compute_gradients(interpolated_batch, target_class_idx)
                    
                    # Accumulate gradients
                    if gradients is not None:
                        integrated_gradients += tf.reduce_sum(gradients, axis=0, keepdims=True)
            
            # Average and scale by input difference
            integrated_gradients = integrated_gradients / len(alphas)
            integrated_gradients = integrated_gradients * (image_tensor - baseline)
            
            return integrated_gradients.numpy()[0], image_tensor.numpy()[0]
            
        except Exception as e:
            print(f"Integrated Gradients explanation failed: {e}")
            return None, None
    
    def visualize_explanations(self, image, explanations, save_path=None):
        """Visualize different explanations for TensorFlow models."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
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
            
            # Enhance SHAP visualization with better scaling
            vmax = max(abs(shap_attr.min()), abs(shap_attr.max()))
            if vmax > 0:
                im = axes[0, 2].imshow(shap_attr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                cbar = plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
                cbar.set_label('SHAP Value', rotation=270, labelpad=20)
            else:
                axes[0, 2].imshow(shap_attr, cmap='RdBu_r')
                
            axes[0, 2].set_title('SHAP Explanation')
            axes[0, 2].axis('off')
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
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=10)
            axes[1, 0].set_title('Grad-CAM')
            axes[1, 0].axis('off')
        
        # Integrated Gradients
        if 'integrated_gradients' in explanations and explanations['integrated_gradients'][0] is not None:
            ig_attr, _ = explanations['integrated_gradients']
            
            # Handle different shapes
            if len(ig_attr.shape) == 3:
                ig_attr = np.mean(ig_attr, axis=-1)
            
            # Enhanced visualization with better scaling
            vmax = max(abs(ig_attr.min()), abs(ig_attr.max()))
            if vmax > 0:
                im = axes[1, 1].imshow(ig_attr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
                cbar.set_label('Attribution', rotation=270, labelpad=20)
            else:
                axes[1, 1].imshow(ig_attr, cmap='RdBu_r')
                
            axes[1, 1].set_title('Integrated Gradients')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'Integrated Gradients\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('Integrated Gradients')
            axes[1, 1].axis('off')
        
        # Prediction probabilities
        axes[1, 2].bar(range(len(prediction)), prediction)
        axes[1, 2].set_xticks(range(len(prediction)))
        axes[1, 2].set_xticklabels(self.class_names)
        axes[1, 2].set_title('Prediction Probabilities')
        axes[1, 2].set_ylabel('Probability')
        
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
            axes[2, 0].text(0.05, y_pos, line, fontweight=fontweight, fontsize=fontsize, 
                           transform=axes[2, 0].transAxes, family='monospace')
        
        axes[2, 0].set_title('CNN Architecture', fontweight='bold', fontsize=12)
        axes[2, 0].axis('off')
        
        # Feature importance summary
        feature_summary = []
        if 'lime' in explanations and explanations['lime'] is not None:
            feature_summary.append('✓ LIME: Local feature importance')
        if 'shap' in explanations and explanations['shap'][0] is not None:
            feature_summary.append('✓ SHAP: Shapley values')
        if 'gradcam' in explanations and explanations['gradcam'][0] is not None:
            feature_summary.append('✓ Grad-CAM: Convolutional attention')
        if 'integrated_gradients' in explanations and explanations['integrated_gradients'][0] is not None:
            feature_summary.append('✓ Integrated Gradients: Pixel attribution')
        
        summary_text = 'Explainability Methods:\n' + '\n'.join(feature_summary)
        axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes, 
                       verticalalignment='top', fontsize=11, fontweight='bold')
        axes[2, 1].set_title('Methods Summary')
        axes[2, 1].axis('off')
        
        # Model performance metrics (if available)
        perf_text = [
            'Model Performance:',
            '─────────────────',
            'Architecture: Simple CNN',
            'Input: 300×300 Grayscale',
            'Classes: OK vs Defective',
            'Output: Sigmoid Probability',
            '',
            'Explainability Coverage:',
            f'• Local: {"✓" if "lime" in explanations else "✗"}',
            f'• Global: {"✓" if "shap" in explanations else "✗"}',
            f'• Attention: {"✓" if "gradcam" in explanations else "✗"}',
            f'• Attribution: {"✓" if "integrated_gradients" in explanations else "✗"}'
        ]
        
        y_start = 0.98
        for i, line in enumerate(perf_text):
            y_pos = y_start - (i * 0.07)
            if y_pos < 0.02:
                break
            fontweight = 'bold' if i < 2 or 'Explainability' in line else 'normal'
            fontsize = 11 if i < 2 else 9
            axes[2, 2].text(0.05, y_pos, line, fontweight=fontweight, fontsize=fontsize, 
                           transform=axes[2, 2].transAxes, family='monospace')
        
        axes[2, 2].set_title('Analysis Summary')
        axes[2, 2].axis('off')
        
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
        
        if 'integrated_gradients' in methods:
            print("Generating Integrated Gradients explanation...")
            try:
                explanations['integrated_gradients'] = self.explain_with_integrated_gradients(image_path)
            except Exception as e:
                print(f"Integrated Gradients failed: {e}")
                explanations['integrated_gradients'] = (None, None)
        
        # Visualize results
        self.visualize_explanations(image_path, explanations, save_path)
        
        return explanations

def main():
    """Example usage of TensorFlow model explainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain TensorFlow model predictions')
    parser.add_argument('--model-path', required=True, help='Path to trained Keras model (.keras)')
    parser.add_argument('--image-path', required=True, help='Path to image to explain')
    parser.add_argument('--model-type', default='simple', help='Model architecture (simple only)')
    parser.add_argument('--methods', nargs='+', 
                       default=['lime', 'shap', 'gradcam', 'integrated_gradients'],
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