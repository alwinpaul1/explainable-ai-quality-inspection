"""
Explainability methods for quality inspection models
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Explainability libraries
from lime import lime_image
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import create_model

class ModelExplainer:
    """Explainability wrapper for quality inspection models."""
    
    def __init__(self, model_path, model_type='resnet50', num_classes=2, device=None):
        """
        Args:
            model_path: Path to trained model
            model_type: Type of model architecture
            num_classes: Number of classes
            device: Device to run on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = create_model(model_type, num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize explainability methods
        self._init_explainers()
        
        # Class names
        self.class_names = ['OK', 'Defective']
    
    def _init_explainers(self):
        """Initialize different explainability methods."""
        # Captum methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)
        
        # GradCAM for CNN models
        if hasattr(self.model, 'backbone'):
            if 'resnet' in str(type(self.model.backbone)):
                self.gradcam = LayerGradCam(self.model, self.model.backbone.layer4)
            elif 'efficientnet' in str(type(self.model.backbone)):
                self.gradcam = LayerGradCam(self.model, self.model.backbone.features)
            else:
                self.gradcam = None
        else:
            self.gradcam = None
        
        # LIME explainer
        self.lime_explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images):
        """Prediction function for LIME."""
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = images[np.newaxis, ...]
            
            # Convert to tensor and preprocess
            images = torch.tensor(images).float()
            if images.shape[-1] == 3:  # HWC to CHW
                images = images.permute(0, 3, 1, 2)
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images / 255.0 - mean) / std
        
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain_with_lime(self, image, num_samples=1000, num_features=10):
        """Generate LIME explanation."""
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Ensure image is in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        explanation = self.lime_explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=self.num_classes,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features
        )
        
        return explanation
    
    def explain_with_integrated_gradients(self, image, target_class=None):
        """Generate Integrated Gradients explanation."""
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).float()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
        
        image = image.to(self.device)
        image.requires_grad_()
        
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image)
                target_class = outputs.argmax(dim=1).item()
        
        # Generate baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Compute attributions
        attributions = self.integrated_gradients.attribute(
            image, baseline, target=target_class, n_steps=50
        )
        
        return attributions, target_class
    
    def explain_with_gradcam(self, image, target_class=None):
        """Generate GradCAM explanation."""
        if self.gradcam is None:
            raise ValueError("GradCAM not available for this model architecture")
        
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).float()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
        
        image = image.to(self.device)
        image.requires_grad_()
        
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image)
                target_class = outputs.argmax(dim=1).item()
        
        # Compute GradCAM
        attributions = self.gradcam.attribute(image, target=target_class)
        
        return attributions, target_class
    
    def explain_with_occlusion(self, image, target_class=None, window_size=(15, 15)):
        """Generate Occlusion explanation."""
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).float()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
        
        image = image.to(self.device)
        
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image)
                target_class = outputs.argmax(dim=1).item()
        
        # Compute occlusion
        attributions = self.occlusion.attribute(
            image,
            strides=(3, 8, 8),
            target=target_class,
            sliding_window_shapes=(3,) + window_size,
            baselines=0
        )
        
        return attributions, target_class
    
    def visualize_explanations(self, image, explanations, save_path=None):
        """Visualize different explanations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        axes[0, 0].imshow(image)
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
        if 'lime' in explanations:
            lime_exp = explanations['lime']
            temp, mask = lime_exp.get_image_and_mask(
                predicted_class, positive_only=False, num_features=10, hide_rest=False
            )
            axes[0, 1].imshow(temp)
            axes[0, 1].set_title('LIME Explanation')
            axes[0, 1].axis('off')
        
        # Integrated Gradients
        if 'integrated_gradients' in explanations:
            ig_attr, _ = explanations['integrated_gradients']
            ig_attr = ig_attr.squeeze().cpu().numpy()
            if len(ig_attr.shape) == 3:
                ig_attr = np.transpose(ig_attr, (1, 2, 0))
            
            ig_attr = np.abs(ig_attr).sum(axis=2) if len(ig_attr.shape) == 3 else ig_attr
            
            im = axes[0, 2].imshow(ig_attr, cmap='hot')
            axes[0, 2].set_title('Integrated Gradients')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
        
        # GradCAM
        if 'gradcam' in explanations:
            gradcam_attr, _ = explanations['gradcam']
            gradcam_attr = gradcam_attr.squeeze().cpu().numpy()
            
            # Resize to match input image
            gradcam_attr = cv2.resize(gradcam_attr, (image.shape[1], image.shape[0]))
            
            im = axes[1, 0].imshow(gradcam_attr, cmap='jet', alpha=0.7)
            axes[1, 0].imshow(image, alpha=0.3)
            axes[1, 0].set_title('GradCAM')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # Occlusion
        if 'occlusion' in explanations:
            occ_attr, _ = explanations['occlusion']
            occ_attr = occ_attr.squeeze().cpu().numpy()
            if len(occ_attr.shape) == 3:
                occ_attr = np.mean(occ_attr, axis=0)
            
            im = axes[1, 1].imshow(occ_attr, cmap='RdBu_r')
            axes[1, 1].set_title('Occlusion')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        # Feature importance summary
        axes[1, 2].bar(range(len(prediction)), prediction)
        axes[1, 2].set_xticks(range(len(prediction)))
        axes[1, 2].set_xticklabels(self.class_names)
        axes[1, 2].set_title('Prediction Probabilities')
        axes[1, 2].set_ylabel('Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def explain_image(self, image_path, methods=['lime', 'integrated_gradients'], save_path=None):
        """Comprehensive explanation of a single image."""
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            image = image_path
        
        # Resize to model input size
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        explanations = {}
        
        # Generate explanations
        if 'lime' in methods:
            print("Generating LIME explanation...")
            try:
                explanations['lime'] = self.explain_with_lime(image)
            except Exception as e:
                print(f"LIME failed: {e}")
        
        if 'integrated_gradients' in methods:
            print("Generating Integrated Gradients explanation...")
            try:
                explanations['integrated_gradients'] = self.explain_with_integrated_gradients(image)
            except Exception as e:
                print(f"Integrated Gradients failed: {e}")
        
        if 'gradcam' in methods and self.gradcam is not None:
            print("Generating GradCAM explanation...")
            try:
                explanations['gradcam'] = self.explain_with_gradcam(image)
            except Exception as e:
                print(f"GradCAM failed: {e}")
        
        if 'occlusion' in methods:
            print("Generating Occlusion explanation...")
            try:
                explanations['occlusion'] = self.explain_with_occlusion(image)
            except Exception as e:
                print(f"Occlusion failed: {e}")
        
        # Visualize results
        self.visualize_explanations(image, explanations, save_path)
        
        return explanations

def main():
    """Example usage of model explainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain model predictions')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--image-path', required=True, help='Path to image to explain')
    parser.add_argument('--model-type', default='resnet50', help='Model architecture')
    parser.add_argument('--methods', nargs='+', 
                       default=['lime', 'integrated_gradients', 'gradcam'],
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
    
    print("Explanation generation completed!")

if __name__ == "__main__":
    main()