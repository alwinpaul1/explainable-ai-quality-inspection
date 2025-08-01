"""
Model evaluation utilities following notebook approach
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import get_data_generators

class ModelEvaluator:
    """Model evaluation following notebook approach."""
    
    def __init__(self, model_path, model_type='simple', num_classes=1):
        """
        Args:
            model_path: Path to trained Keras model (.h5 file)
            model_type: Model architecture type (only 'simple' supported)
            num_classes: Number of classes (1 for binary with sigmoid)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.class_names = ['OK', 'Defective']
        
        # Load model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Loaded model from: {model_path}")
            print(f"Model type: {model_type}")
            print("Model architecture:")
            self.model.summary()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def evaluate(self, test_dataset, threshold=0.5, save_plots=True, save_dir='results/reports'):
        """
        Comprehensive evaluation following notebook approach.
        
        Args:
            test_dataset: TensorFlow dataset generator
            threshold: Classification threshold
            save_plots: Whether to save plots
            save_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded properly!")
            
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION (NOTEBOOK STYLE)")
        print("="*60)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Make predictions
        print("Making predictions on test dataset...")
        y_pred_prob = self.model.predict(test_dataset, verbose=1)
        
        # Convert probabilities to classes using threshold
        y_pred_class = (y_pred_prob >= threshold).reshape(-1,)
        y_true_class = test_dataset.classes[test_dataset.index_array]
        
        # Calculate detailed metrics following notebook
        accuracy = np.mean(y_true_class == y_pred_class)
        precision = precision_score(y_true_class, y_pred_class)
        recall = recall_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class)
        
        print(f"\n🎯 DETAILED TEST RESULTS (Following Notebook):")
        print(f"📊 Accuracy: {accuracy*100:.2f}%")
        print(f"📊 Precision: {precision*100:.2f}%")
        print(f"📊 Recall: {recall*100:.2f}%")
        print(f"📊 F1 Score: {f1*100:.2f}%")
        
        # Add notebook-style summary message
        print(f"\n✨ On test dataset, the model achieves a very good result as follow:")
        print(f"   • Accuracy: {accuracy*100:.2f}%")
        print(f"   • Recall: {recall*100:.2f}%")
        print(f"   • Precision: {precision*100:.2f}%")
        print(f"   • F1 score: {f1*100:.2f}%")
        
        # Create confusion matrix following notebook format
        cm = confusion_matrix(y_true_class, y_pred_class)
        cm_df = pd.DataFrame(
            cm,
            index=[["Actual", "Actual"], ["ok", "defect"]],
            columns=[["Predicted", "Predicted"], ["ok", "defect"]]
        )
        
        print("\n📊 Confusion Matrix:")
        print(cm_df)
        
        # Classification report
        print("\n📋 Classification Report:")
        report = classification_report(y_true_class, y_pred_class, digits=4, 
                                     target_names=['ok', 'defect'])
        print(report)
        
        # Save confusion matrix plot
        if save_plots:
            self._plot_confusion_matrix(cm, save_dir)
            self._plot_roc_curve(y_true_class, y_pred_prob, save_dir)
        
        # Add "Visualize Results" section following notebook
        if save_plots:
            self._visualize_results(test_dataset, y_pred_prob, threshold, save_dir)
        
        # Analyze misclassified samples following notebook
        misclassified_indices = np.where(y_pred_class != y_true_class)[0]
        print(f"\n🔍 Misclassified samples: {len(misclassified_indices)} out of {len(y_true_class)}")
        
        if len(misclassified_indices) > 0 and save_plots:
            self._visualize_misclassified(test_dataset, misclassified_indices, 
                                        y_pred_prob, threshold, save_dir)
        
        # Prepare results dictionary with detailed metrics
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm,
            'y_true': y_true_class,
            'y_pred': y_pred_class,
            'y_pred_prob': y_pred_prob.flatten(),
            'classification_report': report,
            'misclassified_indices': misclassified_indices,
            'threshold': threshold
        }
        
        # Save results to file
        results_file = os.path.join(save_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"DETAILED TEST RESULTS (Following Notebook):\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"Precision: {precision*100:.2f}%\n")
            f.write(f"Recall: {recall*100:.2f}%\n")
            f.write(f"F1 Score: {f1*100:.2f}%\n\n")
            f.write(f"On test dataset, the model achieves a very good result as follow:\n")
            f.write(f"• Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"• Recall: {recall*100:.2f}%\n")
            f.write(f"• Precision: {precision*100:.2f}%\n")
            f.write(f"• F1 score: {f1*100:.2f}%\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm_df) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n\n")
            f.write(f"Misclassified samples: {len(misclassified_indices)} out of {len(y_true_class)}\n")
            f.write(f"Classification threshold: {threshold}\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
    
    def _plot_confusion_matrix(self, cm, save_dir):
        """Plot confusion matrix following notebook style."""
        plt.figure(figsize=(8, 6))
        
        # Plot with seaborn heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['ok', 'defect'],
                   yticklabels=['ok', 'defect'])
        
        plt.title('CONFUSION MATRIX', fontweight='bold', fontsize=16)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('Actual Label', fontweight='bold')
        
        # Save plot
        plot_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Confusion matrix saved: {plot_path}")
        
        # Also create normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['ok', 'defect'],
                   yticklabels=['ok', 'defect'])
        
        plt.title('NORMALIZED CONFUSION MATRIX', fontweight='bold', fontsize=16)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('Actual Label', fontweight='bold')
        
        norm_plot_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plt.savefig(norm_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Normalized confusion matrix saved: {norm_plot_path}")
    
    def _plot_roc_curve(self, y_true, y_pred_prob, save_dir):
        """Plot ROC curve."""
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC CURVE', fontweight='bold', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        roc_path = os.path.join(save_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 ROC curve saved: {roc_path}")
    
    def _visualize_results(self, test_dataset, y_pred_prob, threshold, save_dir):
        """Visualize results comparing true vs predicted labels with probabilities (following notebook)."""
        mapping_class = {0: "ok", 1: "defect"}
        
        # Get first batch of test images
        images, labels = next(iter(test_dataset))
        batch_size = len(images)
        
        # Create 4x4 grid (16 images) following notebook style
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
            if i >= 16:  # Only show first 16 images
                break
                
            ax.imshow(img.squeeze(), cmap="gray")
            true_label = mapping_class[int(label)]
            
            # Get prediction for this specific image
            pred_prob = self.model.predict(img.reshape(1, *img.shape), verbose=0)[0][0]
            pred_label = mapping_class[int(pred_prob >= threshold)]
            
            prob_class = 100 * pred_prob if pred_label == "defect" else 100 * (1 - pred_prob)
            
            ax.set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=18)
            ax.set_xlabel(
                f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {prob_class:.2f}%",
                fontweight="bold", fontsize=15,
                color="blue" if true_label == pred_label else "red"
            )
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        fig.suptitle(
            "TRUE VS PREDICTED LABEL FOR 16 RANDOM TEST IMAGES", 
            size=30, y=1.03, fontweight="bold"
        )
        
        # Save visualization
        results_path = os.path.join(save_dir, 'test_predictions.png')
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Test predictions visualization saved: {results_path}")
    
    def _visualize_misclassified(self, test_dataset, misclassified_indices, 
                               y_pred_prob, threshold, save_dir):
        """Visualize misclassified samples following notebook approach."""
        mapping_class = {0: "ok", 1: "defect"}
        
        # Get a few misclassified samples
        num_samples = min(4, len(misclassified_indices))
        sample_indices = misclassified_indices[:num_samples]
        
        # Get images from dataset
        images, labels = next(iter(test_dataset))
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break
                
            # Get the image at this index (approximate)
            batch_num = idx // test_dataset.batch_size
            image_num = idx % test_dataset.batch_size
            
            try:
                batch_images, batch_labels = test_dataset[batch_num]
                img = batch_images[image_num]
                true_label = mapping_class[int(batch_labels[image_num])]
                
                # Get prediction
                pred_prob = y_pred_prob[idx]
                pred_label = mapping_class[int(pred_prob >= threshold)]
                prob_class = 100 * pred_prob if pred_label == "defect" else 100 * (1 - pred_prob)
                
                axes[i].imshow(img.squeeze(), cmap="gray")
                axes[i].set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=14)
                axes[i].set_xlabel(
                    f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {prob_class:.2f}%",
                    fontweight="bold", fontsize=12, color="red"
                )
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                
            except Exception as e:
                print(f"Error visualizing sample {idx}: {e}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        fig.suptitle(
            f"MISCLASSIFIED TEST IMAGES ({len(misclassified_indices)} total)",
            size=16, y=1.03, fontweight="bold"
        )
        
        # Save visualization
        misc_path = os.path.join(save_dir, 'misclassified_samples.png')
        plt.savefig(misc_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Misclassified samples visualization saved: {misc_path}")

def evaluate_model_notebook_style(model_path, data_dir, config):
    """
    Evaluate model following notebook approach.
    
    Args:
        model_path: Path to trained model (.h5 file)
        data_dir: Path to data directory
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        model_type=config.get('model_type', 'simple'),
        num_classes=config.get('num_classes', 1)
    )
    
    # Load test data
    try:
        _, _, test_dataset = get_data_generators(
            data_dir=data_dir,
            batch_size=config.get('batch_size', 64)
        )
        print(f"✅ Test dataset loaded: {test_dataset.samples} samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # Evaluate
    results = evaluator.evaluate(test_dataset, threshold=0.5)
    
    return results