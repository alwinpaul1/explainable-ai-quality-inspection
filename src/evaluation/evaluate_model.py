"""
Model evaluation utilities
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import create_model
from src.utils.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from src.utils.visualization import plot_sample_predictions

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path, model_type='resnet50', num_classes=2, device=None):
        """
        Args:
            model_path: Path to trained model
            model_type: Model architecture type
            num_classes: Number of classes
            device: Device to run evaluation on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = ['OK', 'Defective']
        
        # Load model
        self.model = create_model(model_type, num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Model type: {model_type}")
        print(f"Device: {self.device}")
    
    def evaluate(self, dataloader, save_plots=True, save_dir='results/reports'):
        """
        Comprehensive evaluation of the model.
        
        Args:
            dataloader: DataLoader for evaluation
            save_plots: Whether to save evaluation plots
            save_dir: Directory to save plots
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Starting model evaluation...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_paths = []
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Evaluating')
            for batch_idx, (data, target, paths) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                # Get probabilities and predictions
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Accumulate results
                running_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_paths.extend(paths)
                
                # Update progress
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate final metrics
        avg_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        print(f"\nEvaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Detailed metrics
        prob_array = np.array(all_probabilities)
        metrics = calculate_metrics(
            all_targets, 
            all_predictions, 
            prob_array,
            self.class_names
        )
        
        # Create results dictionary
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'paths': all_paths
        }
        
        # Save plots if requested
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
            self._save_evaluation_plots(results, save_dir)
        
        return results
    
    def _save_evaluation_plots(self, results, save_dir):
        """Save evaluation plots."""
        print("Generating evaluation plots...")
        
        # Confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            results['metrics']['confusion_matrix'],
            class_names=self.class_names,
            save_path=cm_path
        )
        
        # Normalized confusion matrix
        cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(
            results['metrics']['confusion_matrix'],
            class_names=self.class_names,
            normalize=True,
            title='Normalized Confusion Matrix',
            save_path=cm_norm_path
        )
        
        # ROC curve (if binary classification)
        if self.num_classes == 2:
            roc_path = os.path.join(save_dir, 'roc_curve.png')
            plot_roc_curve(
                results['targets'],
                np.array(results['probabilities']),
                class_names=self.class_names,
                save_path=roc_path
            )
        
        print(f"Evaluation plots saved to: {save_dir}")
    
    def evaluate_single_image(self, image_path):
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        from PIL import Image
        from torchvision import transforms
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        pred_class = predicted_class.cpu().item()
        
        results = {
            'predicted_class': pred_class,
            'predicted_label': self.class_names[pred_class],
            'confidence': probs[pred_class],
            'probabilities': {
                self.class_names[i]: probs[i] for i in range(len(self.class_names))
            }
        }
        
        return results
    
    def find_misclassified_samples(self, dataloader, num_samples=10):
        """
        Find misclassified samples for analysis.
        
        Args:
            dataloader: DataLoader for evaluation
            num_samples: Number of misclassified samples to return
        
        Returns:
            List of misclassified sample information
        """
        misclassified = []
        
        with torch.no_grad():
            for batch_idx, (data, target, paths) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Find misclassified samples in this batch
                for i in range(data.size(0)):
                    if predicted[i] != target[i]:
                        misclassified.append({
                            'path': paths[i],
                            'true_label': self.class_names[target[i].item()],
                            'predicted_label': self.class_names[predicted[i].item()],
                            'confidence': probabilities[i, predicted[i]].item(),
                            'true_class_prob': probabilities[i, target[i]].item()
                        })
                        
                        if len(misclassified) >= num_samples:
                            return misclassified
        
        return misclassified
    
    def analyze_class_performance(self, results):
        """
        Analyze per-class performance.
        
        Args:
            results: Results from evaluate() method
        
        Returns:
            Dictionary with per-class analysis
        """
        metrics = results['metrics']
        
        analysis = {}
        
        for i, class_name in enumerate(self.class_names):
            class_indices = [j for j, target in enumerate(results['targets']) if target == i]
            
            if class_indices:
                class_predictions = [results['predictions'][j] for j in class_indices]
                class_probabilities = [results['probabilities'][j][i] for j in class_indices]
                
                analysis[class_name] = {
                    'num_samples': len(class_indices),
                    'accuracy': sum(1 for pred in class_predictions if pred == i) / len(class_predictions),
                    'precision': metrics['precision_per_class'][i],
                    'recall': metrics['recall_per_class'][i],
                    'f1_score': metrics['f1_per_class'][i],
                    'avg_confidence': np.mean(class_probabilities),
                    'min_confidence': np.min(class_probabilities),
                    'max_confidence': np.max(class_probabilities)
                }
        
        return analysis
    
    def generate_evaluation_report(self, results, save_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Results from evaluate() method
            save_path: Path to save the report
        
        Returns:
            String containing the report
        """
        report = []
        report.append("="*80)
        report.append("QUALITY INSPECTION MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-"*40)
        report.append(f"Test Loss:     {results['loss']:.4f}")
        report.append(f"Test Accuracy: {results['accuracy']:.2f}%")
        report.append(f"Precision:     {results['metrics']['precision']:.4f}")
        report.append(f"Recall:        {results['metrics']['recall']:.4f}")
        report.append(f"F1-Score:      {results['metrics']['f1_score']:.4f}")
        
        if 'auc_score' in results['metrics'] and results['metrics']['auc_score'] is not None:
            report.append(f"AUC Score:     {results['metrics']['auc_score']:.4f}")
        
        report.append("")
        
        # Per-class performance
        analysis = self.analyze_class_performance(results)
        
        report.append("PER-CLASS PERFORMANCE")
        report.append("-"*40)
        
        for class_name, stats in analysis.items():
            report.append(f"\n{class_name}:")
            report.append(f"  Samples:      {stats['num_samples']}")
            report.append(f"  Accuracy:     {stats['accuracy']:.4f}")
            report.append(f"  Precision:    {stats['precision']:.4f}")
            report.append(f"  Recall:       {stats['recall']:.4f}")
            report.append(f"  F1-Score:     {stats['f1_score']:.4f}")
            report.append(f"  Avg Confidence: {stats['avg_confidence']:.4f}")
        
        report.append("")
        
        # Confusion matrix
        report.append("CONFUSION MATRIX")
        report.append("-"*40)
        cm = results['metrics']['confusion_matrix']
        
        # Header
        report.append(f"{'':>12}", end="")
        for name in self.class_names:
            report.append(f"{name:>10}", end="")
        report.append("")
        
        # Matrix
        for i, name in enumerate(self.class_names):
            line = f"{name:>12}"
            for j in range(len(self.class_names)):
                line += f"{cm[i, j]:>10}"
            report.append(line)
        
        report.append("")
        report.append("="*80)
        
        # Join all lines
        report_text = "\n".join(report)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to: {save_path}")
        
        # Print to console
        print(report_text)
        
        return report_text

def main():
    """Example usage of model evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', required=True, help='Path to test data')
    parser.add_argument('--model-type', default='resnet50', help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--save-dir', default='results/reports', help='Save directory')
    
    args = parser.parse_args()
    
    # Load test data
    from src.data.dataset import QualityInspectionDataset
    from torch.utils.data import DataLoader
    
    test_dataset = QualityInspectionDataset(args.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # Evaluate
    results = evaluator.evaluate(test_loader, save_dir=args.save_dir)
    
    # Generate report
    report_path = os.path.join(args.save_dir, 'evaluation_report.txt')
    evaluator.generate_evaluation_report(results, save_path=report_path)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()