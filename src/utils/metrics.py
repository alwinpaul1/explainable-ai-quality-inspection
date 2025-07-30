"""
Metrics calculation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculate comprehensive metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    else:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
    
    # AUC metrics (if probabilities provided)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        if len(np.unique(y_true)) == 2:  # Binary classification
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                metrics['auc_score'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                metrics['auc_score'] = None
                metrics['average_precision'] = None
        else:  # Multi-class
            try:
                metrics['auc_score'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                metrics['auc_score'] = None
    
    return metrics

def plot_confusion_matrix(cm, class_names=None, normalize=False, title='Confusion Matrix', 
                         figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve(y_true, y_prob, class_names=None, figsize=(8, 6), save_path=None):
    """
    Plot ROC curves.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if len(np.unique(y_true)) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
    else:  # Multi-class
        if class_names is None:
            class_names = [f'Class {i}' for i in range(y_prob.shape[1])]
        
        # One-vs-rest ROC curves
        for i in range(len(class_names)):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend()
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, class_names=None, figsize=(8, 6), save_path=None):
    """
    Plot precision-recall curves.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if len(np.unique(y_true)) == 2:  # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
    else:  # Multi-class
        if class_names is None:
            class_names = [f'Class {i}' for i in range(y_prob.shape[1])]
        
        # One-vs-rest PR curves
        for i in range(len(class_names)):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
            ap = average_precision_score(y_true_binary, y_prob[:, i])
            
            plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multi-class Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_metrics_summary(metrics, class_names=None):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics
        class_names: Names of classes
    """
    print("=" * 60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 60)
    
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    
    if 'auc_score' in metrics and metrics['auc_score'] is not None:
        print(f"AUC Score:    {metrics['auc_score']:.4f}")
    
    if 'average_precision' in metrics:
        print(f"Avg Precision: {metrics['average_precision']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 40)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['precision_per_class']))]
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}: P={metrics['precision_per_class'][i]:.3f}, "
              f"R={metrics['recall_per_class'][i]:.3f}, "
              f"F1={metrics['f1_per_class'][i]:.3f}")
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    cm = metrics['confusion_matrix']
    
    # Print header
    print(f"{'':>12}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    
    # Print matrix
    for i, name in enumerate(class_names):
        print(f"{name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    print("=" * 60)

class MetricsTracker:
    """Track metrics during training and validation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """Update with new batch of predictions."""
        if isinstance(predictions, list):
            self.predictions.extend(predictions)
        else:
            self.predictions.extend(predictions.cpu().numpy().tolist())
        
        if isinstance(targets, list):
            self.targets.extend(targets)
        else:
            self.targets.extend(targets.cpu().numpy().tolist())
        
        if probabilities is not None:
            if isinstance(probabilities, list):
                self.probabilities.extend(probabilities)
            else:
                self.probabilities.extend(probabilities.cpu().numpy().tolist())
    
    def compute_metrics(self, class_names=None):
        """Compute final metrics."""
        prob_array = np.array(self.probabilities) if self.probabilities else None
        return calculate_metrics(
            self.targets, 
            self.predictions, 
            prob_array, 
            class_names
        )