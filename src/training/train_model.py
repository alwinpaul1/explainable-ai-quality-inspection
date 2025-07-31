"""
Training script for quality inspection model with explainability
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import get_data_loaders
from src.models.cnn_model import create_model
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_history

class QualityInspectionTrainer:
    """Trainer class for quality inspection models."""
    
    def __init__(self, config):
        self.config = config
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize model
        self.model = create_model(
            model_type=config['model_type'],
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize loss function
        self.criterion = self._get_loss_function()
        
        # Initialize scheduler
        self.scheduler = self._get_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False
    
    def _get_optimizer(self):
        """Initialize optimizer."""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _get_loss_function(self):
        """Initialize loss function."""
        if self.config.get('class_weights'):
            weights = torch.tensor(self.config['class_weights']).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()
    
    def _get_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config['scheduler'] == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.3, min_lr=1e-7
            )
        elif self.config['scheduler'] == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs'], eta_min=1e-6
            )
        elif self.config['scheduler'] == 'warmup_cosine':
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=5
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']-5, eta_min=1e-6
            )
            return SequentialLR(
                self.optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[5]
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target, _) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target, _ in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Model: {self.config['model_type']}")
        print(f"Optimizer: {self.config['optimizer']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            if 'auc_score' in val_metrics:
                print(f"Val AUC: {val_metrics['auc_score']:.4f}")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Save best model and early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(
                    self.config['save_dir'], 'best_model.pth'
                )
                self.save_model(self.best_model_path, epoch, val_metrics)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                self.early_stopping_counter = 0  # Reset counter
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    self.early_stopping_triggered = True
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                checkpoint_path = os.path.join(
                    self.config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_model(checkpoint_path, epoch, val_metrics)
        
        total_time = time.time() - start_time
        
        if self.early_stopping_triggered:
            print(f"\nTraining stopped early after {epoch+1} epochs due to no improvement")
        else:
            print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
        
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.best_model_path
    
    def save_model(self, path, epoch, metrics):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics': metrics,
            'history': self.history
        }, path)
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Learning rate curve
        plt.subplot(1, 3, 3)
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'training_curves.png'), dpi=300)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train quality inspection model')
    parser.add_argument('--data-dir', default='../../data', help='Path to dataset')
    parser.add_argument('--model-type', default='resnet50', 
                       choices=['resnet50', 'efficientnet', 'vgg16', 'simple'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'cosine', 'none'])
    parser.add_argument('--save-dir', default='../../results/models', help='Save directory')
    parser.add_argument('--log-dir', default='../../results/logs', help='Log directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'num_classes': 2,  # defective vs ok
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_workers': args.num_workers,
        'pretrained': args.pretrained,
        'save_frequency': 10
    }
    
    # Print configuration
    print("Training Configuration:")
    print("-" * 30)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 30)
    
    # Create data loaders
    print("Loading dataset...")
    try:
        train_loader, val_loader = get_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure dataset is downloaded and in correct format.")
        return
    
    # Initialize trainer
    trainer = QualityInspectionTrainer(config)
    
    # Start training
    best_model_path = trainer.train(train_loader, val_loader)
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()