"""
Improved training script with advanced techniques for better performance
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.enhanced_dataset import get_enhanced_data_loaders, get_tta_transforms
from src.data.advanced_augmentation import MixUpCutMix, ProgressiveAugmentationScheduler
from src.models.cnn_model import create_model
from src.utils.metrics import calculate_metrics

class ImprovedQualityInspectionTrainer:
    """Enhanced trainer with advanced techniques and augmentation for better performance."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize model with improved architecture
        self.model = self._create_improved_model()
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights()
        
        # Initialize optimizer with better settings
        self.optimizer = self._get_improved_optimizer()
        
        # Initialize loss function with class weights
        self.criterion = self._get_weighted_loss_function()
        
        # Initialize advanced scheduler
        self.scheduler = self._get_advanced_scheduler()
        
        # Initialize MixUp/CutMix augmentation
        self.mixup_cutmix = MixUpCutMix(
            mixup_alpha=config.get('mixup_alpha', 0.2),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            prob=config.get('mixup_prob', 0.5)
        ) if config.get('use_mixup_cutmix', True) else None
        
        # Initialize progressive augmentation scheduler
        self.aug_scheduler = ProgressiveAugmentationScheduler(
            total_epochs=config['epochs'],
            start_strength='light',
            end_strength='strong'
        ) if config.get('use_progressive_aug', False) else None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': [],
            'augmentation_strength': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_path = None
        
        # Early stopping
        self.patience = config.get('patience', 15)
        self.patience_counter = 0
        
        # Enhanced TTA transforms
        self.tta_transforms = get_tta_transforms() if config.get('use_enhanced_tta', True) else None
        
    def _create_improved_model(self):
        """Create model with improved architecture."""
        model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        )
        
        # Add dropout and batch normalization improvements
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            # Improve classifier with higher dropout and batch norm
            classifier_layers = []
            in_features = model.classifier[1].in_features if len(model.classifier) > 1 else model.classifier[0].in_features
            
            classifier_layers.extend([
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, self.config['num_classes'])
            ])
            
            model.classifier = nn.Sequential(*classifier_layers)
        
        return model.to(self.device)
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        # Get class distribution from dataset
        train_data_path = Path(self.config['data_dir']) / 'train'
        
        class_counts = []
        for class_name in ['ok', 'defective']:
            class_path = train_data_path / class_name
            if class_path.exists():
                count = len(list(class_path.glob('*.jpeg'))) + len(list(class_path.glob('*.jpg')))
                class_counts.append(count)
        
        if len(class_counts) == 2:
            # Calculate weights inversely proportional to class frequency
            total = sum(class_counts)
            weights = [total / (len(class_counts) * count) for count in class_counts]
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            print(f"Class weights calculated: OK={weights[0]:.3f}, Defective={weights[1]:.3f}")
            return weights_tensor
        
        return None
    
    def _get_improved_optimizer(self):
        """Initialize optimizer with improved settings."""
        if self.config['optimizer'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay'],
                nesterov=True
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
    
    def _get_weighted_loss_function(self):
        """Initialize loss function with class weights."""
        if self.class_weights is not None:
            return nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        else:
            return nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def _get_advanced_scheduler(self):
        """Initialize advanced learning rate scheduler."""
        if self.config['scheduler'] == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'] * 10,
                epochs=self.config['epochs'],
                steps_per_epoch=self.config.get('steps_per_epoch', 100),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif self.config['scheduler'] == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
            return ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                patience=8, 
                factor=0.5,
                min_lr=1e-7
            )
    
    def train_epoch_with_advanced_augmentation(self, train_loader, epoch):
        """Train epoch with advanced augmentation including MixUp/CutMix."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        # Get current augmentation strength for progressive augmentation
        current_aug_strength = 'strong'  # default
        if self.aug_scheduler:
            current_aug_strength = self.aug_scheduler.get_strength_for_epoch(epoch)
            # Update dataset augmentation strength if needed
            if hasattr(train_loader.dataset, 'update_epoch'):
                train_loader.dataset.update_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f'Training (Aug: {current_aug_strength})')
        for batch_idx, (data, target, _) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply MixUp/CutMix augmentation
            mixed_targets = None
            if self.mixup_cutmix is not None:
                try:
                    augmented_batch = self.mixup_cutmix((data, target))
                    if len(augmented_batch) == 2 and isinstance(augmented_batch[1], tuple):
                        # MixUp/CutMix was applied
                        data, mixed_targets = augmented_batch
                        target_a, target_b, lam = mixed_targets
                    else:
                        # No augmentation applied
                        data, target = augmented_batch
                except Exception as e:
                    print(f"MixUp/CutMix error: {e}")
                    # Continue with original data
                    pass
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Calculate loss based on whether MixUp/CutMix was applied
            if mixed_targets is not None:
                target_a, target_b, lam = mixed_targets
                loss = lam * self.criterion(outputs, target_a) + (1 - lam) * self.criterion(outputs, target_b)
                # For accuracy calculation, use the primary target
                target_for_acc = target_a
            else:
                loss = self.criterion(outputs, target)
                target_for_acc = target
            
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update scheduler if OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_for_acc.size(0)
            correct += (predicted == target_for_acc).sum().item()
            
            # Store for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target_for_acc.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Aug': current_aug_strength
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate F1 score
        metrics = calculate_metrics(all_targets, all_predictions)
        epoch_f1 = metrics['f1_score']
        
        return epoch_loss, epoch_acc, epoch_f1, current_aug_strength
    
    def validate_epoch_with_enhanced_tta(self, val_loader):
        """Validate with Enhanced Test Time Augmentation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation (Enhanced TTA)')
            for data, target, paths in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.tta_transforms and len(self.tta_transforms) > 0:
                    # Enhanced Test Time Augmentation
                    outputs_list = []
                    
                    # Apply each TTA transform
                    for tta_transform in self.tta_transforms:
                        try:
                            # Convert tensor back to numpy for albumentations
                            batch_outputs = []
                            for i in range(data.size(0)):
                                # Convert single image from tensor to numpy
                                img_tensor = data[i]
                                img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                                
                                # Denormalize
                                mean = np.array([0.485, 0.456, 0.406])
                                std = np.array([0.229, 0.224, 0.225])
                                img_np = img_np * std + mean
                                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                                
                                # Apply TTA transform
                                augmented = tta_transform(image=img_np)
                                aug_tensor = augmented['image'].unsqueeze(0).to(self.device)
                                
                                # Get prediction
                                aug_output = self.model(aug_tensor)
                                batch_outputs.append(aug_output)
                            
                            # Stack batch outputs
                            batch_tensor = torch.cat(batch_outputs, dim=0)
                            outputs_list.append(F.softmax(batch_tensor, dim=1))
                            
                        except Exception as e:
                            print(f"TTA transform error: {e}")
                            # Fallback to original prediction
                            outputs = self.model(data)
                            outputs_list.append(F.softmax(outputs, dim=1))
                            break
                    
                    # Average all TTA predictions
                    if outputs_list:
                        outputs = torch.stack(outputs_list).mean(dim=0)
                        outputs = torch.log(outputs + 1e-8)  # Convert back to logits
                    else:
                        outputs = self.model(data)
                else:
                    # No TTA, just regular prediction
                    outputs = self.model(data)
                
                loss = self.criterion(outputs, target)
                
                # Statistics
                running_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Store for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        return epoch_loss, epoch_acc, metrics
    
    def train_improved(self, train_loader, val_loader):
        """Main improved training loop with enhanced augmentation."""
        print(f"Starting improved training for {self.config['epochs']} epochs...")
        print(f"Model: {self.config['model_type']}")
        print(f"Optimizer: {self.config['optimizer']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Class weights: {self.class_weights}")
        print(f"Augmentation strategy: {self.config.get('augmentation_strength', 'strong')}")
        print(f"MixUp/CutMix: {'Enabled' if self.mixup_cutmix else 'Disabled'}")
        print(f"Progressive augmentation: {'Enabled' if self.aug_scheduler else 'Disabled'}")
        print(f"Enhanced TTA: {'Enabled' if self.tta_transforms else 'Disabled'}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 60)
            
            # Train with advanced augmentation
            train_loss, train_acc, train_f1, aug_strength = self.train_epoch_with_advanced_augmentation(
                train_loader, epoch
            )
            
            # Validate with enhanced TTA
            val_loss, val_acc, val_metrics = self.validate_epoch_with_enhanced_tta(val_loader)
            
            val_f1 = val_metrics['f1_score']
            val_auc = val_metrics.get('auc_score', 0.0)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            self.history['augmentation_strength'].append(aug_strength)
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
            if val_auc > 0:
                print(f"Val AUC: {val_auc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Augmentation Strength: {aug_strength}")
            
            # Calculate overfitting indicator
            train_val_gap = train_acc - val_acc
            if train_val_gap > 10:
                print(f"⚠️  High overfitting detected (gap: {train_val_gap:.1f}%)")
            elif train_val_gap > 5:
                print(f"⚠️  Moderate overfitting detected (gap: {train_val_gap:.1f}%)")
            else:
                print(f"✅ Good generalization (gap: {train_val_gap:.1f}%)")
            
            # Update scheduler (except OneCycleLR which updates per batch)
            if not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
            
            # Early stopping and best model saving based on F1 score
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = os.path.join(
                    self.config['save_dir'], 'best_improved_model.pth'
                )
                self.save_model(self.best_model_path, epoch, val_metrics)
                print(f"🎉 New best model saved! Val F1: {val_f1:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                checkpoint_path = os.path.join(
                    self.config['save_dir'], f'improved_checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_model(checkpoint_path, epoch, val_metrics)
        
        total_time = time.time() - start_time
        print(f"\nImproved training completed in {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"Best validation F1 score: {self.best_val_f1:.4f}")
        
        # Save training history
        self.save_training_history()
        
        # Plot improved training curves
        self.plot_improved_training_curves()
        
        return self.best_model_path
    
    def save_model(self, path, epoch, metrics):
        """Save model checkpoint with improved metadata."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'metrics': metrics,
            'history': self.history,
            'class_weights': self.class_weights,
            'model_architecture': str(self.model)
        }, path)
    
    def save_training_history(self):
        """Save comprehensive training history."""
        history_path = os.path.join(self.config['log_dir'], 'improved_training_history.json')
        
        # Convert tensors to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_json[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump({
                'history': history_json,
                'best_val_f1': float(self.best_val_f1),
                'config': self.config
            }, f, indent=2)
    
    def plot_improved_training_curves(self):
        """Plot comprehensive training curves."""
        plt.figure(figsize=(20, 12))
        
        # Loss curves
        plt.subplot(2, 4, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(2, 4, 2)
        plt.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        plt.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score curves
        plt.subplot(2, 4, 3)
        plt.plot(self.history['train_f1'], label='Train F1', linewidth=2)
        plt.plot(self.history['val_f1'], label='Val F1', linewidth=2)
        plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate curve
        plt.subplot(2, 4, 4)
        plt.plot(self.history['learning_rates'], linewidth=2, color='green')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Training vs Validation Gap
        plt.subplot(2, 4, 5)
        acc_gap = [v - t for v, t in zip(self.history['val_acc'], self.history['train_acc'])]
        plt.plot(acc_gap, linewidth=2, color='red')
        plt.title('Validation - Training Accuracy Gap', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap (%)')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Loss Gap
        plt.subplot(2, 4, 6)
        loss_gap = [v - t for v, t in zip(self.history['val_loss'], self.history['train_loss'])]
        plt.plot(loss_gap, linewidth=2, color='purple')
        plt.title('Validation - Training Loss Gap', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gap')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Model Performance Summary
        plt.subplot(2, 4, 7)
        final_metrics = ['Train Acc', 'Val Acc', 'Train F1', 'Val F1']
        final_values = [
            self.history['train_acc'][-1],
            self.history['val_acc'][-1],
            self.history['train_f1'][-1] * 100,  # Convert to percentage
            self.history['val_f1'][-1] * 100
        ]
        bars = plt.bar(final_metrics, final_values, color=['blue', 'orange', 'green', 'red'])
        plt.title('Final Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score (%)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Training Progress
        plt.subplot(2, 4, 8)
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['val_f1'], 'o-', linewidth=2, markersize=4)
        plt.title(f'Best Val F1: {self.best_val_f1:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation F1 Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 'improved_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {plot_path}")

def main():
    """Main function for improved training."""
    parser = argparse.ArgumentParser(description='Improved Training for Quality Inspection')
    parser.add_argument('--data-dir', default='../../data', help='Path to dataset')
    parser.add_argument('--model-type', default='resnet50', 
                       choices=['resnet50', 'efficientnet', 'vgg16', 'simple'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', default='onecycle', 
                       choices=['plateau', 'cosine', 'onecycle'])
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='MixUp alpha parameter')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, help='CutMix alpha parameter')
    parser.add_argument('--mixup-prob', type=float, default=0.5, help='Probability of applying MixUp/CutMix')
    parser.add_argument('--augmentation-strength', default='strong',
                       choices=['light', 'medium', 'strong', 'progressive'],
                       help='Augmentation strength level')
    parser.add_argument('--use-progressive-aug', action='store_true', default=False,
                       help='Use progressive augmentation (strength increases over epochs)')
    parser.add_argument('--use-mixup-cutmix', action='store_true', default=True,
                       help='Use MixUp and CutMix augmentation')
    parser.add_argument('--use-enhanced-tta', action='store_true', default=True,
                       help='Use Enhanced Test Time Augmentation')
    parser.add_argument('--use-tta', action='store_true', default=True,
                       help='Use Test Time Augmentation')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save-dir', default='../../results/models', help='Save directory')
    parser.add_argument('--log-dir', default='../../results/logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Enhanced configuration with advanced augmentation
    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'num_classes': 2,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'mixup_alpha': args.mixup_alpha,
        'cutmix_alpha': args.cutmix_alpha,
        'mixup_prob': args.mixup_prob,
        'augmentation_strength': args.augmentation_strength,
        'use_progressive_aug': args.use_progressive_aug,
        'use_mixup_cutmix': args.use_mixup_cutmix,
        'use_enhanced_tta': args.use_enhanced_tta,
        'use_tta': args.use_tta,
        'patience': args.patience,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_workers': 4,
        'pretrained': True,
        'save_frequency': 10
    }
    
    print("🚀 IMPROVED TRAINING CONFIGURATION")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Create enhanced data loaders with advanced augmentation
    train_loader, val_loader = get_enhanced_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_advanced_aug=True,
        augmentation_strength=config['augmentation_strength'],
        total_epochs=config['epochs']
    )
    
    config['steps_per_epoch'] = len(train_loader)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize improved trainer
    trainer = ImprovedQualityInspectionTrainer(config)
    
    # Start improved training
    best_model_path = trainer.train_improved(train_loader, val_loader)
    
    print("\n🎉 IMPROVED TRAINING COMPLETED!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation F1 score: {trainer.best_val_f1:.4f}")

if __name__ == "__main__":
    main()