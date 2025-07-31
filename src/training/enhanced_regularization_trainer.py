"""
Enhanced training script with comprehensive regularization to fix overfitting issues
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.enhanced_dataset import get_enhanced_data_loaders, get_tta_transforms
from src.data.advanced_augmentation import MixUpCutMix
from src.models.cnn_model import create_model
from src.utils.metrics import calculate_metrics

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and reducing overconfident predictions."""
    
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Enhanced Label Smoothing with customizable smoothing factor."""
    
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            
        if self.weight is not None:
            # Apply class weights
            true_dist = true_dist * self.weight[target].unsqueeze(1)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class EarlyStoppingWithMultipleCriteria:
    """Enhanced early stopping with multiple criteria for better overfitting detection."""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True,
                 monitor_val_loss=True, monitor_overfitting_gap=True, max_gap=15.0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor_val_loss = monitor_val_loss
        self.monitor_overfitting_gap = monitor_overfitting_gap
        self.max_gap = max_gap
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, val_score, val_loss, train_acc, val_acc, model):
        # Primary criterion: validation score improvement
        score_improved = False
        if self.best_score is None:
            self.best_score = val_score
            score_improved = True
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            score_improved = True
            
        # Secondary criterion: overfitting gap
        overfitting_gap = train_acc - val_acc
        gap_violation = self.monitor_overfitting_gap and overfitting_gap > self.max_gap
        
        if score_improved and not gap_violation:
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience or gap_violation:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
                
        return self.should_stop

class EnhancedRegularizationTrainer:
    """Enhanced trainer with comprehensive regularization techniques to fix overfitting."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize model with enhanced regularization
        self.model = self._create_regularized_model()
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights()
        
        # Initialize optimizer with enhanced settings
        self.optimizer = self._get_enhanced_optimizer()
        
        # Initialize enhanced loss function
        self.criterion = self._get_enhanced_loss_function()
        
        # Initialize enhanced scheduler
        self.scheduler = self._get_enhanced_scheduler()
        
        # Initialize Stochastic Weight Averaging
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_start = config.get('swa_start_epoch', max(config['epochs'] // 2, 10))
        
        # Initialize mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Initialize MixUp/CutMix augmentation with conservative settings
        self.mixup_cutmix = MixUpCutMix(
            mixup_alpha=config.get('mixup_alpha', 0.1),  # More conservative
            cutmix_alpha=config.get('cutmix_alpha', 0.5),  # More conservative
            prob=config.get('mixup_prob', 0.3)  # Lower probability
        ) if config.get('use_mixup_cutmix', True) else None
        
        # Enhanced early stopping
        self.early_stopping = EarlyStoppingWithMultipleCriteria(
            patience=config.get('patience', 8),  # More conservative
            min_delta=config.get('min_delta', 0.001),
            monitor_overfitting_gap=True,
            max_gap=config.get('max_overfitting_gap', 10.0)  # Stop if gap > 10%
        )
        
        # Training history with enhanced tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': [],
            'overfitting_gaps': [],
            'gradient_norms': [],
            'swa_performance': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_path = None
        
        # Overfitting monitoring
        self.overfitting_warnings = deque(maxlen=5)  # Track last 5 epochs
        
        # Enhanced TTA transforms
        self.tta_transforms = get_tta_transforms() if config.get('use_enhanced_tta', True) else None
        
    def _create_regularized_model(self):
        """Create model with enhanced regularization."""
        model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        )
        
        # Enhance classifier with more aggressive regularization
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            in_features = None
            
            # Find input features from the original classifier
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
                    
            if in_features is None:
                # Fallback to backbone feature extraction
                if hasattr(model, 'backbone'):
                    if hasattr(model.backbone, 'fc'):
                        in_features = model.backbone.fc.in_features
                    elif hasattr(model.backbone, 'classifier'):
                        in_features = model.backbone.classifier[-1].in_features
            
            if in_features:
                # Create enhanced classifier with aggressive regularization
                enhanced_classifier = nn.Sequential(
                    nn.Dropout(0.7),  # Higher dropout
                    nn.Linear(in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, self.config['num_classes'])
                )
                
                model.classifier = enhanced_classifier
        
        return model.to(self.device)
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        train_data_path = Path(self.config['data_dir']) / 'train'
        
        class_counts = []
        for class_name in ['ok', 'defective']:
            class_path = train_data_path / class_name
            if class_path.exists():
                count = len(list(class_path.glob('*.jpeg'))) + len(list(class_path.glob('*.jpg')))
                class_counts.append(count)
        
        if len(class_counts) == 2:
            total = sum(class_counts)
            weights = [total / (len(class_counts) * count) for count in class_counts]
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            print(f"Class weights calculated: OK={weights[0]:.3f}, Defective={weights[1]:.3f}")
            return weights_tensor
        
        return None
    
    def _get_enhanced_optimizer(self):
        """Initialize optimizer with enhanced regularization settings."""
        # Separate parameters for different regularization
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        if self.config['optimizer'] == 'adamw':
            return optim.AdamW([
                {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.5, 'weight_decay': self.config['weight_decay']},
                {'params': classifier_params, 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay'] * 2}
            ], betas=(0.9, 0.999), eps=1e-8)
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD([
                {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.5, 'weight_decay': self.config['weight_decay']},
                {'params': classifier_params, 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay'] * 2}
            ], momentum=0.9, nesterov=True)
        else:
            return optim.Adam([
                {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.5, 'weight_decay': self.config['weight_decay']},
                {'params': classifier_params, 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay'] * 2}
            ])
    
    def _get_enhanced_loss_function(self):
        """Initialize enhanced loss function with multiple regularization techniques."""
        loss_type = self.config.get('loss_type', 'label_smoothing')
        
        if loss_type == 'focal':
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 1),
                gamma=self.config.get('focal_gamma', 2),
                weight=self.class_weights
            )
        elif loss_type == 'label_smoothing':
            return LabelSmoothingCrossEntropyLoss(
                num_classes=self.config['num_classes'],
                smoothing=self.config.get('label_smoothing', 0.15),  # Increased smoothing
                weight=self.class_weights
            )
        else:
            return nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.config.get('label_smoothing', 0.15)
            )
    
    def _get_enhanced_scheduler(self):
        """Initialize enhanced learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,  # More aggressive
                factor=0.3,  # Larger reduction
                min_lr=1e-8,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.001
            )
        else:
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,
                factor=0.3,
                min_lr=1e-8
            )
    
    def _init_swa(self, epoch):
        """Initialize Stochastic Weight Averaging."""
        if epoch == self.swa_start and self.swa_model is None:
            print(f"Initializing SWA at epoch {epoch}")
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=self.config['learning_rate'] * 0.1,
                anneal_epochs=5
            )
            return True
        return False
    
    def _update_swa(self, epoch):
        """Update SWA model."""
        if self.swa_model is not None and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            if isinstance(self.swa_scheduler, SWALR):
                self.swa_scheduler.step()
    
    def _apply_l1_regularization(self, model, l1_lambda=1e-5):
        """Apply L1 regularization to model parameters."""
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        return l1_lambda * l1_norm
    
    def train_epoch_with_enhanced_regularization(self, train_loader, epoch):
        """Train epoch with comprehensive regularization techniques."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        gradient_norms = []
        
        # Initialize SWA if needed
        swa_initialized = self._init_swa(epoch)
        if swa_initialized:
            print("🔄 SWA initialized - starting weight averaging")
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target, _) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Apply MixUp/CutMix augmentation with conservative settings
            mixed_targets = None
            if self.mixup_cutmix is not None and np.random.random() < 0.2:  # Lower probability
                try:
                    augmented_batch = self.mixup_cutmix((data, target))
                    if len(augmented_batch) == 2 and isinstance(augmented_batch[1], tuple):
                        data, mixed_targets = augmented_batch
                        target_a, target_b, lam = mixed_targets
                    else:
                        data, target = augmented_batch
                except Exception:
                    # Continue with original data on error
                    pass
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    
                    # Calculate loss
                    if mixed_targets is not None:
                        target_a, target_b, lam = mixed_targets
                        loss = lam * self.criterion(outputs, target_a) + (1 - lam) * self.criterion(outputs, target_b)
                        target_for_acc = target_a
                    else:
                        loss = self.criterion(outputs, target)
                        target_for_acc = target
                    
                    # Add L1 regularization
                    l1_reg = self._apply_l1_regularization(self.model, l1_lambda=1e-6)
                    loss = loss + l1_reg
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping before optimizer step
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # More aggressive clipping
                    gradient_norms.append(grad_norm.item())
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = self.model(data)
                
                if mixed_targets is not None:
                    target_a, target_b, lam = mixed_targets
                    loss = lam * self.criterion(outputs, target_a) + (1 - lam) * self.criterion(outputs, target_b)
                    target_for_acc = target_a
                else:
                    loss = self.criterion(outputs, target)
                    target_for_acc = target
                
                # Add L1 regularization
                l1_reg = self._apply_l1_regularization(self.model, l1_lambda=1e-6)
                loss = loss + l1_reg
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    gradient_norms.append(grad_norm.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += target_for_acc.size(0)
            correct += (predicted == target_for_acc).sum().item()
            
            # Store for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target_for_acc.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'GradNorm': f'{gradient_norms[-1]:.3f}' if gradient_norms else 'N/A'
            })
        
        # Update SWA model
        self._update_swa(epoch)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        epoch_f1 = metrics['f1_score']
        
        # Store gradient norm statistics
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        return epoch_loss, epoch_acc, epoch_f1, avg_grad_norm
    
    def validate_epoch_with_enhanced_monitoring(self, val_loader):
        """Validate with enhanced monitoring and TTA."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target, paths in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Enhanced TTA with conservative settings
                if self.tta_transforms and len(self.tta_transforms) > 0:
                    outputs_list = []
                    
                    # Original prediction
                    outputs = self.model(data)
                    outputs_list.append(F.softmax(outputs, dim=1))
                    
                    # TTA predictions (limited to avoid overfitting)
                    for i, tta_transform in enumerate(self.tta_transforms[:3]):  # Limit TTA
                        try:
                            batch_outputs = []
                            for j in range(min(data.size(0), 8)):  # Limit batch size for TTA
                                img_tensor = data[j]
                                img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                                
                                # Denormalize
                                mean = np.array([0.485, 0.456, 0.406])
                                std = np.array([0.229, 0.224, 0.225])
                                img_np = img_np * std + mean
                                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                                
                                # Apply TTA
                                augmented = tta_transform(image=img_np)
                                aug_tensor = augmented['image'].unsqueeze(0).to(self.device)
                                
                                aug_output = self.model(aug_tensor)
                                batch_outputs.append(aug_output)
                            
                            if batch_outputs:
                                batch_tensor = torch.cat(batch_outputs, dim=0)
                                outputs_list.append(F.softmax(batch_tensor, dim=1))
                                
                        except Exception:
                            break
                    
                    # Average TTA predictions
                    if len(outputs_list) > 1:
                        outputs = torch.stack(outputs_list).mean(dim=0)
                        outputs = torch.log(outputs + 1e-8)
                    else:
                        outputs = self.model(data)
                else:
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
    
    def evaluate_swa_model(self, val_loader):
        """Evaluate SWA model performance."""
        if self.swa_model is None:
            return None
            
        # Update BN statistics for SWA model
        torch.optim.swa_utils.update_bn(val_loader, self.swa_model, device=self.device)
        
        self.swa_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.swa_model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        swa_acc = 100. * correct / total
        swa_metrics = calculate_metrics(all_targets, all_predictions)
        
        return {
            'accuracy': swa_acc,
            'f1_score': swa_metrics['f1_score'],
            'metrics': swa_metrics
        }
    
    def train_with_enhanced_regularization(self, train_loader, val_loader):
        """Main training loop with comprehensive regularization."""
        print(f"Starting Enhanced Regularization Training for {self.config['epochs']} epochs...")
        print(f"Model: {self.config['model_type']}")
        print(f"Optimizer: {self.config['optimizer']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Loss function: {self.config.get('loss_type', 'label_smoothing')}")
        print(f"Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        print(f"SWA start epoch: {self.swa_start}")
        print(f"Early stopping patience: {self.early_stopping.patience}")
        print(f"Max overfitting gap: {self.early_stopping.max_gap}%")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 70)
            
            # Train with enhanced regularization
            train_loss, train_acc, train_f1, avg_grad_norm = self.train_epoch_with_enhanced_regularization(
                train_loader, epoch
            )
            
            # Validate with enhanced monitoring
            val_loss, val_acc, val_metrics = self.validate_epoch_with_enhanced_monitoring(val_loader)
            val_f1 = val_metrics['f1_score']
            val_auc = val_metrics.get('auc_score', 0.0)
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_f1)
            elif not isinstance(self.scheduler, (OneCycleLR, SWALR)):
                self.scheduler.step()
            
            # Calculate overfitting gap
            overfitting_gap = train_acc - val_acc
            self.overfitting_warnings.append(overfitting_gap)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['overfitting_gaps'].append(overfitting_gap)
            self.history['gradient_norms'].append(avg_grad_norm)
            
            # Evaluate SWA model if available
            swa_results = None
            if epoch >= self.swa_start and self.swa_model is not None:
                swa_results = self.evaluate_swa_model(val_loader)
                self.history['swa_performance'].append(swa_results)
                print(f"SWA Model - Acc: {swa_results['accuracy']:.2f}%, F1: {swa_results['f1_score']:.4f}")
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
            if val_auc > 0:
                print(f"Val AUC: {val_auc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Overfitting Gap: {overfitting_gap:.2f}%")
            print(f"Average Gradient Norm: {avg_grad_norm:.4f}")
            
            # Enhanced overfitting monitoring
            if overfitting_gap > 15:
                print(f"🚨 SEVERE overfitting detected (gap: {overfitting_gap:.1f}%)")
            elif overfitting_gap > 10:
                print(f"⚠️  HIGH overfitting detected (gap: {overfitting_gap:.1f}%)")
            elif overfitting_gap > 5:
                print(f"⚠️  Moderate overfitting detected (gap: {overfitting_gap:.1f}%)")
            else:
                print(f"✅ Good generalization (gap: {overfitting_gap:.1f}%)")
            
            # Check for consistent overfitting
            if len(self.overfitting_warnings) >= 3:
                recent_gaps = list(self.overfitting_warnings)[-3:]
                if all(gap > 8 for gap in recent_gaps):
                    print("🚨 CONSISTENT overfitting detected across last 3 epochs!")
            
            # Save best model based on validation F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = os.path.join(
                    self.config['save_dir'], 'best_regularized_model.pth'
                )
                self.save_model(self.best_model_path, epoch, val_metrics)
                print(f"🎉 New best model saved! Val F1: {val_f1:.4f}")
            
            # Enhanced early stopping check
            should_stop = self.early_stopping(epoch, val_f1, val_loss, train_acc, val_acc, self.model)
            if should_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                if self.early_stopping.best_weights:
                    print(f"Restored best weights from epoch {self.early_stopping.best_epoch + 1}")
                break
            
            # Save SWA model if available and performing well
            if swa_results and swa_results['f1_score'] > val_f1:
                swa_model_path = os.path.join(
                    self.config['save_dir'], 'best_swa_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.swa_model.state_dict(),
                    'swa_results': swa_results,
                    'config': self.config
                }, swa_model_path)
                print(f"🌟 SWA model outperforming regular model! Saved to: {swa_model_path}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                checkpoint_path = os.path.join(
                    self.config['save_dir'], f'regularized_checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_model(checkpoint_path, epoch, val_metrics)
        
        total_time = time.time() - start_time
        print(f"\nEnhanced Regularization Training completed in {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"Best validation F1 score: {self.best_val_f1:.4f}")
        
        # Final SWA evaluation
        if self.swa_model is not None:
            final_swa_results = self.evaluate_swa_model(val_loader)
            print(f"Final SWA Model F1: {final_swa_results['f1_score']:.4f}")
            
            if final_swa_results['f1_score'] > self.best_val_f1:
                print("🌟 SWA model achieved better performance than best regular model!")
                self.best_model_path = os.path.join(
                    self.config['save_dir'], 'final_best_swa_model.pth'
                )
                torch.save({
                    'model_state_dict': self.swa_model.state_dict(),
                    'swa_results': final_swa_results,
                    'config': self.config,
                    'training_completed': True
                }, self.best_model_path)
        
        # Save training history
        self.save_training_history()
        
        # Plot enhanced training curves
        self.plot_enhanced_training_curves()
        
        return self.best_model_path
    
    def save_model(self, path, epoch, metrics):
        """Save model checkpoint with enhanced metadata."""
        checkpoint = {
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
        }
        
        # Add SWA model if available
        if self.swa_model is not None:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
            checkpoint['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict() if self.swa_scheduler else None
        
        # Add mixed precision scaler state
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def save_training_history(self):
        """Save comprehensive training history with regularization metrics."""
        history_path = os.path.join(self.config['log_dir'], 'enhanced_regularization_history.json')
        
        # Convert tensors to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                if key == 'swa_performance':
                    # Handle SWA performance dict
                    history_json[key] = value
                else:
                    history_json[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                history_json[key] = value
        
        # Add regularization summary
        regularization_summary = {
            'max_overfitting_gap': max(self.history['overfitting_gaps']) if self.history['overfitting_gaps'] else 0,
            'avg_overfitting_gap': np.mean(self.history['overfitting_gaps']) if self.history['overfitting_gaps'] else 0,
            'final_overfitting_gap': self.history['overfitting_gaps'][-1] if self.history['overfitting_gaps'] else 0,
            'avg_gradient_norm': np.mean(self.history['gradient_norms']) if self.history['gradient_norms'] else 0,
            'early_stopping_triggered': self.early_stopping.should_stop,
            'early_stopping_epoch': self.early_stopping.best_epoch if self.early_stopping.should_stop else None,
            'swa_enabled': self.swa_model is not None,
            'mixed_precision': self.use_amp,
            'gradient_accumulation': self.accumulation_steps > 1
        }
        
        with open(history_path, 'w') as f:
            json.dump({
                'history': history_json,
                'best_val_f1': float(self.best_val_f1),
                'config': self.config,
                'regularization_summary': regularization_summary
            }, f, indent=2)
        
        print(f"Enhanced training history saved to: {history_path}")
    
    def plot_enhanced_training_curves(self):
        """Plot comprehensive training curves with regularization metrics."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping plot generation.")
            return
            
        plt.figure(figsize=(24, 16))
        
        # Loss curves
        plt.subplot(3, 4, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(3, 4, 2)
        plt.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        plt.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score curves
        plt.subplot(3, 4, 3)
        plt.plot(self.history['train_f1'], label='Train F1', linewidth=2)
        plt.plot(self.history['val_f1'], label='Val F1', linewidth=2)
        plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate curve
        plt.subplot(3, 4, 4)
        plt.plot(self.history['learning_rates'], linewidth=2, color='green')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Overfitting gap
        plt.subplot(3, 4, 5)
        plt.plot(self.history['overfitting_gaps'], linewidth=2, color='red')
        plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='High Overfitting (10%)')
        plt.axhline(y=5, color='yellow', linestyle='--', alpha=0.7, label='Moderate Overfitting (5%)')
        plt.title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gradient norms
        plt.subplot(3, 4, 6)
        plt.plot(self.history['gradient_norms'], linewidth=2, color='purple')
        plt.title('Average Gradient Norms', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        
        # Loss Gap
        plt.subplot(3, 4, 7)
        if len(self.history['val_loss']) > 0 and len(self.history['train_loss']) > 0:
            loss_gap = [v - t for v, t in zip(self.history['val_loss'], self.history['train_loss'])]
            plt.plot(loss_gap, linewidth=2, color='brown')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Validation - Training Loss Gap', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gap')
        plt.grid(True, alpha=0.3)
        
        # SWA Performance (if available)
        plt.subplot(3, 4, 8)
        if self.history['swa_performance']:
            swa_f1_scores = [perf['f1_score'] for perf in self.history['swa_performance'] if perf]
            if swa_f1_scores:
                epochs_with_swa = range(self.swa_start, self.swa_start + len(swa_f1_scores))
                plt.plot(epochs_with_swa, swa_f1_scores, 'o-', linewidth=2, markersize=6, color='gold')
                plt.title('SWA Model F1 Performance', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('SWA F1 Score')
                plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No SWA Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('SWA Model F1 Performance', fontsize=14, fontweight='bold')
        
        # Model Performance Summary
        plt.subplot(3, 4, 9)
        final_metrics = ['Train Acc', 'Val Acc', 'Train F1', 'Val F1']
        final_values = [
            self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            self.history['train_f1'][-1] * 100 if self.history['train_f1'] else 0,
            self.history['val_f1'][-1] * 100 if self.history['val_f1'] else 0
        ]
        bars = plt.bar(final_metrics, final_values, color=['blue', 'orange', 'green', 'red'])
        plt.title('Final Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score (%)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Regularization Summary
        plt.subplot(3, 4, 10)
        reg_metrics = ['Max Gap', 'Avg Gap', 'Final Gap', 'Avg Grad Norm']
        reg_values = [
            max(self.history['overfitting_gaps']) if self.history['overfitting_gaps'] else 0,
            np.mean(self.history['overfitting_gaps']) if self.history['overfitting_gaps'] else 0,
            self.history['overfitting_gaps'][-1] if self.history['overfitting_gaps'] else 0,
            np.mean(self.history['gradient_norms']) if self.history['gradient_norms'] else 0
        ]
        bars = plt.bar(reg_metrics, reg_values, color=['red', 'orange', 'yellow', 'purple'])
        plt.title('Regularization Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, reg_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Training Progress Timeline
        plt.subplot(3, 4, 11)
        epochs = range(1, len(self.history['val_f1']) + 1) if self.history['val_f1'] else []
        if epochs:
            plt.plot(epochs, self.history['val_f1'], 'o-', linewidth=2, markersize=4)
            plt.axhline(y=self.best_val_f1, color='red', linestyle='--', alpha=0.7, label=f'Best: {self.best_val_f1:.3f}')
        plt.title('Validation F1 Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting Detection Timeline
        plt.subplot(3, 4, 12)
        if self.history['overfitting_gaps']:
            colors = ['red' if gap > 10 else 'orange' if gap > 5 else 'green'
                     for gap in self.history['overfitting_gaps']]
            plt.bar(range(len(self.history['overfitting_gaps'])),
                   self.history['overfitting_gaps'], color=colors, alpha=0.7)
            plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='High (10%)')
            plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate (5%)')
        plt.title('Overfitting Detection Timeline', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Overfitting Gap (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 'enhanced_regularization_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Enhanced training curves saved to: {plot_path}")

def main():
    """Main function for enhanced regularization training."""
    parser = argparse.ArgumentParser(description='Enhanced Regularization Training for Quality Inspection')
    parser.add_argument('--data-dir', default='../../data', help='Path to dataset')
    parser.add_argument('--model-type', default='resnet50',
                       choices=['resnet50', 'efficientnet', 'vgg16', 'simple'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'cosine', 'onecycle'])
    parser.add_argument('--loss-type', default='label_smoothing',
                       choices=['focal', 'label_smoothing', 'cross_entropy'])
    parser.add_argument('--label-smoothing', type=float, default=0.15, help='Label smoothing factor')
    parser.add_argument('--focal-alpha', type=float, default=1, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2, help='Focal loss gamma')
    parser.add_argument('--mixup-alpha', type=float, default=0.1, help='MixUp alpha parameter')
    parser.add_argument('--cutmix-alpha', type=float, default=0.5, help='CutMix alpha parameter')
    parser.add_argument('--mixup-prob', type=float, default=0.3, help='Probability of applying MixUp/CutMix')
    parser.add_argument('--use-mixup-cutmix', action='store_true', default=True,
                       help='Use MixUp and CutMix augmentation')
    parser.add_argument('--use-enhanced-tta', action='store_true', default=True,
                       help='Use Enhanced Test Time Augmentation')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--max-overfitting-gap', type=float, default=10.0,
                       help='Maximum allowed overfitting gap (%)')
    parser.add_argument('--swa-start-epoch', type=int, default=15,
                       help='Epoch to start Stochastic Weight Averaging')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--save-dir', default='../../results/models', help='Save directory')
    parser.add_argument('--log-dir', default='../../results/logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Enhanced configuration with comprehensive regularization
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
        'loss_type': args.loss_type,
        'label_smoothing': args.label_smoothing,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'mixup_alpha': args.mixup_alpha,
        'cutmix_alpha': args.cutmix_alpha,
        'mixup_prob': args.mixup_prob,
        'use_mixup_cutmix': args.use_mixup_cutmix,
        'use_enhanced_tta': args.use_enhanced_tta,
        'patience': args.patience,
        'max_overfitting_gap': args.max_overfitting_gap,
        'swa_start_epoch': args.swa_start_epoch,
        'use_amp': args.use_amp,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_workers': 4,
        'pretrained': True,
        'save_frequency': 10
    }
    
    print("🚀 ENHANCED REGULARIZATION TRAINING CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Create enhanced data loaders
    train_loader, val_loader = get_enhanced_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_advanced_aug=True,
        augmentation_strength='strong',
        total_epochs=config['epochs']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize enhanced regularization trainer
    trainer = EnhancedRegularizationTrainer(config)
    
    # Start enhanced training
    best_model_path = trainer.train_with_enhanced_regularization(train_loader, val_loader)
    
    print("\n🎉 ENHANCED REGULARIZATION TRAINING COMPLETED!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation F1 score: {trainer.best_val_f1:.4f}")

if __name__ == "__main__":
    main()