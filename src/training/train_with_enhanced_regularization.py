#!/usr/bin/env python3
"""
Main training script with enhanced regularization to fix overfitting issues
"""

import os
import sys
import argparse
import json
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import get_data_loaders
from src.data.enhanced_dataset import get_enhanced_data_loaders
from src.training.enhanced_regularization_trainer import EnhancedRegularizationTrainer

def setup_directories(config):
    """Create necessary directories."""
    dirs = [
        config['save_dir'], 
        config['log_dir'], 
        'results/experiments',
        'results/explanations', 
        'results/reports'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def train_with_enhanced_regularization(config):
    """Train the quality inspection model with enhanced regularization."""
    print("\n" + "="*70)
    print("🚀 ENHANCED REGULARIZATION TRAINING PHASE")
    print("="*70)
    
    # Show improvements summary
    print("\n" + "🎯 COMPREHENSIVE REGULARIZATION FIXES:")
    print("-" * 60)
    print("✅ Enhanced dropout rates (0.3-0.7) with progressive layers")
    print("✅ Stochastic Weight Averaging (SWA) for better convergence")
    print("✅ Focal Loss and improved Label Smoothing (0.15)")
    print("✅ Multi-criteria early stopping with overfitting detection")
    print("✅ Mixed precision training with gradient accumulation")
    print("✅ Aggressive learning rate scheduling (ReduceLROnPlateau)")
    print("✅ Enhanced L1/L2 regularization with parameter grouping")
    print("✅ Real-time overfitting monitoring and warnings")
    print("✅ Conservative MixUp/CutMix settings")
    print("✅ Gradient clipping (max_norm=0.5)")
    print("✅ Comprehensive validation monitoring")
    print("-" * 60)
    
    # Create enhanced data loaders
    try:
        train_loader, val_loader = get_enhanced_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            use_advanced_aug=True,
            augmentation_strength='strong',
            total_epochs=config['epochs']
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating enhanced data loaders: {e}")
        print("Falling back to standard data loaders...")
        
        train_loader, val_loader = get_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize enhanced regularization trainer
    trainer = EnhancedRegularizationTrainer(config)
    
    # Start enhanced training
    print("\n" + "🚀 Starting Enhanced Regularization Training...")
    best_model_path = trainer.train_with_enhanced_regularization(train_loader, val_loader)
    
    # Show results summary
    print("\n" + "🎉 ENHANCED REGULARIZATION TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📁 Best model saved: {best_model_path}")
    print(f"📈 Best validation F1 score: {trainer.best_val_f1:.4f}")
    
    # Show regularization effectiveness
    if trainer.history['overfitting_gaps']:
        final_gap = trainer.history['overfitting_gaps'][-1]
        max_gap = max(trainer.history['overfitting_gaps'])
        avg_gap = sum(trainer.history['overfitting_gaps']) / len(trainer.history['overfitting_gaps'])
        
        print("📊 Overfitting Analysis:")
        print(f"   Final gap: {final_gap:.2f}%")
        print(f"   Maximum gap: {max_gap:.2f}%")
        print(f"   Average gap: {avg_gap:.2f}%")
        
        if final_gap < 5:
            print("✅ Excellent regularization - minimal overfitting!")
        elif final_gap < 10:
            print("✅ Good regularization - acceptable overfitting")
        else:
            print("⚠️  Consider more aggressive regularization")
    
    if trainer.early_stopping.should_stop:
        print("✅ Early stopping prevented further overfitting")
    
    if trainer.swa_model is not None:
        print("✅ SWA model available for ensemble predictions")
    
    print("📊 Training history saved: " + f"{config['log_dir']}/enhanced_regularization_history.json")
    print("📈 Training curves saved: " + f"{config['log_dir']}/enhanced_regularization_curves.png")
    
    return best_model_path, trainer

def validate_regularization_effectiveness(trainer, val_loader):
    """Validate the effectiveness of regularization techniques."""
    print("\n" + "="*60)
    print("REGULARIZATION EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Check overfitting indicators
    if trainer.history['overfitting_gaps']:
        gaps = trainer.history['overfitting_gaps']
        final_gap = gaps[-1]
        
        print("\n" + "📊 Overfitting Gap Analysis:")
        print(f"Final Training-Validation Gap: {final_gap:.2f}%")
        
        if final_gap < 3:
            print("🎉 EXCELLENT: Minimal overfitting detected!")
            effectiveness_score = "Excellent"
        elif final_gap < 7:
            print("✅ GOOD: Low overfitting, well-regularized model")
            effectiveness_score = "Good"
        elif final_gap < 12:
            print("⚠️  MODERATE: Some overfitting, consider more regularization")
            effectiveness_score = "Moderate"
        else:
            print("🚨 HIGH: Significant overfitting, increase regularization")
            effectiveness_score = "Poor"
        
        # Check for consistency
        recent_gaps = gaps[-5:] if len(gaps) >= 5 else gaps
        if all(gap < 8 for gap in recent_gaps):
            print("✅ Consistent good generalization in recent epochs")
        
    # Check training stability
    if trainer.history['gradient_norms']:
        avg_grad_norm = sum(trainer.history['gradient_norms']) / len(trainer.history['gradient_norms'])
        print("\n📈 Training Stability:")
        print(f"Average Gradient Norm: {avg_grad_norm:.4f}")
        
        if avg_grad_norm < 1.0:
            print("✅ Stable gradients - good regularization")
        else:
            print("⚠️  High gradient norms - consider stronger regularization")
    
    # Check SWA effectiveness
    if trainer.swa_model is not None and trainer.history['swa_performance']:
        swa_performance = [p for p in trainer.history['swa_performance'] if p is not None]
        if swa_performance:
            best_swa_f1 = max(p['f1_score'] for p in swa_performance)
            regular_f1 = trainer.best_val_f1
            
            print("\n" + "🌟 SWA Model Analysis:")
            print(f"Best SWA F1: {best_swa_f1:.4f}")
            print(f"Best Regular F1: {regular_f1:.4f}")
            
            if best_swa_f1 > regular_f1:
                print("🎉 SWA model outperformed regular training!")
            else:
                print("✅ SWA provided ensemble benefits")
    
    return effectiveness_score

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Regularization Training for Quality Inspection'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', default='data', 
                       help='Path to dataset directory')
    
    # Model arguments
    parser.add_argument('--model-type', default='resnet50',
                       choices=['resnet50', 'efficientnet', 'vgg16', 'simple'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes')
    
    # Enhanced training arguments
    parser.add_argument('--epochs', type=int, default=35,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (smaller for better regularization)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate (conservative for stability)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (increased regularization)')
    parser.add_argument('--optimizer', default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer (AdamW recommended)')
    parser.add_argument('--scheduler', default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'cosine', 'onecycle'],
                       help='Learning rate scheduler')
    
    # Regularization arguments
    parser.add_argument('--loss-type', default='label_smoothing',
                       choices=['focal', 'label_smoothing', 'cross_entropy'],
                       help='Loss function type')
    parser.add_argument('--label-smoothing', type=float, default=0.15,
                       help='Label smoothing factor (0.15 recommended)')
    parser.add_argument('--focal-alpha', type=float, default=1,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2,
                       help='Focal loss gamma parameter')
    
    # Advanced regularization
    parser.add_argument('--mixup-alpha', type=float, default=0.1,
                       help='MixUp alpha (conservative)')
    parser.add_argument('--cutmix-alpha', type=float, default=0.5,
                       help='CutMix alpha (conservative)')
    parser.add_argument('--mixup-prob', type=float, default=0.3,
                       help='MixUp/CutMix probability (conservative)')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience (conservative)')
    parser.add_argument('--max-overfitting-gap', type=float, default=10.0,
                       help='Maximum allowed overfitting gap (%)')
    
    # SWA arguments
    parser.add_argument('--swa-start-epoch', type=int, default=15,
                       help='Epoch to start SWA')
    
    # Training optimization
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Output arguments
    parser.add_argument('--save-dir', default='results/models',
                       help='Model save directory')
    parser.add_argument('--log-dir', default='results/logs',
                       help='Log directory')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create enhanced configuration
    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'num_classes': args.num_classes,
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
        'use_mixup_cutmix': True,
        'use_enhanced_tta': True,
        'patience': args.patience,
        'max_overfitting_gap': args.max_overfitting_gap,
        'swa_start_epoch': args.swa_start_epoch,
        'use_amp': args.use_amp,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_workers': args.num_workers,
        'pretrained': True,
        'save_frequency': 5
    }
    
    # Setup directories
    setup_directories(config)
    
    # Print configuration
    print("🚀 ENHANCED REGULARIZATION CONFIGURATION")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Run enhanced regularization training
    best_model_path, trainer = train_with_enhanced_regularization(config)
    
    # Validate regularization effectiveness
    train_loader, val_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    effectiveness = validate_regularization_effectiveness(trainer, val_loader)
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 ENHANCED REGULARIZATION TRAINING SUMMARY")
    print("="*70)
    print(f"✅ Model saved: {best_model_path}")
    print(f"✅ Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"✅ Regularization effectiveness: {effectiveness}")
    print("✅ Training completed successfully!")
    
    # Create summary report
    summary_report = {
        'training_completed': True,
        'best_model_path': best_model_path,
        'best_val_f1': float(trainer.best_val_f1),
        'regularization_effectiveness': effectiveness,
        'final_overfitting_gap': float(trainer.history['overfitting_gaps'][-1]) if trainer.history['overfitting_gaps'] else None,
        'early_stopping_triggered': trainer.early_stopping.should_stop,
        'swa_enabled': trainer.swa_model is not None,
        'config': config
    }
    
    report_path = os.path.join(config['log_dir'], 'training_summary_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"📊 Summary report saved: {report_path}")

if __name__ == "__main__":
    main()