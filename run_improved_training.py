#!/usr/bin/env python3
"""
Run improved training with optimized hyperparameters
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import enhanced modules
from src.data.enhanced_dataset import get_enhanced_data_loaders
from src.training.improved_trainer import ImprovedQualityInspectionTrainer

def run_improved_training():
    """Run training with all improvements."""
    
    print("🚀 RUNNING IMPROVED QUALITY INSPECTION TRAINING")
    print("=" * 70)
    
    # Optimized configuration based on analysis
    config = {
        'data_dir': 'data',
        'model_type': 'resnet50',  # Better than simple CNN
        'num_classes': 2,
        'epochs': 25,  # Increased from 3
        'batch_size': 24,  # Slightly smaller for stability
        'learning_rate': 0.0008,  # Optimized learning rate
        'weight_decay': 5e-4,  # Increased regularization
        'optimizer': 'adamw',  # Better optimizer
        'scheduler': 'onecycle',  # Advanced scheduler
        'mixup_alpha': 0.3,  # MixUp augmentation
        'use_tta': True,  # Test Time Augmentation
        'patience': 12,  # Early stopping
        'save_dir': 'results/models',
        'log_dir': 'results/logs',
        'num_workers': 4,
        'pretrained': True,
        'save_frequency': 5
    }
    
    print("📋 OPTIMIZED CONFIGURATION:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("-" * 40)
    
    # Create enhanced data loaders
    print("\n📊 Loading Enhanced Dataset...")
    try:
        train_loader, val_loader = get_enhanced_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            val_split=0.2,
            use_advanced_aug=True,
            use_stratified_split=True
        )
        
        config['steps_per_epoch'] = len(train_loader)
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initialize improved trainer
    print("\n🧠 Initializing Improved Trainer...")
    trainer = ImprovedQualityInspectionTrainer(config)
    
    # Show improvements summary
    print("\n🎯 IMPLEMENTED IMPROVEMENTS:")
    print("-" * 50)
    print("✅ Advanced Data Augmentation (Albumentations)")
    print("✅ Class Balancing with Weighted Loss")
    print("✅ MixUp Data Augmentation")
    print("✅ Test Time Augmentation (TTA)")
    print("✅ AdamW Optimizer with Weight Decay")
    print("✅ OneCycle Learning Rate Scheduler")
    print("✅ Improved Model Architecture")
    print("✅ Label Smoothing")
    print("✅ Gradient Clipping")
    print("✅ Early Stopping")
    print("✅ Stratified Data Splitting")
    print("-" * 50)
    
    # Start improved training
    print("\n🚀 Starting Improved Training...")
    try:
        best_model_path = trainer.train_improved(train_loader, val_loader)
        
        print("\n🎉 IMPROVED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📁 Best model saved: {best_model_path}")
        print(f"📈 Best validation F1: {trainer.best_val_f1:.4f}")
        print(f"📊 Training history saved: {config['log_dir']}/improved_training_history.json")
        print(f"📈 Training curves saved: {config['log_dir']}/improved_training_curves.png")
        
        # Performance comparison
        print("\n📊 EXPECTED IMPROVEMENTS:")
        print("-" * 40)
        print("Previous Performance:")
        print("  Training Accuracy:   61.69%")
        print("  Validation Accuracy: 77.01%")
        print("  F1 Score:           0.77")
        print("  AUC Score:          0.77")
        print("")
        print("Target Improvements:")
        print("  Training Accuracy:   75-85%")
        print("  Validation Accuracy: 85-95%") 
        print("  F1 Score:           0.85-0.95")
        print("  AUC Score:          0.85-0.95")
        print("-" * 40)
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_improved_model(model_path):
    """Test the improved model performance."""
    if not model_path or not os.path.exists(model_path):
        print("❌ Model file not found")
        return
    
    print(f"\n🧪 TESTING IMPROVED MODEL")
    print("=" * 50)
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("✅ Model loaded successfully")
        print(f"📈 Best F1 Score: {checkpoint.get('best_val_f1', 'Unknown'):.4f}")
        print(f"🔄 Training Epochs: {checkpoint.get('epoch', 'Unknown') + 1}")
        
        # Show final metrics
        if 'history' in checkpoint:
            history = checkpoint['history']
            if history['val_acc']:
                final_val_acc = history['val_acc'][-1]
                final_train_acc = history['train_acc'][-1]
                final_val_f1 = history['val_f1'][-1]
                
                print(f"\n📊 FINAL PERFORMANCE:")
                print(f"  Final Training Accuracy:   {final_train_acc:.2f}%")
                print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
                print(f"  Final Validation F1:       {final_val_f1:.4f}")
                print(f"  Accuracy Gap:              {final_val_acc - final_train_acc:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Quality Inspection AI - Improved Training Pipeline")
    print("=" * 70)
    
    # Check if data exists
    if not os.path.exists('data/train'):
        print("❌ Dataset not found!")
        print("Please run: python prepare_data.py")
        return
    
    # Run improved training
    model_path = run_improved_training()
    
    # Test the improved model
    if model_path:
        test_improved_model(model_path)
        
        print("\n🎯 NEXT STEPS:")
        print("-" * 30)
        print("1. Compare results with previous training")
        print("2. Run evaluation: python main.py --mode evaluate")
        print("3. Generate explanations: python main.py --mode explain")
        print("4. Launch dashboard: streamlit run dashboard/app.py")
    
    print("\n✨ Improved training pipeline completed!")

if __name__ == "__main__":
    main()