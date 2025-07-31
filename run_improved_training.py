#!/usr/bin/env python3
"""
Run improved training with all fixes applied
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import standard modules (fixed versions)
from src.data.dataset import get_data_loaders
from src.training.train_model import QualityInspectionTrainer

def run_improved_training():
    """Run training with all improvements."""
    
    print("🚀 RUNNING IMPROVED QUALITY INSPECTION TRAINING")
    print("=" * 70)
    
    # Optimized configuration with all fixes
    config = {
        'data_dir': 'data',
        'model_type': 'resnet50',
        'num_classes': 2,
        'epochs': 30,  # Reasonable number with early stopping
        'batch_size': 16,  # Smaller for better generalization
        'learning_rate': 0.0001,  # Lower learning rate
        'weight_decay': 0.01,  # Increased regularization
        'optimizer': 'adam',
        'scheduler': 'warmup_cosine',  # Better scheduler
        'early_stopping_patience': 10,  # Early stopping
        'save_dir': 'results/models',
        'log_dir': 'results/logs',
        'num_workers': 2,  # Reduced for stability
        'pretrained': True,
        'save_frequency': 10
    }
    
    print("📋 OPTIMIZED CONFIGURATION:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("-" * 40)
    
    # Create data loaders with improved augmentation
    print("\n📊 Loading Dataset with Enhanced Augmentation...")
    try:
        train_loader, val_loader = get_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            val_split=0.2
        )
        
        print(f"✅ Training samples: {len(train_loader.dataset)}")
        print(f"✅ Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initialize trainer with all fixes
    print("\n🧠 Initializing Fixed Trainer...")
    trainer = QualityInspectionTrainer(config)
    
    # Show improvements summary
    print("\n🎯 IMPLEMENTED FIXES:")
    print("-" * 50)
    print("✅ Fixed MPS/GPU device selection")
    print("✅ Updated deprecated 'pretrained' parameter")
    print("✅ Enhanced data augmentation with more techniques")
    print("✅ Added early stopping to prevent overfitting")
    print("✅ Improved learning rate scheduling with warmup")
    print("✅ Proper regularization (dropout + weight decay)")
    print("✅ Fixed precision warnings in metrics")
    print("✅ Non-blocking data transfer for performance")
    print("-" * 50)
    
    # Start fixed training
    print("\n🚀 Starting Fixed Training...")
    try:
        best_model_path = trainer.train(train_loader, val_loader)
        
        print("\n🎉 IMPROVED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📁 Best model saved: {best_model_path}")
        print(f"📈 Best validation accuracy: {trainer.best_val_acc:.2f}%")
        print(f"📊 Training history saved: {config['log_dir']}/training_history.json")
        print(f"📈 Training curves saved: {config['log_dir']}/training_curves.png")
        
        if trainer.early_stopping_triggered:
            print("✅ Early stopping prevented overfitting")
        
        # Performance comparison
        print("\n📊 EXPECTED IMPROVEMENTS:")
        print("-" * 40)
        print("Issues Fixed:")
        print("  ❌ Overfitting (Training 87% vs Val 85% → 46%)")
        print("  ❌ Hardware warnings (CPU-only training)")
        print("  ❌ Deprecated parameters")
        print("  ❌ Erratic validation performance")
        print("")
        print("Expected Results:")
        print("  ✅ Stable validation performance")
        print("  ✅ Reduced overfitting gap (<10%)")
        print("  ✅ Faster training with GPU/MPS")
        print("  ✅ Better convergence")
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
                
                print(f"\n📊 FINAL PERFORMANCE:")
                print(f"  Final Training Accuracy:   {final_train_acc:.2f}%")
                print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
                print(f"  Accuracy Gap:              {abs(final_val_acc - final_train_acc):.2f}%")
                
                if abs(final_val_acc - final_train_acc) < 10:
                    print("  ✅ Good generalization (gap < 10%)")
                else:
                    print("  ⚠️  Potential overfitting (gap > 10%)")
        
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