#!/usr/bin/env python3
"""
Main execution script for Explainable AI Quality Inspection
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import get_data_loaders, QualityInspectionDataset
from src.training.train_model import QualityInspectionTrainer
from src.explainability.explain_model import ModelExplainer
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.metrics import print_metrics_summary

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'results/models', 'results/logs', 'results/experiments',
        'results/explanations', 'results/reports'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_dataset(dataset_name='casting', data_dir='data'):
    """Download and setup dataset."""
    import urllib.request
    import zipfile
    import shutil
    from pathlib import Path
    
    print(f"🔽 Downloading {dataset_name} dataset...")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if dataset_name == 'casting':
            print("📦 Downloading Casting Product Image Dataset...")
            
            # Try Kaggle API first (most reliable)
            try:
                import kaggle
                print("🔑 Kaggle API found, downloading dataset...")
                
                # Download using Kaggle API
                kaggle.api.dataset_download_files(
                    'ravirajsinh45/real-life-industrial-dataset-of-casting-product',
                    path=data_dir,
                    unzip=True
                )
                
                # Verify download and structure
                if verify_casting_dataset_structure(data_dir):
                    print("✅ Kaggle dataset downloaded and verified successfully!")
                    return True
                else:
                    print("⚠️ Dataset structure verification failed, attempting alternative download...")
                    
            except ImportError:
                print("💡 Kaggle API not found, attempting direct download...")
            except Exception as e:
                print(f"❌ Kaggle API error: {e}, attempting alternative download...")
            
            # Alternative: Direct download from mirror or GitHub release
            try:
                print("🔄 Attempting direct download from alternative source...")
                
                # Try to download from a reliable source (you may need to find/host the dataset)
                # For now, provide clear instructions for manual download
                print("📋 Please download the dataset manually:")
                print("   1. Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
                print("   2. Click 'Download' button (requires Kaggle account)")
                print("   3. Extract the downloaded zip file")
                print("   4. Copy the contents to your 'data/' directory")
                print("   5. Ensure this structure:")
                print("      data/")
                print("      ├── train/")
                print("      │   ├── defective/")
                print("      │   └── ok/")
                print("      └── test/")
                print("          ├── defective/")
                print("          └── ok/")
                
                print("\n💡 Alternative: Install Kaggle API for automatic download:")
                print("   pip install kaggle")
                print("   # Get API token from kaggle.com/account")
                print("   # Place kaggle.json in ~/.kaggle/")
                
                return False
                
        elif dataset_name == 'mvtec':
            print("📦 MVTec Anomaly Detection Dataset...")
            print("   Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad")
            print("   Manual download required due to license agreement.")
            return False
            
        elif dataset_name == 'neu':
            print("📦 NEU Surface Defect Dataset...")
            print("   Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
            print("   Manual download required.")
            return False
            
        else:
            print(f"❌ Dataset '{dataset_name}' not supported.")
            print("   Supported datasets: casting, mvtec, neu")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def verify_casting_dataset_structure(data_dir):
    """Verify that the casting dataset has the correct structure."""
    from pathlib import Path
    
    data_path = Path(data_dir)
    required_dirs = [
        data_path / 'train' / 'ok',
        data_path / 'train' / 'defective',
        data_path / 'test' / 'ok',
        data_path / 'test' / 'defective'
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
        
        # Check if directory has images
        image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.jpeg')) + list(dir_path.glob('*.png'))
        if len(image_files) == 0:
            print(f"❌ No images found in: {dir_path}")
            return False
        
        print(f"✅ Found {len(image_files)} images in {dir_path}")
    
    return True

def install_kaggle_api():
    """Install Kaggle API if not present."""
    import subprocess
    import sys
    
    try:
        print("📦 Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle API installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Kaggle API: {e}")
        return False

def create_dummy_dataset(data_dir='data'):
    """Create a dummy dataset for testing when real data is not available."""
    import numpy as np
    from PIL import Image
    from pathlib import Path
    
    print("🎯 Creating dummy dataset for testing...")
    
    data_path = Path(data_dir)
    
    # Create directory structure
    dirs = [
        data_path / 'train' / 'ok',
        data_path / 'train' / 'defective',
        data_path / 'val' / 'ok',
        data_path / 'val' / 'defective',
        data_path / 'test' / 'ok',
        data_path / 'test' / 'defective'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy images
    np.random.seed(42)  # For reproducible dummy data
    
    splits = {
        'train': {'ok': 100, 'defective': 100},
        'val': {'ok': 20, 'defective': 20},
        'test': {'ok': 30, 'defective': 30}
    }
    
    for split, classes in splits.items():
        for class_name, count in classes.items():
            class_dir = data_path / split / class_name
            
            for i in range(count):
                # Create dummy image
                if class_name == 'ok':
                    # Mostly uniform gray image (simulating good product)
                    img_array = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
                else:
                    # Add some "defects" - darker spots and noise
                    img_array = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
                    # Add random dark spots (simulating defects)
                    for _ in range(np.random.randint(3, 8)):
                        x, y = np.random.randint(0, 200, 2)
                        img_array[x:x+24, y:y+24] = np.random.randint(20, 60, (24, 24, 3))
                
                # Save image
                img = Image.fromarray(img_array)
                img_path = class_dir / f'{class_name}_{i:04d}.jpg'
                img.save(img_path)
    
    print(f"✅ Created dummy dataset with {sum(sum(classes.values()) for classes in splits.values())} images")
    print(f"   Structure: {data_dir}/{{train,val,test}}/{{ok,defective}}/")
    return True

def train_model(config):
    """Train the quality inspection model with optimizations."""
    print("\n" + "="*60)
    print("🚀 OPTIMIZED TRAINING PHASE")
    print("="*60)
    
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
    
    # Create data loaders
    try:
        train_loader, val_loader = get_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Using dummy dataset for demonstration...")
        
        from src.data.dataset import DummyDataset
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = DummyDataset(size=1000, transform=transform)
        val_dataset = DummyDataset(size=200, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    trainer = QualityInspectionTrainer(config)
    
    # Start training with all improvements
    print("\n🚀 Starting Optimized Training...")
    best_model_path = trainer.train(train_loader, val_loader)
    
    # Show results summary
    print("\n🎉 OPTIMIZED TRAINING COMPLETED!")
    print("=" * 60)
    print(f"📁 Best model saved: {best_model_path}")
    print(f"📈 Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    if hasattr(trainer, 'early_stopping_triggered') and trainer.early_stopping_triggered:
        print("✅ Early stopping prevented overfitting")
    
    print(f"📊 Training history saved: {config['log_dir']}/training_history.json")
    print(f"📈 Training curves saved: {config['log_dir']}/training_curves.png")
    
    return best_model_path

def evaluate_model(model_path, data_dir, config):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        model_type=config['model_type'],
        num_classes=config['num_classes']
    )
    
    # Load test data
    try:
        test_dataset = QualityInspectionDataset(
            data_dir, split='test'
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    except (FileNotFoundError, OSError):
        # Use validation data if test not available
        _, test_loader = get_data_loaders(data_dir, batch_size=32)
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    
    # Print results
    print_metrics_summary(results['metrics'], class_names=['OK', 'Defective'])
    
    return results

def explain_predictions(model_path, data_dir, config, num_samples=5):
    """Generate explanations for sample predictions."""
    print("\n" + "="*60)
    print("EXPLAINABILITY PHASE")
    print("="*60)
    
    # Create explainer
    explainer = ModelExplainer(
        model_path=model_path,
        model_type=config['model_type'],
        num_classes=config['num_classes']
    )
    
    # Load some test samples
    try:
        test_dataset = QualityInspectionDataset(data_dir, split='test')
        
        # Explain a few samples
        for i in range(min(num_samples, len(test_dataset))):
            sample_path = test_dataset.samples[i][0]
            print(f"\nExplaining sample {i+1}: {os.path.basename(sample_path)}")
            
            save_path = f"results/explanations/explanation_sample_{i+1}.png"
            
            try:
                explainer.explain_image(
                    sample_path,
                    methods=['lime', 'integrated_gradients', 'gradcam'],
                    save_path=save_path
                )
                print(f"Explanation saved to: {save_path}")
            except Exception as e:
                print(f"Error generating explanation: {e}")
    
    except Exception as e:
        print(f"Error in explanation phase: {e}")
        print("Please make sure test data is available and model is properly trained.")

def run_complete_pipeline(config):
    """Run the complete pipeline from data download to explanation."""
    print("Starting Complete Explainable AI Quality Inspection Pipeline")
    print("="*70)
    
    # Setup directories
    setup_directories()
    
    # Handle dataset setup - prioritize real data download
    dataset_ready = False
    
    if config.get('download_data', False):
        print("🔽 Downloading real dataset...")
        dataset_ready = download_dataset(config.get('dataset_name', 'casting'), config['data_dir'])
        if not dataset_ready:
            print("❌ Dataset download failed. Please download manually or install Kaggle API.")
            print("   Run: pip install kaggle")
            print("   Configure: Place kaggle.json API key in ~/.kaggle/")
            return
    else:
        # Check if dataset already exists
        import os
        data_exists = (
            os.path.exists(os.path.join(config['data_dir'], 'train', 'ok')) and
            os.path.exists(os.path.join(config['data_dir'], 'train', 'defective'))
        )
        if data_exists:
            print("✅ Existing dataset found!")
            dataset_ready = True
        else:
            print("❌ No dataset found. Please use --download-data to download the casting dataset.")
            print("   Or manually place data in data/train/{ok,defective}/ and data/test/{ok,defective}/")
            return
    
    # Only proceed if we have real data
    if not dataset_ready:
        print("❌ Cannot proceed without real dataset.")
        return
    
    # Train model
    if config.get('train', True):
        best_model_path = train_model(config)
        config['model_path'] = best_model_path
    else:
        if not config.get('model_path'):
            print("No model path specified and training disabled.")
            return
    
    # Evaluate model
    if config.get('evaluate', True):
        evaluate_model(
            config['model_path'],
            config['data_dir'],
            config
        )
    
    # Generate explanations
    if config.get('explain', True):
        explain_predictions(
            config['model_path'],
            config['data_dir'],
            config,
            num_samples=config.get('num_explanation_samples', 5)
        )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved at: {config['model_path']}")
    print("Results available in: results/")
    print("Explanations available in: results/explanations/")

def main():
    parser = argparse.ArgumentParser(
        description='Explainable AI Quality Inspection Pipeline'
    )
    
    # Pipeline control
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'train', 'evaluate', 'explain'],
                       help='Pipeline mode')
    
    # Data arguments
    parser.add_argument('--data-dir', default='data',
                       help='Path to dataset directory')
    parser.add_argument('--download-data', action='store_true',
                       help='Download dataset before training')
    parser.add_argument('--create-dummy', action='store_true',
                       help='Create dummy dataset for testing')
    parser.add_argument('--dataset-name', default='casting',
                       choices=['casting', 'mvtec', 'neu'],
                       help='Dataset to download')
    
    # Model arguments
    parser.add_argument('--model-type', default='resnet50',
                       choices=['resnet50', 'efficientnet', 'vgg16', 'simple'],
                       help='Model architecture')
    parser.add_argument('--model-path', 
                       help='Path to pre-trained model (for evaluation/explanation)')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes')
    
    # Training arguments (optimized defaults)
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (smaller for better generalization)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (lower for stability)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (increased regularization)')
    parser.add_argument('--optimizer', default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', default='warmup_cosine',
                       choices=['plateau', 'cosine', 'warmup_cosine', 'none'],
                       help='Learning rate scheduler (warmup_cosine recommended)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    
    # Output arguments
    parser.add_argument('--save-dir', default='results/models',
                       help='Model save directory')
    parser.add_argument('--log-dir', default='results/logs',
                       help='Log directory')
    
    # Explainability arguments
    parser.add_argument('--num-explanation-samples', type=int, default=5,
                       help='Number of samples to explain')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers (reduced for stability)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'mode': args.mode,
        'data_dir': args.data_dir,
        'download_data': args.download_data,
        'create_dummy': args.create_dummy,
        'dataset_name': args.dataset_name,
        'model_type': args.model_type,
        'model_path': args.model_path,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_explanation_samples': args.num_explanation_samples,
        'num_workers': args.num_workers,
        'pretrained': True,
        'save_frequency': 10,
        'early_stopping_patience': args.early_stopping_patience
    }
    
    # Set training/evaluation/explanation flags based on mode
    if config['mode'] == 'full':
        config.update({'train': True, 'evaluate': True, 'explain': True})
    elif config['mode'] == 'train':
        config.update({'train': True, 'evaluate': False, 'explain': False})
    elif config['mode'] == 'evaluate':
        config.update({'train': False, 'evaluate': True, 'explain': False})
    elif config['mode'] == 'explain':
        config.update({'train': False, 'evaluate': False, 'explain': True})
    
    # Print configuration
    print("Configuration:")
    print("-" * 40)
    for key, value in config.items():
        if key not in ['train', 'evaluate', 'explain']:  # Don't print internal flags
            print(f"{key}: {value}")
    print("-" * 40)
    
    # Check GPU availability
    if args.gpu and torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    elif args.gpu:
        print("GPU requested but not available. Using CPU.")
    else:
        print("Using CPU")
    
    # Run pipeline
    run_complete_pipeline(config)

if __name__ == "__main__":
    main()