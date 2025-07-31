#!/usr/bin/env python3
"""
Main execution script for Explainable AI Quality Inspection
"""

import os
import sys
import argparse

import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import get_data_generators, QualityInspectionDataset
from src.training.train_model import train_model_notebook_style, QualityInspectionTrainer
from src.explainability.explain_model import ModelExplainer
from src.evaluation.evaluate_model import ModelEvaluator

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'results/models', 'results/logs', 'results/experiments',
        'results/explanations', 'results/reports'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_dataset(data_dir='data'):
    """Download and setup casting dataset."""
    import urllib.request
    import zipfile
    import requests
    from pathlib import Path
    
    print("🔽 Downloading Casting Product Image Dataset...")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try direct download from alternative sources
        print("📦 Attempting to download dataset...")
        
        # Alternative download URLs (you may need to find a direct link)
        urls = [
            # Add direct download URLs here when available
            # "https://example.com/casting-dataset.zip",
        ]
        
        download_success = False
        
        for i, url in enumerate(urls):
            try:
                print(f"🔄 Trying download source {i+1}/{len(urls)}...")
                
                # Download file
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                zip_path = data_path / 'casting_dataset.zip'
                
                # Save with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r⬇️  Downloaded: {percent:.1f}%", end='', flush=True)
                
                print("\n📦 Extracting dataset...")
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                
                # Clean up zip file
                zip_path.unlink()
                
                # Verify download and structure
                if verify_casting_dataset_structure(data_dir):
                    print("✅ Dataset downloaded and verified successfully!")
                    download_success = True
                    break
                else:
                    print("⚠️ Dataset structure verification failed.")
                    
            except Exception as e:
                print(f"❌ Download from source {i+1} failed: {e}")
                continue
        
        if not download_success:
            # Provide manual download instructions
            print("📋 Automatic download failed. Please download manually:")
            print("   1. Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
            print("   2. Download the dataset (requires free Kaggle account)")
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
            print("\n   Then run the command again without --download-data")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    
    return download_success

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
    """Train the quality inspection model following notebook approach."""
    return train_model_notebook_style(config)

def evaluate_model(model_path, data_dir, config):
    """Evaluate the trained model using TensorFlow/Keras."""
    print("\n" + "="*60)
    print("EVALUATION PHASE (NOTEBOOK STYLE)")
    print("="*60)
    
    # Load the trained model
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create test data generator
    try:
        _, _, test_dataset = get_data_generators(
            data_dir=data_dir,
            batch_size=config.get('batch_size', 64)
        )
        print(f"✅ Test dataset loaded: {test_dataset.samples} samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # Create trainer instance for evaluation
    trainer = QualityInspectionTrainer(config)
    trainer.model = model  # Use the loaded model
    
    # Evaluate on test dataset
    results = trainer.evaluate_on_test(test_dataset, threshold=0.5)
    
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
        dataset_ready = download_dataset(config['data_dir'])
        if not dataset_ready:
            print("❌ Dataset download failed. Please download manually.")
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
                       help='Download casting dataset before training')
    parser.add_argument('--create-dummy', action='store_true',
                       help='Create dummy dataset for testing')
    
    # Model arguments
    parser.add_argument('--model-type', default='simple',
                       choices=['simple'],  # Only simple CNN following notebook
                       help='Model architecture (only simple CNN supported)')
    parser.add_argument('--model-path', 
                       help='Path to pre-trained model (for evaluation/explanation)')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes')
    
    # Training arguments (following notebook)
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs (notebook default: 25)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (notebook default: 64)')
    parser.add_argument('--steps-per-epoch', type=int, default=150,
                       help='Steps per epoch (notebook default: 150)')
    parser.add_argument('--validation-steps', type=int, default=150,
                       help='Validation steps (notebook default: 150)')
    parser.add_argument('--optimizer', default='adam',
                       help='Optimizer (notebook uses adam)')
    parser.add_argument('--image-size', type=int, default=300,
                       help='Image size (notebook uses 300x300)')
    
    # Output arguments
    parser.add_argument('--save-dir', default='results/models',
                       help='Model save directory')
    parser.add_argument('--log-dir', default='results/logs',
                       help='Log directory')
    
    # Explainability arguments
    parser.add_argument('--num-explanation-samples', type=int, default=5,
                       help='Number of samples to explain')
    
    # System arguments
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available (TensorFlow will auto-detect)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for reproducibility (notebook uses 123)')
    
    args = parser.parse_args()
    
    # Create configuration following notebook approach
    config = {
        'mode': args.mode,
        'data_dir': args.data_dir,
        'download_data': args.download_data,
        'create_dummy': args.create_dummy,
        'model_type': args.model_type,
        'model_path': args.model_path,
        'num_classes': 1,  # Binary classification with sigmoid
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'steps_per_epoch': args.steps_per_epoch,
        'validation_steps': args.validation_steps,
        'optimizer': args.optimizer,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'num_explanation_samples': args.num_explanation_samples,
        'image_size': (args.image_size, args.image_size),
        'seed': args.seed,
        'use_gpu': args.gpu
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
    
    # Configure TensorFlow GPU usage
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} GPU(s) detected")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("GPU requested but not available. Using CPU.")
    else:
        print("Using CPU (following notebook approach)")
    
    # Set random seeds for reproducibility (following notebook)
    tf.random.set_seed(config['seed'])
    import numpy as np
    np.random.seed(config['seed'])
    
    # Run pipeline with notebook approach
    run_complete_pipeline(config)

if __name__ == "__main__":
    main()