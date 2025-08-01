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

from src.data.dataset import get_data_generators
from src.training.train_model import train_model_notebook_style, QualityInspectionTrainer
from src.explainability.explain_model import ModelExplainer

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
    """Download casting dataset using Kaggle API only."""
    from pathlib import Path
    import subprocess
    
    print("🔽 Downloading Casting Product Image Dataset...")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📦 Attempting Kaggle API download...")
        
        # Install kaggle package if not available
        try:
            import kaggle
        except ImportError:
            print("📦 Installing Kaggle API using uv...")
            try:
                # Use uv pip install in virtual environment
                subprocess.check_call(["uv", "pip", "install", "kaggle"])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise ImportError("Failed to install kaggle package. Please install manually: uv pip install kaggle")
            import kaggle
        
        # Authenticate Kaggle API
        kaggle.api.authenticate()
        
        # Download the specific dataset
        dataset_name = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
        print(f"⬇️  Downloading: {dataset_name}...")
        
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=str(data_path), 
            unzip=True
        )
        
        print("✅ Dataset downloaded and extracted successfully!")
        print(f"📁 Dataset saved to: {data_path}")
        
        # List the contents to show what was downloaded
        print("\n📋 Downloaded dataset structure:")
        import os
        for root, dirs, files in os.walk(data_path):
            level = root.replace(str(data_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ Kaggle API download failed: {e}")
        print("💡 Manual download required:")
        print("   1. Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
        print("   2. Download the dataset (requires free Kaggle account)")
        print("   3. Extract the zip file to the 'data/' directory")
        print("   4. The dataset will keep its original structure")
        return False


def verify_casting_dataset_structure(data_dir):
    """Verify that the casting dataset structure exists and has images."""
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory does not exist: {data_path}")
        return False
    
    # Count total images in the dataset
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    total_images = 0
    
    for ext in image_extensions:
        total_images += len(list(data_path.rglob(ext)))
    
    if total_images == 0:
        print(f"❌ No images found in: {data_path}")
        return False
    
    print(f"✅ Found {total_images} images in dataset")
    
    # Show the actual structure
    print("📁 Dataset structure:")
    import os
    for root, dirs, files in os.walk(data_path):
        level = root.replace(str(data_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Count images in this directory
        image_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if image_count > 0:
            subindent = ' ' * 2 * (level + 1)
            print(f"{subindent}[{image_count} images]")
    
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
    
    # Get data generators to access test samples correctly
    try:
        _, _, test_dataset = get_data_generators(data_dir, batch_size=32)
        
        # Get actual test directory path
        test_dir = os.path.join(data_dir, 'test')
        val_dir = os.path.join(data_dir, 'val')
        
        # Use val dir if test doesn't exist
        actual_test_dir = val_dir if not os.path.exists(test_dir) and os.path.exists(val_dir) else test_dir
        print(f"Using test samples from: {actual_test_dir}")
        
        # Get sample file paths directly from filesystem
        sample_paths = []
        for class_name in ['ok', 'defective']:
            class_dir = os.path.join(actual_test_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir)[:num_samples//2]:  # Split samples between classes
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_paths.append(os.path.join(class_dir, filename))
        
        # Explain samples
        for i, sample_path in enumerate(sample_paths[:num_samples]):
            print(f"\nExplaining sample {i+1}: {os.path.basename(sample_path)}")
            
            save_path = f"results/explanations/explanation_sample_{i+1}.png"
            
            try:
                explainer.explain_image(
                    sample_path,
                    methods=['lime', 'shap', 'gradcam'],  # Changed from integrated_gradients to shap
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
        dataset_ready = verify_casting_dataset_structure(config['data_dir'])
        if dataset_ready:
            print("✅ Existing dataset found!")
        else:
            print("❌ No dataset found. Please use --download-data to download the casting dataset.")
            print("   Or manually place the casting dataset in the 'data/' directory")
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
    
    # Model arguments (TensorFlow/Keras only)
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
        'model_type': 'simple',  # Fixed to simple CNN following notebook
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