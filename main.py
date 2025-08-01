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

def check_tensorflow_gpu_support():
    """Check TensorFlow GPU support and provide detailed information."""
    print(f"🔍 TensorFlow Version: {tf.__version__}")
    print(f"🔍 Python Version: {sys.version}")
    
    # Check CUDA support
    cuda_available = tf.test.is_built_with_cuda()
    print(f"🔍 CUDA Support: {'✅ Available' if cuda_available else '❌ Not Available'}")
    
    # Check GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"🔍 CUDA GPUs: {len(gpu_devices)} device(s)")
    
    # Check MPS support (Apple Silicon)
    try:
        mps_devices = tf.config.list_physical_devices('MPS')
        print(f"🔍 MPS Devices: {len(mps_devices)} device(s)")
    except:
        print(f"🔍 MPS Devices: Not available in this TensorFlow version")
    
    return cuda_available, len(gpu_devices)

def test_gpu_performance():
    """Test GPU performance with a simple computation."""
    print(f"\n🧪 GPU Performance Test:")
    print("-" * 40)
    
    try:
        # Create a simple tensor operation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = tf.timestamp()
            c = tf.matmul(a, b)
            end_time = tf.timestamp()
            gpu_time = end_time - start_time
            print(f"✅ GPU computation time: {gpu_time:.4f} seconds")
    except:
        print(f"❌ GPU test failed")
        gpu_time = None
    
    try:
        # Compare with CPU
        with tf.device('/CPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = tf.timestamp()
            c = tf.matmul(a, b)
            end_time = tf.timestamp()
            cpu_time = end_time - start_time
            print(f"✅ CPU computation time: {cpu_time:.4f} seconds")
            
            if gpu_time:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU speedup: {speedup:.2f}x faster than CPU")
    except:
        print(f"❌ CPU test failed")
    
    print("-" * 40)

from src.data.dataset import get_data_generators, analyze_data_distribution
from src.training.train_model import train_model, QualityInspectionTrainer
from src.evaluation.evaluate_model import ModelEvaluator
from src.explainability.explain_model import ModelExplainer

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'results/models', 'results/logs',
        'results/explanations', 'results/reports'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_dataset(data_dir='data'):
    """Download casting dataset using Kaggle API only if not already present."""
    from pathlib import Path
    import subprocess
    
    # Check if dataset already exists and is complete
    if verify_casting_dataset_structure(data_dir):
        print("✅ Dataset already exists and is complete!")
        print(f"📁 Using existing dataset at: {data_dir}")
        return True
    
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
        return False
    
    # Check for expected casting dataset structure
    casting_data_path = data_path / 'casting_data' / 'casting_data'
    
    # Expected structure validation
    expected_dirs = [
        casting_data_path / 'train' / 'ok_front',
        casting_data_path / 'train' / 'def_front',
        casting_data_path / 'test' / 'ok_front', 
        casting_data_path / 'test' / 'def_front'
    ]
    
    # Check if all expected directories exist
    for expected_dir in expected_dirs:
        if not expected_dir.exists():
            return False
    
    # Count images in each directory to ensure dataset is complete
    image_extensions = ['.jpg', '.jpeg', '.png']
    min_images_per_class = 100  # Minimum threshold for a complete dataset
    
    for expected_dir in expected_dirs:
        image_count = sum(1 for f in expected_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions)
        if image_count < min_images_per_class:
            return False
    
    # Count total images in the dataset
    total_images = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        total_images += len(list(data_path.rglob(ext)))
    
    if total_images < 1000:  # Expect at least 1000 images for complete dataset
        return False
    
    return True

def _show_dataset_info(data_dir):
    """Show dataset information when found."""
    from pathlib import Path
    
    data_path = Path(data_dir)
    casting_data_path = data_path / 'casting_data' / 'casting_data'
    
    # Count images in each directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    print("📁 Dataset structure:")
    for split in ['train', 'test']:
        split_path = casting_data_path / split
        if split_path.exists():
            print(f"  {split}/")
            for class_name in ['ok_front', 'def_front']:
                class_path = split_path / class_name
                if class_path.exists():
                    image_count = sum(1 for f in class_path.iterdir() 
                                     if f.is_file() and f.suffix.lower() in image_extensions)
                    print(f"    {class_name}/  [{image_count} images]")
    
    # Total count
    total_images = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        total_images += len(list(data_path.rglob(ext)))
    
    print(f"📊 Total images in dataset: {total_images}")

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


def train_model_pipeline(config):
    """Train the quality inspection model."""
    return train_model(config)

def evaluate_model(model_path, data_dir, config):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("TESTING ON UNSEEN IMAGES")
    print("="*60)
    
    # Create evaluator with detailed reporting
    evaluator = ModelEvaluator(
        model_path=model_path,
        model_type=config.get('model_type', 'simple'),
        num_classes=config.get('num_classes', 1)
    )
    
    # Load test data
    try:
        train_dataset, validation_dataset, test_dataset = get_data_generators(
            data_dir=data_dir,
            batch_size=config.get('batch_size', 64)
        )
        print(f"✅ Test dataset loaded: {test_dataset.samples} samples")
        
        # Add data distribution analysis
        print("\n📊 ANALYZING DATA DISTRIBUTION:")
        analyze_data_distribution(train_dataset, validation_dataset, test_dataset, 
                                save_plots=True, save_dir='results/reports')
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # Comprehensive evaluation with visualization
    results = evaluator.evaluate(test_dataset, threshold=0.5, save_plots=True, 
                               save_dir='results/reports')
    
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
        
        # Get actual test directory path following dataset structure
        casting_data_dir = os.path.join(data_dir, 'casting_data', 'casting_data')
        test_dir = os.path.join(casting_data_dir, 'test')
        val_dir = os.path.join(casting_data_dir, 'val')
        
        # Try test dir first, fallback to val dir if needed
        if os.path.exists(test_dir):
            actual_test_dir = test_dir
            print(f"Using test samples from: {actual_test_dir}")
        elif os.path.exists(val_dir):
            actual_test_dir = val_dir  
            print(f"Using validation samples for explanation from: {actual_test_dir}")
        else:
            # Fallback to train dir for demonstration
            train_dir = os.path.join(casting_data_dir, 'train')
            actual_test_dir = train_dir
            print(f"Using training samples for explanation from: {actual_test_dir}")
        
        # Get sample file paths using actual class names from dataset
        sample_paths = []
        for class_name in ['ok_front', 'def_front']:
            class_dir = os.path.join(actual_test_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for filename in files[:max(1, num_samples//2)]:  # At least 1 sample per class
                    sample_paths.append(os.path.join(class_dir, filename))
                print(f"Found {len(files)} images in {class_name}, using {min(len(files), max(1, num_samples//2))}")
            else:
                print(f"Class directory not found: {class_dir}")
        
        # Explain samples
        for i, sample_path in enumerate(sample_paths[:num_samples]):
            print(f"\nExplaining sample {i+1}: {os.path.basename(sample_path)}")
            
            save_path = f"results/explanations/explanation_sample_{i+1}.png"
            
            try:
                explainer.explain_image(
                    sample_path,
                    methods=['lime', 'shap', 'gradcam', 'integrated_gradients'],
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
    
    # Handle dataset setup - smart data management
    dataset_ready = False
    
    if config.get('download_data', False):
        print("🔽 Checking and downloading dataset if needed...")
        dataset_ready = download_dataset(config['data_dir'])
        if not dataset_ready:
            print("❌ Dataset download failed. Please download manually.")
            return
    else:
        # Check if dataset already exists
        dataset_ready = verify_casting_dataset_structure(config['data_dir'])
        if dataset_ready:
            print("✅ Existing dataset found!")
            # Show dataset info when found
            _show_dataset_info(config['data_dir'])
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
        best_model_path = train_model_pipeline(config)
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
        'seed': args.seed
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
    
    # Check TensorFlow GPU support
    print("\n🔧 GPU Configuration:")
    print("-" * 40)
    cuda_available, num_gpus = check_tensorflow_gpu_support()
    print("-" * 40)
    
    # Test GPU performance if available
    if cuda_available or (hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('MPS')):
        test_gpu_performance()
    
    # Check for CUDA GPUs first
    cuda_gpus = tf.config.experimental.list_physical_devices('GPU')
    if cuda_gpus:
        try:
            # Enable memory growth for CUDA GPUs
            for gpu in cuda_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ CUDA GPU detected: {len(cuda_gpus)} GPU(s)")
            print(f"   - Memory growth enabled for optimal performance")
            print(f"   - Using CUDA acceleration")
            
            # Set mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"   - Mixed precision enabled for faster training")
            
        except RuntimeError as e:
            print(f"❌ CUDA GPU configuration error: {e}")
            print(f"   - Falling back to CPU")
    
    # Check for Apple Silicon MPS (Metal Performance Shaders)
    elif hasattr(tf.config, 'list_physical_devices'):
        try:
            mps_devices = tf.config.list_physical_devices('MPS')
            if mps_devices:
                print(f"✅ Apple Silicon MPS detected: {len(mps_devices)} device(s)")
                print(f"   - Metal Performance Shaders enabled")
                print(f"   - Using MPS acceleration")
                
                # Enable MPS for Apple Silicon
                try:
                    tf.config.experimental.set_memory_growth(mps_devices[0], True)
                    print(f"   - Memory growth enabled for optimal performance")
                except:
                    print(f"   - Memory growth not available for MPS")
                
            else:
                print(f"ℹ️  No MPS devices detected. Using CPU.")
        except Exception as e:
            print(f"ℹ️  MPS detection failed: {e}")
            print(f"   - Using CPU")
    
    else:
        print(f"ℹ️  No GPU detected. Using CPU.")
    
    print("-" * 40)
    
    # Set optimal device strategy
    try:
        if cuda_gpus:
            # Use MirroredStrategy for multi-GPU training
            if len(cuda_gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(f"🚀 Using MirroredStrategy for {len(cuda_gpus)} GPUs")
            else:
                strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
                print(f"🚀 Using single GPU strategy")
        elif hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('MPS'):
            strategy = tf.distribute.OneDeviceStrategy("/MPS:0")
            print(f"🚀 Using MPS strategy")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
            print(f"🚀 Using CPU strategy")
    except Exception as e:
        print(f"⚠️  Strategy setup failed: {e}")
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        print(f"🚀 Falling back to CPU strategy")
    
    # Set random seeds for reproducibility (following notebook)
    tf.random.set_seed(config['seed'])
    import numpy as np
    np.random.seed(config['seed'])
    
    # Add strategy to config
    config['strategy'] = strategy
    
    # Run pipeline with notebook approach
    run_complete_pipeline(config)

if __name__ == "__main__":
    main()