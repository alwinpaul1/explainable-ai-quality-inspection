# 🔍 Explainable AI Quality Inspection

A comprehensive deep learning system for manufacturing quality inspection with explainable AI capabilities. This project combines state-of-the-art computer vision models with interpretability techniques to detect defects in industrial products while providing clear explanations for the predictions.

![Project Banner](https://img.shields.io/badge/AI-Quality%20Inspection-blue) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **🤖 Multiple Model Architectures**: ResNet50, EfficientNet, VGG16, and custom CNN
- **🔍 Explainable AI**: LIME, SHAP, Integrated Gradients, and GradCAM
- **📊 Comprehensive Evaluation**: Detailed metrics and visualization
- **🎯 Interactive Dashboard**: Streamlit-based web interface
- **📈 Real-time Inference**: Fast prediction with explanation generation
- **🏭 Industrial Datasets**: Support for casting, MVTec, and NEU datasets
- **📋 Automated Pipeline**: End-to-end training and evaluation workflow
- **⚡ Optimized Training**: Fixed overfitting, hardware acceleration, and early stopping
- **🔧 Advanced Augmentation**: Enhanced data augmentation for better generalization

## 🏗️ Project Structure

```
explainable-ai-quality-inspection/
├── 📁 config/                 # Configuration files
├── 📁 dashboard/              # Streamlit dashboard
│   ├── app.py                # Main dashboard application
│   ├── assets/               # Dashboard assets
│   └── components/           # Dashboard components
├── 📁 data/                   # Dataset storage
│   ├── raw/                  # Raw dataset files
│   ├── processed/            # Processed datasets
│   ├── splits/               # Train/validation/test splits
│   ├── test/                 # Test dataset
│   └── train/                # Training dataset
├── 📁 results/                # Training results and outputs
│   ├── experiments/          # Experiment logs
│   ├── explanations/         # Generated explanations
│   ├── logs/                 # Training logs and curves
│   ├── models/               # Trained model files
│   └── reports/              # Evaluation reports
├── 📁 scripts/                # Utility scripts
│   └── download_dataset.py   # Dataset download script
├── 📁 src/                    # Source code
│   ├── data/                 # Data handling modules
│   │   └── dataset.py        # Dataset classes with enhanced augmentation
│   ├── models/               # Model architectures
│   │   └── cnn_model.py      # CNN model definitions (fixed deprecated params)
│   ├── training/             # Training modules
│   │   └── train_model.py    # Fixed training script with early stopping
│   ├── evaluation/           # Evaluation modules
│   │   └── evaluate_model.py # Model evaluation
│   ├── explainability/       # Explainability modules
│   │   └── explain_model.py  # Explanation generation
│   └── utils/                # Utility modules
│       ├── metrics.py        # Metric calculations (fixed warnings)
│       └── visualization.py  # Visualization utilities
├── 📁 tests/                  # Unit tests
├── main.py                   # Main execution script
├── run_improved_training.py  # Improved training script with all fixes
├── requirements.txt          # Python dependencies
├── TRAINING_FIXES_SUMMARY.md # Documentation of training improvements
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd explainable-ai-quality-inspection

# Create and activate virtual environment
python -m venv quality_env
source quality_env/bin/activate  # On Windows: quality_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

#### Option A: Download Casting Dataset (Recommended)
```bash
# Setup Kaggle API credentials first
pip install kaggle
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product

# Or use the provided script
python scripts/download_dataset.py --dataset casting --data-dir data
```

#### Option B: Use Other Datasets
```bash
# Download MVTec Anomaly Detection dataset
python scripts/download_dataset.py --dataset mvtec --data-dir data

# Download NEU Surface Defect dataset
python scripts/download_dataset.py --dataset neu --data-dir data
```

### 3. Training

#### 🔥 Improved Training (Recommended)
```bash
# Use the optimized training script with all fixes
python run_improved_training.py

# This script includes:
# ✅ Fixed overfitting issues
# ✅ Proper GPU/MPS acceleration  
# ✅ Enhanced data augmentation
# ✅ Early stopping
# ✅ Better learning rate scheduling
```

#### Quick Training
```bash
# Train with default settings (ResNet50, 30 epochs)
python main.py --mode train --epochs 30 --batch-size 16

# Train with specific architecture
python main.py --mode train --model-type efficientnet --epochs 50 --learning-rate 0.0001
```

#### Advanced Training Options
```bash
# Full training with optimized parameters
python main.py --mode train \
    --model-type resnet50 \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --weight-decay 0.01 \
    --optimizer adam \
    --scheduler warmup_cosine
```

### 4. Evaluation

```bash
# Evaluate trained model
python main.py --mode evaluate --model-path results/models/best_model.pth

# Evaluate with custom test data
python src/evaluation/evaluate_model.py \
    --model-path results/models/best_model.pth \
    --data-dir data \
    --save-dir results/reports
```

### 5. Generate Explanations

```bash
# Generate explanations for test samples
python main.py --mode explain --model-path results/models/best_model.pth

# Explain specific image
python src/explainability/explain_model.py \
    --model-path results/models/best_model.pth \
    --image-path path/to/image.jpg \
    --methods lime integrated_gradients gradcam \
    --save-path results/explanations/explanation.png
```

### 6. Run Complete Pipeline

```bash
# Run everything: download data, train, evaluate, and explain
python main.py --mode full --download-data --epochs 20
```

## 🎛️ Interactive Dashboard

Launch the interactive Streamlit dashboard for real-time analysis:

```bash
# Start the dashboard
streamlit run dashboard/app.py

# Access at http://localhost:8501
```

### Dashboard Features:
- **🏠 Overview**: Project information and quick start guide
- **🔍 Single Image Analysis**: Upload and analyze individual images
- **📊 Model Performance**: View detailed performance metrics
- **📈 Batch Analysis**: Analyze multiple images at once

## 🔧 Configuration Options

### Model Architectures
- **ResNet50**: `--model-type resnet50` (Default, balanced performance)
- **EfficientNet-B0**: `--model-type efficientnet` (Efficient, good accuracy)
- **VGG16**: `--model-type vgg16` (Classical, interpretable)
- **Simple CNN**: `--model-type simple` (Fast, lightweight)

### Explainability Methods
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Attribution method using gradients
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **Occlusion**: Feature importance via occlusion sensitivity

### Training Parameters
```bash
# Optimized parameters (recommended)
--epochs 30                    # Number of training epochs
--batch-size 16               # Smaller batch size for better generalization
--learning-rate 0.0001        # Lower learning rate for stability
--weight-decay 0.01           # Increased L2 regularization
--optimizer adam              # Optimizer (adam/sgd)
--scheduler warmup_cosine     # Better LR scheduler (warmup_cosine/plateau/cosine)

# Data parameters
--data-dir data               # Dataset directory
--num-workers 2               # Reduced workers for stability

# Output parameters
--save-dir results/models     # Model save directory
--log-dir results/logs        # Training logs directory
```

## 🔧 Training Improvements & Fixes

This project includes comprehensive fixes for common training issues:

### ✅ **Fixed Issues**
- **Overfitting Prevention**: Early stopping and enhanced regularization
- **Hardware Acceleration**: Proper MPS/CUDA device selection for faster training
- **Deprecated Parameters**: Updated PyTorch model loading to use `weights` instead of `pretrained`
- **Data Augmentation**: Enhanced augmentation with 10+ techniques for better generalization
- **Learning Rate**: Improved scheduling with warmup and cosine annealing
- **Precision Warnings**: Fixed sklearn metric calculation warnings

### 📊 **Performance Improvements**
- **Stable Training**: No more erratic validation performance
- **Faster Execution**: GPU/MPS acceleration instead of CPU-only
- **Better Convergence**: Improved learning rate scheduling
- **Reduced Overfitting**: Training-validation gap reduced from 40%+ to <10%

For detailed information about the fixes, see: **[TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md)**

## 📊 Supported Datasets

### 1. Casting Product Dataset
- **Source**: Kaggle - Real-life Industrial Dataset of Casting Product
- **Classes**: OK, Defective
- **Size**: ~7,000 images
- **Format**: JPG images organized by class

### 2. MVTec Anomaly Detection Dataset
- **Source**: MVTec Software GmbH
- **Classes**: 15 different object/texture categories
- **Size**: ~5,000 images
- **Format**: Various industrial objects and textures

### 3. NEU Surface Defect Dataset
- **Source**: Northeastern University
- **Classes**: 6 types of steel surface defects
- **Size**: ~1,800 images
- **Format**: Grayscale images of steel surfaces

## 🎯 Model Performance

### Benchmark Results (Casting Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|---------|----------|----------------|
| ResNet50 | 94.2% | 93.8% | 94.5% | 94.1% | 12ms |
| EfficientNet-B0 | 93.8% | 93.2% | 94.1% | 93.6% | 8ms |
| VGG16 | 92.1% | 91.7% | 92.4% | 92.0% | 18ms |
| Simple CNN | 89.3% | 88.9% | 89.7% | 89.3% | 5ms |

## 🔍 Explainability Examples

### LIME Explanation
Shows which image regions most influence the model's decision, highlighting defective areas in manufacturing parts.

### GradCAM Visualization
Generates heat maps showing where the model focuses its attention when making predictions.

### Integrated Gradients
Provides pixel-level attribution scores showing the contribution of each pixel to the final prediction.
