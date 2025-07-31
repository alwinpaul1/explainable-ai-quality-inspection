# Explainable AI Quality Inspection

Automated defect detection for industrial quality inspection with PyTorch training, comprehensive evaluation, advanced explainability methods, and an interactive Streamlit dashboard.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ![MPS Support](https://img.shields.io/badge/Apple%20Silicon-MPS%20Support-green.svg)

## Overview
This repository provides a complete end-to-end pipeline for industrial quality inspection:
- **🤖 Advanced CNN Training**: ResNet50, EfficientNet, VGG16 architectures with optimizations
- **📊 Comprehensive Evaluation**: Detailed metrics, confusion matrices, ROC curves, per-class analysis
- **🔍 Multi-Method Explainability**: LIME, Integrated Gradients, GradCAM, and Occlusion analysis
- **🖥️ Interactive Dashboard**: Streamlit-based UI with real-time analysis and batch processing
- **🚀 Production-Ready**: Early stopping, learning rate scheduling, MPS/GPU support

The project exposes a single CLI entry point in [`main.py`](main.py) with modes: `full`, `train`, `evaluate`, `explain`, and an interactive dashboard in [`dashboard/app.py`](dashboard/app.py).

## 🌟 Key Features

### Training & Optimization
- **Advanced Architectures**: ResNet50, EfficientNet-B0, VGG16, Simple CNN
- **Smart Training**: Early stopping, learning rate warmup, cosine annealing
- **Regularization**: Weight decay, dropout, advanced data augmentation
- **Platform Support**: Automatic MPS (Apple Silicon), CUDA, CPU detection
- **Robust Data Handling**: Automatic dummy dataset generation for testing

### Evaluation & Analysis
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC
- **Visual Analysis**: Confusion matrices, ROC curves, training history plots
- **Error Analysis**: Misclassified sample identification and analysis
- **Per-Class Performance**: Detailed class-wise performance breakdown

### Explainability Methods
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Attribution-based explanations with baseline comparison
- **GradCAM**: Gradient-weighted Class Activation Mapping for CNN visualization
- **Occlusion**: Systematic feature importance through occlusion analysis

### Interactive Dashboard
- **4-Tab Interface**: Overview, Single Image Analysis, Model Performance, Batch Analysis
- **Real-time Inference**: Upload and analyze images instantly
- **Multi-Method Explanations**: Generate and compare different explanation types
- **Batch Processing**: Analyze multiple images with summary statistics

## 🏗️ Architecture Layout
```
.
├── main.py                      # Main CLI entry point with all modes
├── requirements.txt             # Python dependencies
├── dashboard/
│   └── app.py                   # Streamlit dashboard application
├── src/
│   ├── data/
│   │   ├── dataset.py           # Core dataset handling
│   │   ├── enhanced_dataset.py  # Enhanced dataset with advanced features
│   │   └── advanced_augmentation.py  # Advanced data augmentation
│   ├── models/
│   │   └── cnn_model.py         # Model creation and architectures
│   ├── training/
│   │   ├── train_model.py       # Main training module
│   │   ├── enhanced_regularization_trainer.py  # Advanced training
│   │   └── improved_trainer.py  # Enhanced training features
│   ├── evaluation/
│   │   └── evaluate_model.py    # Comprehensive model evaluation
│   ├── explainability/
│   │   └── explain_model.py     # Multi-method explainability
│   └── utils/
│       ├── metrics.py           # Evaluation metrics and calculations
│       └── visualization.py     # Plotting and visualization utilities
├── data/                        # Dataset directory
│   ├── train/
│   │   ├── defective/          # Defective samples
│   │   └── ok/                 # Good samples
│   └── val/                    # Validation split
│       ├── defective/
│       └── ok/
└── results/                    # Output directory
    ├── models/                 # Trained model checkpoints
    ├── logs/                   # Training logs and history
    ├── explanations/           # Generated explanations
    ├── reports/               # Evaluation reports and plots
    └── experiments/           # Experiment tracking
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+ (3.9+ recommended)
- macOS (for MPS support), Linux, or Windows
- 4GB+ RAM (8GB+ recommended for training)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation (optional)
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 📊 Dataset Structure

### Expected Directory Layout
```
data/
├── train/
│   ├── defective/          # Defective product images
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── ok/                 # Good quality images
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── val/                    # Validation split (optional)
│   ├── defective/
│   └── ok/
└── test/                   # Test split (optional)
    ├── defective/
    └── ok/
```

### Supported Formats
- **Image Types**: `.jpg`, `.jpeg`, `.png`
- **Minimum Size**: 32x32 pixels
- **Recommended Size**: 224x224 pixels or larger
- **Color Channels**: RGB (3-channel)

**Important Notes**:
- **Automatic Dataset Download**: The system now includes built-in dataset download functionality
- **Kaggle API Support**: For the casting dataset, install `kaggle` package and configure API token
- **Smart Fallback**: If download fails or no dataset exists, the system automatically creates dummy datasets for testing
- **Dummy Dataset**: Use `--create-dummy` to explicitly create test data with 370 synthetic images
- For production use, manually place your real dataset under `data/` as shown above

## 🚀 Quick Start

### Option 1: Complete Pipeline with Dummy Data
```bash
# Activate environment
source venv/bin/activate

# Create dummy dataset and run full pipeline
python main.py --mode full --create-dummy --epochs 10 --batch-size 8

# Launch interactive dashboard
streamlit run dashboard/app.py
```

### Option 2: Download Real Data (Kaggle API Required)
```bash
# Install and configure Kaggle API first:
# pip install kaggle
# Set up API token from kaggle.com/account

# Download dataset and run pipeline
python main.py --mode full --download-data --dataset-name casting --epochs 30

# Launch interactive dashboard
streamlit run dashboard/app.py
```

### Option 3: Step-by-Step with Custom Data
```bash
# 1. Place your data in data/train/{ok,defective}/ and data/val/{ok,defective}/

# 2. Train model with early stopping
python main.py --mode train --epochs 50 --early-stopping-patience 10

# 3. Evaluate trained model
python main.py --mode evaluate --model-path results/models/best_model.pth

# 4. Generate explanations
python main.py --mode explain --model-path results/models/best_model.pth --num-explanation-samples 10

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## 🛠️ CLI Reference

### Main Command Structure
```bash
python main.py [--mode MODE] [OPTIONS]
```

### Modes
| Mode | Description | Usage |
|------|-------------|-------|
| `full` | Complete pipeline: train → evaluate → explain | `--mode full` |
| `train` | Training only | `--mode train` |
| `evaluate` | Evaluation only (requires `--model-path`) | `--mode evaluate` |
| `explain` | Generate explanations only (requires `--model-path`) | `--mode explain` |

### Key Parameters

#### Data & Model Options
```bash
--data-dir DATA_DIR           # Dataset directory (default: data)
--download-data               # Download dataset before training (requires Kaggle API for some datasets)
--create-dummy                # Create dummy dataset for testing
--dataset-name {casting,mvtec,neu}  # Dataset to download (default: casting)
--model-type {resnet50,efficientnet,vgg16,simple}  # Architecture (default: resnet50)
--model-path MODEL_PATH       # Path to saved model (for eval/explain)
--num-classes NUM_CLASSES     # Number of classes (default: 2)
```

#### Training Parameters
```bash
--epochs EPOCHS              # Training epochs (default: 30)
--batch-size BATCH_SIZE       # Batch size (default: 16)
--learning-rate LR            # Learning rate (default: 0.0001)
--weight-decay WD            # Weight decay (default: 0.01)
--optimizer {adam,sgd}        # Optimizer choice (default: adam)
--scheduler {plateau,cosine,warmup_cosine,none}  # LR scheduler (default: warmup_cosine)
--early-stopping-patience P   # Early stopping patience (default: 10)
```

#### Output & System Options
```bash
--save-dir SAVE_DIR          # Model save directory (default: results/models)
--log-dir LOG_DIR            # Log directory (default: results/logs)
--num-explanation-samples N   # Number of samples to explain (default: 5)
--num-workers NUM_WORKERS     # Data loading workers (default: 2)
--gpu                        # Use GPU if available
```

### Example Commands

#### Advanced Training
```bash
# Train EfficientNet with cosine annealing
python main.py --mode train \
  --model-type efficientnet \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --scheduler warmup_cosine \
  --early-stopping-patience 15

# Train with specific regularization
python main.py --mode train \
  --weight-decay 0.05 \
  --batch-size 8 \
  --optimizer sgd
```

#### Detailed Evaluation
```bash
# Evaluate with custom model
python main.py --mode evaluate \
  --model-path results/models/best_model.pth \
  --data-dir ./custom_test_data \
  --batch-size 64
```

#### Focused Explanations
```bash
# Generate explanations for specific samples
python main.py --mode explain \
  --model-path results/models/best_model.pth \
  --num-explanation-samples 20

# Explain single image with all methods
python -m src.explainability.explain_model \
  --model-path results/models/best_model.pth \
  --image-path path/to/image.jpg \
  --methods lime integrated_gradients gradcam \
  --save-path explanation.png
```

## 🖥️ Dashboard Features

The Streamlit dashboard provides four main interfaces:

### 1. 🏠 Overview Tab
- Project introduction and architecture comparison
- Model performance benchmarks
- Quick start code examples

### 2. 🔍 Single Image Analysis Tab
- **Upload Interface**: Drag-and-drop image upload
- **Real-time Prediction**: Instant classification with confidence scores
- **Multi-Method Explanations**: LIME, Integrated Gradients, GradCAM
- **Interactive Visualization**: Side-by-side original and explanation views

### 3. 📊 Model Performance Tab
- **Metrics Dashboard**: Accuracy, precision, recall, F1-score widgets
- **Confusion Matrix**: Interactive heatmap with normalization options
- **Training History**: Loss and accuracy curves over epochs
- **ROC Curves**: Performance visualization for binary classification

### 4. 📈 Batch Analysis Tab
- **Multi-Image Upload**: Process multiple images simultaneously
- **Batch Statistics**: Aggregated results with pie charts
- **Confidence Analysis**: Distribution of prediction confidence scores
- **Export Results**: Download analysis results as CSV

### Dashboard Commands
```bash
# Launch on default port (8501)
streamlit run dashboard/app.py

# Launch on custom port
streamlit run dashboard/app.py --server.port 8502

# Launch with custom configuration
streamlit run dashboard/app.py --server.headless true --server.address 0.0.0.0
```

## 🍎 Apple Silicon (MPS) Support

### Automatic Detection
The system automatically detects and uses the optimal compute device:
- **Apple Silicon Macs**: MPS (Metal Performance Shaders)
- **NVIDIA GPUs**: CUDA
- **Fallback**: CPU

### Verification
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test MPS training
python main.py --mode train --epochs 1 --batch-size 4
```

### Troubleshooting MPS
```bash
# Enable CPU fallback for MPS issues
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Force CPU usage (note: no --device flag, use environment variable)
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --mode train

# Memory optimization for M1/M2 8GB
python main.py --mode train --batch-size 4 --num-workers 0
```

## 📁 Expected Outputs

### Training Outputs
```
results/
├── models/
│   ├── best_model.pth          # Best validation model
│   ├── checkpoint_epoch_*.pth  # Periodic checkpoints
│   └── final_model.pth         # Final epoch model
└── logs/
    ├── training_history.json   # Metrics history
    └── training_curves.png     # Loss/accuracy plots
```

### Evaluation Outputs
```
results/reports/
├── confusion_matrix.png           # Confusion matrix heatmap
├── confusion_matrix_normalized.png # Normalized confusion matrix
├── roc_curve.png                  # ROC curve (binary classification)
└── evaluation_report.txt          # Comprehensive text report
```

### Explanation Outputs
```
results/explanations/
├── explanation_sample_1.png    # Multi-method explanations
├── explanation_sample_2.png
└── ...
```

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Python/pip issues
which python3
python3 --version

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Virtual environment
deactivate && rm -rf venv
python3 -m venv venv && source venv/bin/activate
```

#### Memory Issues
```bash
# Reduce memory usage
python main.py --batch-size 4 --num-workers 0

# Monitor memory
activity monitor  # macOS
htop             # Linux
```

#### Model Loading Errors
```bash
# Check model file
ls -la results/models/best_model.pth

# Verify model architecture match
python main.py --mode evaluate --model-type resnet50 --model-path results/models/best_model.pth
```

#### Dashboard Issues
```bash
# Port conflicts
lsof -i :8501
streamlit run dashboard/app.py --server.port 8502

# Clear Streamlit cache
rm -rf ~/.streamlit/
```

#### Dataset Issues
```bash
# Check dataset structure
find data -name "*.jpg" | head -10
ls -la data/train/*/

# Verify permissions
chmod -R 755 data/
```

### Performance Optimization

#### For Training
- **Batch Size**: Start with 16, reduce if OOM errors
- **Workers**: Use 2-4 for optimal data loading
- **Learning Rate**: 0.0001 works well for most cases
- **Early Stopping**: Prevents overfitting, saves time

#### For Inference
- **Model Choice**: EfficientNet for best speed/accuracy trade-off
- **Batch Processing**: Process multiple images together
- **Image Preprocessing**: Resize to 224x224 for optimal performance

### System Requirements

#### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.8+

#### Recommended for Training
- **CPU**: 4+ cores or Apple Silicon
- **RAM**: 8GB+ (16GB for large datasets)
- **GPU**: CUDA-compatible or Apple Silicon MPS
- **Storage**: 10GB+ free space

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🆘 Getting Help

```bash
# Command line help
python main.py --help

# Module-specific help
python -m src.training.train_model --help
python -m src.evaluation.evaluate_model --help
python -m src.explainability.explain_model --help
```

For issues and questions, please check the troubleshooting section above or create an issue in the repository.

---

**Repository**: [https://github.com/alwinpaul1/explainable-ai-quality-inspection](https://github.com/alwinpaul1/explainable-ai-quality-inspection)