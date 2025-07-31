# Explainable AI Quality Inspection

Automated defect detection for industrial quality inspection using TensorFlow/Keras, with comprehensive evaluation, advanced explainability methods, and an interactive Streamlit dashboard.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg) ![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)

## Overview
This repository provides a complete end-to-end pipeline for industrial quality inspection:
- **🤖 Simple CNN Training**: Sequential CNN with Conv2D layers and extensive augmentation
- **📊 Comprehensive Evaluation**: Detailed metrics, confusion matrices, ROC curves, threshold-based classification
- **🔍 Multi-Method Explainability**: LIME, SHAP, and other explainability methods for TensorFlow models
- **🖥️ Interactive Dashboard**: Streamlit-based UI with real-time analysis and batch processing
- **🚀 Production-Ready**: TensorFlow/Keras training with ModelCheckpoint and 300x300 grayscale processing

The project exposes a single CLI entry point in [`main.py`](main.py) with modes: `full`, `train`, `evaluate`, `explain`, and an interactive dashboard in [`dashboard/app.py`](dashboard/app.py).

## 🌟 Key Features

### Training & Optimization
- **Simple CNN Architecture**: Sequential model with 32→16 Conv2D filters, MaxPooling, Dense layers
- **Optimized Training**: 25 epochs, 150 steps/epoch, Adam optimizer, binary crossentropy
- **Extensive Augmentation**: 360° rotation, shifts, brightness, flips for improved generalization
- **Platform Support**: TensorFlow GPU detection with memory growth, CPU fallback
- **Grayscale Processing**: 300x300 pixel images for optimal performance

### Evaluation & Analysis
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC
- **Visual Analysis**: Confusion matrices, ROC curves, training history plots
- **Error Analysis**: Misclassified sample identification and analysis
- **Per-Class Performance**: Detailed class-wise performance breakdown

### Explainability Methods
- **LIME**: Local Interpretable Model-agnostic Explanations for TensorFlow models
- **SHAP**: SHapley Additive exPlanations for feature importance
- **TensorFlow Compatible**: Adapted explainability methods for .h5 model files
- **Threshold Analysis**: Binary classification with configurable threshold (default: 0.5)

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
│   │   └── dataset.py           # TensorFlow data generators and image processing
│   ├── models/
│   │   └── cnn_model.py         # Simple CNN creation with optimized architecture
│   ├── training/
│   │   └── train_model.py       # Keras training with optimized approach
│   ├── evaluation/
│   │   └── evaluate_model.py    # TensorFlow model evaluation with .h5 loading
│   ├── explainability/
│   │   └── explain_model.py     # TensorFlow-compatible explainability methods
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
    ├── models/                 # Trained Keras models (.h5 files)
    ├── logs/                   # Training history and curves
    ├── explanations/           # Generated explanations
    ├── reports/               # Evaluation reports and plots
    └── experiments/           # Experiment tracking
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+ (3.9+ recommended)
- [uv](https://docs.astral.sh/uv/) package manager (recommended for fast dependency management)
- macOS, Linux, or Windows
- 4GB+ RAM (8GB+ recommended for training)

### Installation with uv (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# 2. Create virtual environment with uv
uv venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies with uv (much faster than pip)
uv pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "from src.data.dataset import get_data_generators; print('✅ Data generators working')"
```

### Alternative Installation with pip
```bash
# If you don't have uv installed
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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
- **Processing Size**: 300x300 pixels (optimized for performance)
- **Color Channels**: Grayscale (1-channel) for efficient processing
- **Automatic Resizing**: Images automatically resized to 300x300

**Important Notes**:
- **Real Dataset Required**: This system is designed for the actual casting product dataset
- **Direct Download**: Automatic download from public source (no API tokens required)
- **No Dummy Data**: The system requires real industrial data for meaningful results
- **Dataset Verification**: Automatic verification ensures correct dataset structure after download
- For manual setup, place your real dataset under `data/` as shown above

## 🚀 Quick Start

### Option 1: Automatic Download & Complete Pipeline
```bash
# Activate environment
source venv/bin/activate

# Download real casting dataset and run full pipeline with optimized parameters
python main.py --mode full --download-data --epochs 25 --batch-size 64

# Launch interactive dashboard
streamlit run dashboard/app.py
```

### Option 2: Manual Dataset Setup
```bash
# 1. Download dataset manually from:
# https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

# 2. Extract and place in data/ directory with structure:
# data/train/{ok,defective}/ and data/test/{ok,defective}/

# 3. Run pipeline without download flag
python main.py --mode full --epochs 30

# 4. Launch dashboard
streamlit run dashboard/app.py
```

### Option 3: Step-by-Step Training
```bash
# 1. Train model with optimized settings
python main.py --mode train --epochs 25 --batch-size 64 --steps-per-epoch 150

# 2. Evaluate trained model
python main.py --mode evaluate --model-path results/models/cnn_casting_inspection_model.h5

# 3. Generate explanations
python main.py --mode explain --model-path results/models/cnn_casting_inspection_model.h5 --num-explanation-samples 10

# 4. Launch dashboard
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
--download-data               # Download casting dataset before training
--create-dummy                # Create dummy dataset for testing
--model-path MODEL_PATH       # Path to saved model (for eval/explain)
--num-classes NUM_CLASSES     # Number of classes (default: 2)
```

#### Training Parameters
```bash
--epochs EPOCHS              # Training epochs (default: 25)
--batch-size BATCH_SIZE       # Batch size (default: 64)
--steps-per-epoch STEPS       # Steps per epoch (default: 150)
--validation-steps STEPS      # Validation steps (default: 150)
--optimizer {adam}            # Optimizer (adam only)
--image-size SIZE             # Image size (default: 300, for 300x300 processing)
```

#### Output & System Options
```bash
--save-dir SAVE_DIR          # Model save directory (default: results/models)
--log-dir LOG_DIR            # Log directory (default: results/logs)
--num-explanation-samples N   # Number of samples to explain (default: 5)
--gpu                        # Use GPU if available (TensorFlow auto-detection)
--seed SEED                  # Random seed for reproducibility (default: 123)
```

### Example Commands

#### Optimized Training
```bash
# Train with optimized defaults
python main.py --mode train \
  --epochs 25 \
  --batch-size 64 \
  --steps-per-epoch 150 \
  --validation-steps 150

# Train with custom image size (default: 300x300)
python main.py --mode train \
  --image-size 300 \
  --epochs 25 \
  --batch-size 64
```

#### Detailed Evaluation
```bash
# Evaluate with Keras model
python main.py --mode evaluate \
  --model-path results/models/cnn_casting_inspection_model.h5 \
  --data-dir ./custom_test_data \
  --batch-size 64
```

#### Focused Explanations
```bash
# Generate explanations for specific samples
python main.py --mode explain \
  --model-path results/models/cnn_casting_inspection_model.h5 \
  --num-explanation-samples 20

# Explain single image with TensorFlow-compatible methods
python -m src.explainability.explain_model \
  --model-path results/models/cnn_casting_inspection_model.h5 \
  --image-path path/to/image.jpg \
  --methods lime shap \
  --save-path explanation.png
```

## 🖥️ Dashboard Features

The Streamlit dashboard provides four main interfaces:

### 1. 🏠 Overview Tab
- Project introduction and architecture comparison
- Model performance benchmarks
- Quick start code examples

### 2. 🔍 Single Image Analysis Tab
- **Upload Interface**: Drag-and-drop image upload (auto-converts to 300x300 grayscale)
- **Real-time Prediction**: Instant classification with confidence scores using threshold
- **Multi-Method Explanations**: LIME, SHAP for TensorFlow models
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

## 🖥️ TensorFlow GPU Support

### Automatic Detection
TensorFlow automatically detects and uses available compute devices:
- **NVIDIA GPUs**: CUDA support with automatic memory growth
- **Apple Silicon**: TensorFlow Metal support (if available)
- **Fallback**: CPU (default behavior)

### Verification
```bash
# Check TensorFlow GPU availability
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Test GPU training
python main.py --mode train --epochs 1 --batch-size 4 --gpu
```

### GPU Configuration
```bash
# Enable GPU with memory growth
python main.py --mode train --gpu

# Force CPU usage (default behavior)
python main.py --mode train

# Memory optimization
python main.py --mode train --batch-size 32
```

## 📁 Expected Outputs

### Training Outputs
```
results/
├── models/
│   └── cnn_casting_inspection_model.h5    # Best Keras model
└── logs/
    ├── training_history.json              # Metrics history
    ├── training_curves.png                # Loss/accuracy plots
    └── test_predictions.png               # Prediction visualizations
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
# Python/uv issues
which python3
python3 --version
uv --version

# Dependencies with uv (recommended)
uv pip install -r requirements.txt

# Alternative with pip
pip install --upgrade pip
pip install -r requirements.txt

# Recreate virtual environment with uv
deactivate && rm -rf venv
uv venv venv && source venv/bin/activate
```

#### Memory Issues
```bash
# Reduce memory usage
python main.py --batch-size 32 --steps-per-epoch 100

# Monitor memory
activity monitor  # macOS
htop             # Linux
```

#### Model Loading Errors
```bash
# Check model file
ls -la results/models/cnn_casting_inspection_model.h5

# Verify TensorFlow model loading
python -c "import tensorflow as tf; model = tf.keras.models.load_model('results/models/cnn_casting_inspection_model.h5'); print('Model loaded successfully')"
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
- **Batch Size**: Start with 64 (default), reduce if OOM errors
- **Steps per Epoch**: 150 (default), adjust based on dataset size
- **Image Size**: 300x300 grayscale (optimized for casting defects)
- **Extensive Augmentation**: Improves generalization and model robustness

#### For Inference
- **Model**: Simple CNN optimized for casting defect detection
- **Batch Processing**: Process multiple 300x300 grayscale images together
- **Threshold**: Adjust classification threshold (default: 0.5) based on precision/recall needs

### System Requirements

#### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.8+

#### Recommended for Training
- **CPU**: 4+ cores (TensorFlow CPU optimization)
- **RAM**: 8GB+ (16GB for large datasets)
- **GPU**: CUDA-compatible (TensorFlow GPU support)
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