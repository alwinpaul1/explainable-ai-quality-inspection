# Explainable AI Quality Inspection

**Advanced industrial defect detection system with comprehensive explainable AI for casting product quality inspection using TensorFlow/Keras.**

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## Overview

This project provides a complete production-ready solution for automated industrial quality inspection:

### Industrial Problem Solved
- **Real-World Dataset**: 7,348+ high-resolution casting product images
- **Binary Classification**: Automated detection of defective vs good quality products  
- **Explainable Decisions**: Four complementary AI explanation methods
- **Production Ready**: Complete pipeline from data to deployment

### Key Capabilities
- **Advanced CNN**: Optimized Sequential architecture achieving 99.44% accuracy
- **Explainable AI**: LIME, SHAP, Grad-CAM, and Integrated Gradients
- **Comprehensive Analytics**: Detailed evaluation with confidence metrics
- **Smart Pipeline**: Automated data download, training, and analysis
- **Rich Visualizations**: Publication-ready explanation dashboards

---

## System Architecture

```
explainable-ai-quality-inspection/
├── main.py                     # Main CLI entry point - complete pipeline orchestration
├── requirements.txt            # Production dependencies (TensorFlow, LIME, SHAP, etc.)
├── CLAUDE.md                   # Development guide and best practices
├── src/                        # Modular source code
│   ├── data/
│   │   └── dataset.py            # Smart Kaggle integration + TensorFlow data generators
│   ├── models/
│   │   └── cnn_model.py          # Optimized CNN architecture (676,945 parameters)
│   ├── training/
│   │   └── train_model.py        # Advanced training pipeline with callbacks
│   ├── evaluation/
│   │   └── evaluate_model.py     # Comprehensive model evaluation and metrics
│   └── explainability/
│       └── explain_model.py      # Four explainability methods (LIME/SHAP/Grad-CAM/IG)
├── data/                       # Smart dataset management
│   └── casting_data/casting_data/ # 7,348 casting product images (auto-downloaded)
│       ├── train/ (6,633 images)
│       │   ├── ok_front/         # Good quality products (2,875 images)
│       │   └── def_front/        # Defective products (3,758 images)
│       └── test/ (715 images) 
│           ├── ok_front/         # Good quality test set (262 images)
│           └── def_front/        # Defective test set (453 images)
└── results/                   # Generated outputs and artifacts
    ├── models/                   # Trained models (.hdf5 format)
    ├── logs/                     # Training history and visualizations  
    ├── explanations/             # AI explanation dashboards
    └── reports/                  # Evaluation reports and analytics
```

---

## Advanced CNN Architecture

### Model Specifications
- **Input**: 300×300 grayscale images
- **Architecture**: Sequential CNN with optimized layer design
- **Parameters**: 676,945 trainable parameters (~2.6 MB model size)
- **Performance**: 99.44% accuracy, 99.78% recall, 99.34% precision
- **Training**: Adam optimizer with binary crossentropy loss

### Layer-by-Layer Design
```python
Sequential([
    # Optimized Convolutional Feature Extraction
    Conv2D(32, kernel_size=3, strides=2, activation='relu'),  # 149×149×32
    MaxPooling2D(pool_size=2, strides=2),                     # 74×74×32
    
    Conv2D(16, kernel_size=3, strides=2, activation='relu'),  # 36×36×16  
    MaxPooling2D(pool_size=2, strides=2),                     # 18×18×16
    
    # Classification Head
    Flatten(),                                                 # 5,184 features
    Dense(128, activation='relu'), Dropout(0.2),              # 128 units + regularization
    Dense(64, activation='relu'), Dropout(0.2),               # 64 units + regularization
    Dense(1, activation='sigmoid')                             # Binary classification
])
```

### Training Configuration
- **Epochs**: 25 (with early stopping)
- **Batch Size**: 64 images
- **Steps per Epoch**: 150 (9,600 images per epoch)
- **Data Augmentation**: Rotation, shifts, brightness, flips
- **Validation Split**: 20% of training data

---

## Explainable AI System

### Four Complementary Methods

| Method | Type | Purpose | Visualization |
|--------|------|---------|---------------|
| **LIME** | Local Interpretability | Superpixel-based local explanations | Segmented regions |
| **SHAP** | Global Feature Importance | Shapley value attribution | Heat maps |
| **Grad-CAM** | Attention Mapping | Gradient-weighted activations | Attention overlays |
| **Integrated Gradients** | Pixel Attribution | Path-based feature attribution | Attribution maps |

### Advanced Visualization Dashboard
- **3×3 Comprehensive Grid**: All methods in single view
- **CNN Architecture Diagram**: Layer-by-layer visual breakdown  
- **Prediction Confidence**: Probability distributions and metrics
- **Method Coverage**: Status indicators for each explanation technique
- **Interactive Elements**: Detailed attribution analysis

---

## Installation & Setup

### Requirements
- Python 3.8+ 
- TensorFlow 2.13+
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional, auto-detected)

### Quick Setup with UV (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# 2. Create virtual environment with uv
uv venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

### Alternative Setup (Standard)
```bash
# Using standard Python venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### Complete Pipeline (Recommended)
```bash
# Downloads dataset, trains model, evaluates, generates explanations
# Smart download: automatically skips if dataset already exists
python main.py --mode full --download-data --epochs 25 --batch-size 64

# Expected Results:
# 99.44% accuracy after 25 epochs (~45-60 minutes)
# Comprehensive evaluation reports  
# 4-method explainability analysis
```

### Quick Verification Test
```bash
# Fast 3-epoch test to verify system works
python main.py --mode full --download-data --epochs 3 --batch-size 32

# Expected: ~85% accuracy in 5-10 minutes
```

### Individual Pipeline Components

#### Training Only
```bash
python main.py --mode train --epochs 25 --batch-size 64 --steps-per-epoch 150
```

#### Evaluation Only
```bash
python main.py --mode evaluate --model-path results/models/cnn_casting_inspection_model.hdf5
```

#### Explainability Only
```bash
python main.py --mode explain --model-path results/models/cnn_casting_inspection_model.hdf5 --num-explanation-samples 10
```

### Advanced Configuration
```bash
# Custom training parameters
python main.py --mode full \
    --download-data \
    --epochs 50 \
    --batch-size 128 \
    --image-size 300 \
    --steps-per-epoch 200 \
    --validation-steps 150

```

---

## Results & Outputs

### Generated Artifacts

#### Models & Training
- `results/models/cnn_casting_inspection_model.hdf5` - Trained model
- `results/logs/training_curves.png` - Training/validation curves
- `results/logs/training_history.json` - Complete training metrics

#### Evaluation Reports
- `results/reports/evaluation_results.txt` - Detailed performance metrics
- `results/reports/confusion_matrix.png` - Confusion matrix visualization
- `results/reports/roc_curve.png` - ROC curve analysis
- `results/reports/test_predictions.png` - 4×4 prediction visualization

#### Explainability Analysis
- `results/explanations/explanation_sample_*.png` - 3×3 explanation dashboards
- Individual method outputs with attribution analysis
- CNN architecture diagrams and method coverage reports

### Advanced Visualizations

#### Training Analysis
- **8×8 Batch Grids**: Complete batch visualization with augmentation
- **Pixel-Level Analysis**: 25×25 detailed pixel value inspection
- **Data Distribution**: Proportional analysis across train/validation/test
- **Training Curves**: Loss and accuracy progression with seaborn styling

#### Evaluation Insights
- **Misclassified Analysis**: Detailed examination of edge cases
- **Confidence Distributions**: Prediction probability analysis
- **ROC Analysis**: Threshold optimization curves
- **Class Balance**: Performance across ok_front vs def_front classes

---

## Technical Specifications

### Dataset Intelligence
- **Smart Download**: Automatic skip if complete dataset exists
- **Validation Checks**: Structure integrity, image counts, file formats
- **Class Balance**: Handles imbalanced dataset (ok_front: 3,137, def_front: 4,211)
- **Augmentation**: 7 transformation types for robust training

### Explainability Engine
- **LIME**: Local superpixel perturbation with 1000+ samples
- **SHAP**: Enhanced background generation with 10 diverse samples
- **Grad-CAM**: Gradient-weighted class activation mapping
- **Integrated Gradients**: Path-based attribution with 50-step integration

### Performance Optimizations
- **Memory Efficient**: Batch processing for large images
- **GPU Compatible**: Automatic CUDA detection and memory growth
- **TensorFlow 2.x**: Modern gradient computation with @tf.function
- **Error Resilient**: Comprehensive exception handling and recovery

### Evaluation Metrics
- **Binary Classification**: Optimized for industrial defect detection
- **Threshold Analysis**: ROC-based optimal threshold selection
- **Statistical Significance**: Confidence intervals and error bounds
- **Production Metrics**: False positive/negative cost analysis

---

