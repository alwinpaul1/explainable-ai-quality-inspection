# Explainable AI Quality Inspection

Deep learning system for automated defect detection in manufacturing with comprehensive explainability analysis using TensorFlow/Keras.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## Project Summary

This project demonstrates end-to-end machine learning pipeline development for computer vision applications in manufacturing quality control. The system combines deep learning classification with explainable AI techniques to provide interpretable predictions.

### Technical Achievements
- **CNN Architecture**: Custom Sequential model with 676,945 parameters achieving 99.44% accuracy
- **Explainability Methods**: Implementation of LIME, SHAP, Grad-CAM, and Integrated Gradients
- **Dataset Handling**: Processing of 7,348 industrial casting images with automated data pipeline
- **Class Imbalance Solution**: Comprehensive data augmentation methodology with 9 transformation techniques
- **Model Optimization**: Training pipeline with data augmentation, regularization, and performance monitoring
- **Visualization System**: Comprehensive analysis dashboards and model interpretation tools
- **Docker Containerization**: Complete containerized setup with GPU support and live Gradio dashboard
- **Live Dashboard**: Real-time web interface for pipeline monitoring and control

---

## Implementation Architecture

```
explainable-ai-quality-inspection/
├── main.py                     # CLI interface for pipeline execution
├── dashboard.py                # Live Gradio web dashboard
├── requirements.txt            # Dependencies: TensorFlow, LIME, SHAP, OpenCV, Gradio
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore               # Docker build exclusions
├── src/                        # Source code modules
│   ├── data/
│   │   └── dataset.py            # Data loading and preprocessing pipeline
│   ├── models/
│   │   └── cnn_model.py          # CNN architecture definition
│   ├── training/
│   │   └── train_model.py        # Model training and validation logic
│   ├── evaluation/
│   │   └── evaluate_model.py     # Performance evaluation and metrics
│   └── explainability/
│       └── explain_model.py      # Explainability method implementations
├── data/                       # Dataset storage (mounted volume)
│   └── casting_data/casting_data/ # Industrial casting images (7,348 samples)
│       ├── train/                # Training set (6,633 images)
│       └── test/                 # Test set (715 images)
└── results/                    # Output artifacts (mounted volume)
    ├── models/                   # Trained model files
    ├── logs/                     # Training metrics and plots
    ├── explanations/             # Explainability visualizations
    └── reports/                  # Evaluation results
```

---

## CNN Architecture

### Model Design
- **Input Layer**: 300×300 grayscale image preprocessing
- **Feature Extraction**: Two convolutional blocks with MaxPooling
- **Classification**: Fully connected layers with dropout regularization
- **Output**: Sigmoid activation for binary classification
- **Parameters**: 676,945 trainable parameters

### Network Structure
```python
Sequential([
    Conv2D(32, kernel_size=3, strides=2, activation='relu'),  # Feature extraction
    MaxPooling2D(pool_size=2, strides=2),                     
    Conv2D(16, kernel_size=3, strides=2, activation='relu'),  
    MaxPooling2D(pool_size=2, strides=2),                     
    Flatten(),                                                 
    Dense(128, activation='relu'), Dropout(0.2),              # Classification layers
    Dense(64, activation='relu'), Dropout(0.2),               
    Dense(1, activation='sigmoid')                             # Binary output
])
```

### Training Implementation & Class Balance Strategy
- **Optimization**: Adam optimizer with binary crossentropy loss
- **Regularization**: Dropout layers (0.2) and comprehensive data augmentation
- **Class Imbalance Handling**: 
  - Primary: Extensive data augmentation pipeline (9 different transformations)
  - Secondary: Balanced validation split (20%) for unbiased evaluation
  - Monitoring: Precision, recall, and F1-score tracking for both classes
- **Data Augmentation Pipeline**: 
  - Geometric: 360° rotation, ±5% translation, ±5% shear, ±5% zoom
  - Flip: Horizontal and vertical flipping
  - Intensity: ±25% brightness variation
- **Validation Strategy**: 20% split with ModelCheckpoint for best model selection
- **Performance Monitoring**: Training curves, confusion matrix, and per-class metrics tracking
- **Threshold Optimization**: Configurable classification threshold for precision-recall balance

---

## Explainability Implementation

### Interpretation Methods

| Method | Technique | Implementation | Output |
|--------|-----------|----------------|--------|
| **LIME** | Local surrogate models | Superpixel perturbation analysis | Feature importance regions |
| **SHAP** | Shapley value computation | Background sample generation | Attribution heatmaps |
| **Grad-CAM** | Gradient-weighted attention | Convolutional layer activation mapping | Attention visualization |
| **Integrated Gradients** | Path integral attribution | Baseline to input interpolation | Pixel-level attribution |

### Visualization System
- **Comprehensive Dashboard**: Multi-method comparison in unified interface
- **Architecture Visualization**: Model structure and parameter analysis
- **Performance Metrics**: Confidence scores and prediction analysis
- **Method Validation**: Cross-technique verification of explanations

---

## Setup and Usage

### Option 1: Docker (Recommended)

The easiest way to run the application is using Docker, which provides a consistent environment across different systems.

#### Prerequisites
- Docker - [Install Docker](https://docs.docker.com/get-docker/)
- Docker Compose - Usually included with Docker Desktop
- NVIDIA Docker (for GPU support) - [Install NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Quick Start with Docker
```bash
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# Build all services
docker-compose build
```

#### ✅ Super Simple Docker Commands

**Main Pipeline:**
```bash
# Run full AI pipeline (complete training)
docker-compose up ai-quality-inspection

# Or run and remove container after completion
docker-compose run --rm ai-quality-inspection
```

**🎯 Live Dashboard (Recommended):**
```bash
# Launch interactive web dashboard
docker-compose up dashboard

# Access at: http://localhost:7860
```

**🌟 Key Dashboard Features:**
- 🚀 **Real-time Pipeline Control**: Start/stop training with custom parameters
- 📊 **Live Monitoring**: Training curves, accuracy metrics, and progress logs
- 🎯 **Interactive Results**: Confusion matrices, ROC curves, and performance analytics
- 🔍 **Explainability Viewer**: LIME, SHAP, Grad-CAM, and Integrated Gradients visualizations
- ⚙️ **Parameter Control**: Adjustable epochs, batch size, and pipeline modes


#### For GPU Support
Uncomment the GPU configuration in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Option 2: Local Installation

#### Environment Requirements
- Python 3.8+
- TensorFlow 2.13+
- GPU support (CUDA/MPS) automatically detected
- Memory: 4GB minimum, 8GB recommended

#### Installation
```bash
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# Create virtual environment with uv
uv venv venv

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies with uv
uv pip install -r requirements.txt
```

---

### Execution

#### Full Pipeline
```bash
# Complete training and analysis pipeline
python main.py --mode full --download-data --epochs 25 --batch-size 64
```

#### Individual Components
```bash
# Model training
python main.py --mode train --epochs 25 --batch-size 64

# Model evaluation
python main.py --mode evaluate --model-path results/models/cnn_casting_inspection_model.keras

# Explainability analysis
python main.py --mode explain --model-path results/models/cnn_casting_inspection_model.keras
```

---

## Technical Results

### Model Performance
- **Test Accuracy**: 99.44%
- **Precision**: 99.34%
- **Recall**: 99.78%
- **F1 Score**: 99.56%

### Output Artifacts

#### Model Files
- Trained CNN model (Keras format, 2.6MB)
- Training history and performance metrics
- Model architecture visualization

#### Evaluation Results
- Confusion matrix analysis
- ROC curve and AUC metrics
- Classification report with per-class statistics
- Misclassified sample analysis

#### Explainability Outputs
- LIME superpixel explanations
- SHAP attribution heatmaps
- Grad-CAM attention visualizations
- Integrated Gradients pixel attributions
- Comparative analysis dashboard

---

## Implementation Details

### Dataset Processing & Class Imbalance Handling

#### Data Distribution Strategy
- **Dataset Split**: Train (6,633 images), Test (715 images) with 80/20 train-validation split
- **Class Balance Analysis**: Comprehensive data distribution analysis with visualization
- **Image Format**: 300×300 grayscale preprocessing with normalization (1/255 scaling)

#### Data Augmentation Methodology for Class Imbalance
The system employs a comprehensive data augmentation strategy to address class imbalance and improve model generalization:

**Augmentation Techniques Applied:**
- **Rotation Augmentation**: `rotation_range=360` - Full 360-degree rotation for maximum variety
- **Spatial Transformations**: 
  - `width_shift_range=0.05` - Horizontal translation (±5%)
  - `height_shift_range=0.05` - Vertical translation (±5%)
  - `shear_range=0.05` - Shear transformation (±5%)
- **Scale Variations**: `zoom_range=0.05` - Zoom in/out (±5%)
- **Flip Augmentations**: 
  - `horizontal_flip=True` - Horizontal flipping
  - `vertical_flip=True` - Vertical flipping
- **Brightness Variation**: `brightness_range=[0.75, 1.25]` - ±25% brightness adjustment

**Implementation Details:**
- **Training Data**: Full augmentation pipeline applied to training set
- **Validation Data**: Same augmentation pipeline for validation consistency
- **Test Data**: No augmentation (clean evaluation data)
- **Reproducibility**: Fixed random seed (123) for consistent results

#### Class Imbalance Mitigation Strategy
1. **Data Augmentation as Primary Method**: Extensive augmentation creates synthetic samples to balance class distribution
2. **Validation Split**: 20% validation split ensures balanced evaluation during training
3. **Performance Monitoring**: Comprehensive metrics tracking (precision, recall, F1-score) to monitor class-specific performance
4. **Threshold Optimization**: Configurable classification threshold (default 0.5) for precision-recall trade-off

#### Automated Download & Validation
- **Kaggle API Integration**: Automated dataset download with authentication
- **Dataset Validation**: Structure verification and completeness checks
- **Error Handling**: Robust fallback mechanisms for data loading issues

### Model Implementation
- **Framework**: TensorFlow 2.13+ with Keras API
- **Architecture**: Custom CNN with efficient parameter usage
- **Training Strategy**: Adam optimization with learning rate scheduling
- **Regularization**: Dropout layers and early stopping implementation

### Explainability Framework
- **LIME**: Implemented with custom superpixel segmentation
- **SHAP**: DeepExplainer with enhanced background sampling
- **Grad-CAM**: Gradient computation using TensorFlow GradientTape
- **Integrated Gradients**: Path integral implementation with baseline interpolation

### System Features
- **CLI Interface**: Modular pipeline execution with parameter customization
- **GPU Acceleration**: Automatic hardware detection and memory optimization
- **Result Persistence**: Structured output with comprehensive logging
- **Error Handling**: Robust exception management and recovery mechanisms

---

