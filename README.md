# Explainable AI Quality Inspection

Deep learning system for automated defect detection in manufacturing with comprehensive explainability analysis using TensorFlow/Keras.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## Project Summary

This project demonstrates end-to-end machine learning pipeline development for computer vision applications in manufacturing quality control. The system combines deep learning classification with explainable AI techniques to provide interpretable predictions.

### Technical Achievements
- **CNN Architecture**: Custom Sequential model with 676,945 parameters achieving 99.44% accuracy
- **Explainability Methods**: Implementation of LIME, SHAP, Grad-CAM, and Integrated Gradients
- **Dataset Handling**: Processing of 7,348 industrial casting images with automated data pipeline
- **Model Optimization**: Training pipeline with data augmentation, regularization, and performance monitoring
- **Visualization System**: Comprehensive analysis dashboards and model interpretation tools

---

## Implementation Architecture

```
explainable-ai-quality-inspection/
├── main.py                     # CLI interface for pipeline execution
├── requirements.txt            # Dependencies: TensorFlow, LIME, SHAP, OpenCV
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
├── data/                       # Dataset storage
│   └── casting_data/casting_data/ # Industrial casting images (7,348 samples)
│       ├── train/                # Training set (6,633 images)
│       └── test/                 # Test set (715 images)
└── results/                    # Output artifacts
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

### Training Implementation
- **Optimization**: Adam optimizer with binary crossentropy loss
- **Regularization**: Dropout layers (0.2) and data augmentation
- **Data Augmentation**: Rotation, translation, brightness variation, flipping
- **Validation**: 20% split with ModelCheckpoint for best model selection
- **Monitoring**: Training curves and performance metrics tracking

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

### Environment Requirements
- Python 3.8+
- TensorFlow 2.13+
- GPU support (CUDA/MPS) automatically detected
- Memory: 4GB minimum, 8GB recommended

### Installation
```bash
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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
python main.py --mode evaluate --model-path results/models/cnn_casting_inspection_model.hdf5

# Explainability analysis
python main.py --mode explain --model-path results/models/cnn_casting_inspection_model.hdf5
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
- Trained CNN model (HDF5 format, 2.6MB)
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

### Dataset Processing
- **Image Format**: 300×300 grayscale preprocessing
- **Data Split**: Train (6,633), Test (715) with class balance handling
- **Augmentation Pipeline**: Rotation, translation, brightness, scaling transformations
- **Automated Download**: Kaggle API integration with dataset validation

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

