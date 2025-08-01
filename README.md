# Explainable AI Quality Inspection

Automated defect detection for industrial casting products using TensorFlow/Keras with explainable AI methods.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## What It Does

This project solves **industrial quality inspection** for casting products using AI:

- **Real Dataset**: 7,348 casting product images (defective vs good quality)
- **CNN Model**: Simple but effective architecture achieving 98%+ accuracy
- **Explainable AI**: LIME and SHAP explanations to understand model decisions
- **Automated Pipeline**: Single command downloads data, trains model, and generates results

**Problem Solved**: Automatically detect defective casting products in manufacturing, with explanations for why the AI made each decision.

## Project Structure

```
explainable-ai-quality-inspection/
├── main.py                     # Main CLI entry point - run entire pipeline
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Development guide
├── src/                        # Source code modules
│   ├── data/
│   │   └── dataset.py         # Kaggle dataset integration & TensorFlow data generators
│   ├── models/
│   │   └── cnn_model.py       # Simple CNN architecture (Sequential model)
│   ├── training/
│   │   └── train_model.py     # Training pipeline with ModelCheckpoint
│   ├── evaluation/
│   │   └── evaluate_model.py  # Model evaluation, confusion matrices, ROC curves
│   └── explainability/
│       └── explain_model.py   # LIME/SHAP explanations for TensorFlow models
├── data/                      # Dataset (auto-downloaded)
│   └── casting_data/
│       └── casting_data/      # 7,348 casting product images
│           ├── train/
│           │   ├── ok_front/  # Good quality products (training)
│           │   └── def_front/ # Defective products (training)
│           └── test/
│               ├── ok_front/  # Good quality products (testing)
│               └── def_front/ # Defective products (testing)
└── results/                   # Generated outputs
    ├── models/                # Trained Keras models (.h5 files)
    ├── logs/                  # Training history, curves, predictions
    ├── explanations/          # LIME/SHAP explanation images
    └── reports/               # Evaluation reports, confusion matrices
```

### CNN Architecture

The model uses a simple but effective Sequential CNN architecture:

```python
Sequential([
    # First convolutional block
    Conv2D(32, kernel_size=3, strides=2, activation='relu'),  # 32 filters
    MaxPooling2D(pool_size=2, strides=2),
    
    # Second convolutional block  
    Conv2D(16, kernel_size=3, strides=2, activation='relu'),  # 16 filters
    MaxPooling2D(pool_size=2, strides=2),
    
    # Fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'), 
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

**Input**: 300x300 grayscale images  
**Output**: Sigmoid probability (0 = good quality, 1 = defective)  
**Total Parameters**: ~1.2M parameters  
**Training**: Adam optimizer, binary crossentropy loss

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Main CLI - handles data download, training, evaluation, explanations |
| `src/data/dataset.py` | TensorFlow data generators with augmentation |
| `src/models/cnn_model.py` | Simple CNN architecture (32→16 Conv2D filters) |
| `src/training/train_model.py` | Training logic with ModelCheckpoint and visualization |
| `src/evaluation/evaluate_model.py` | Model evaluation with detailed metrics and plots |
| `src/explainability/explain_model.py` | LIME and SHAP explanations for model decisions |

## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/alwinpaul1/explainable-ai-quality-inspection.git
cd explainable-ai-quality-inspection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

### Complete Pipeline (Recommended)
```bash
# Downloads dataset, trains model, evaluates, and generates explanations
python main.py --mode full --download-data --epochs 25

# Expected: ~98% accuracy after 25 epochs (45-60 minutes)
```

### Quick Test
```bash
# Fast 3-epoch test to verify everything works
python main.py --mode full --download-data --epochs 3

# Expected: ~62% accuracy after 3 epochs (5-10 minutes)
```

### Individual Steps
```bash
# 1. Train only
python main.py --mode train --epochs 25

# 2. Evaluate trained model
python main.py --mode evaluate --model-path results/models/cnn_casting_inspection_model.h5

# 3. Generate explanations
python main.py --mode explain --model-path results/models/cnn_casting_inspection_model.h5
```

## Results

After training, the following outputs are generated:
- **Trained model**: `results/models/cnn_casting_inspection_model.h5`
- **Training plots**: `results/logs/training_curves.png` and `results/logs/test_predictions.png`
- **Evaluation reports**: `results/reports/evaluation_results.txt`, confusion matrices, ROC curves
- **Explanations**: `results/explanations/explanation_sample_*.png` (LIME/SHAP visualizations)

---

**Repository**: [https://github.com/alwinpaul1/explainable-ai-quality-inspection](https://github.com/alwinpaul1/explainable-ai-quality-inspection)