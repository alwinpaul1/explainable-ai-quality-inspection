# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Explainable AI Quality Inspection** system - a comprehensive deep learning project for manufacturing defect detection with AI interpretability. The system combines state-of-the-art computer vision models (ResNet50, EfficientNet, VGG16) with explainability techniques (LIME, SHAP, GradCAM, Integrated Gradients) to detect defects in industrial products.

## Essential Commands

### Development Setup
```bash
pip install -r requirements.txt
```

### Main Pipeline Operations
```bash
# Quick training (20 epochs, ResNet50)
python main.py --mode train --epochs 20 --batch-size 32

# Full pipeline with data download
python main.py --mode full --download-data --epochs 20

# Evaluate existing model
python main.py --mode evaluate --model-path results/models/best_model.pth

# Generate explanations  
python main.py --mode explain --model-path results/models/best_model.pth

# Launch interactive dashboard
streamlit run dashboard/app.py
```

### Dataset Management
```bash
# Download datasets
python scripts/download_dataset.py --dataset casting --data-dir data
python scripts/download_dataset.py --dataset mvtec --data-dir data
python scripts/download_dataset.py --dataset neu --data-dir data
```

### Code Quality
```bash
# Format and lint code after changes
black src/ dashboard/ scripts/ main.py
flake8 src/ dashboard/ scripts/ main.py
```

## Architecture Overview

### Core Module Structure
- **`src/data/`**: Dataset classes (`QualityInspectionDataset`, `EnhancedQualityInspectionDataset`)
- **`src/models/`**: CNN architectures (`QualityInspectionCNN` with configurable backbones)
- **`src/training/`**: Training logic (`QualityInspectionTrainer`, `ImprovedQualityInspectionTrainer`)
- **`src/evaluation/`**: Model evaluation (`ModelEvaluator` with comprehensive metrics)
- **`src/explainability/`**: AI explanation methods (`ModelExplainer` with LIME/SHAP/GradCAM)
- **`src/utils/`**: Utilities (`metrics.py`, `visualization.py`)

### Key Classes
- **`QualityInspectionCNN`**: Main model class supporting ResNet50/EfficientNet/VGG16 backbones
- **`QualityInspectionTrainer`**: Handles training loop with callbacks and validation
- **`ModelExplainer`**: Generates multi-method explanations for predictions
- **`ModelEvaluator`**: Comprehensive evaluation with metrics and visualizations

### Data Flow
1. **Data Loading**: Dataset classes handle preprocessing and augmentation
2. **Training**: Trainer manages epochs, validation, and model checkpointing  
3. **Evaluation**: Evaluator generates performance metrics and reports
4. **Explanation**: Explainer creates visual explanations for model decisions
5. **Dashboard**: Streamlit interface provides interactive access

## Development Patterns

### Code Style
- **Format**: Black (88 char lines)
- **Linting**: Flake8
- **Naming**: PEP 8 (CamelCase classes, snake_case functions)
- **Documentation**: Google-style docstrings
- **Type hints**: Used throughout

### Model Development
- All models inherit from `nn.Module`
- Custom weight initialization via `_initialize_weights()`
- Feature extraction methods: `get_features()`, `get_feature_maps()`
- Configurable architectures via constructor arguments

### Training Workflow  
- Configuration via argparse in `main.py`
- Automatic directory creation for results
- Model checkpointing in `results/models/`
- Training logs in `results/logs/`
- Explanations saved to `results/explanations/`

## Testing and Validation

### Quick Test Pipeline
```bash
# Test full pipeline with minimal epochs
python main.py --mode full --epochs 5

# Test dashboard
streamlit run dashboard/app.py
```

### Model Architecture Testing
```bash
# Test different architectures
python main.py --mode train --model-type resnet50 --epochs 5
python main.py --mode train --model-type efficientnet --epochs 5  
python main.py --mode train --model-type vgg16 --epochs 5
python main.py --mode train --model-type simple --epochs 5
```

## Common Development Tasks

### Adding New Model Architecture
1. Extend `QualityInspectionCNN.__init__()` with new backbone
2. Update `create_model()` function in `src/models/cnn_model.py`
3. Add choice to argparse in `main.py`
4. Test with `python main.py --mode train --model-type new_model --epochs 5`

### Adding New Dataset
1. Create dataset class in `src/data/dataset.py`
2. Update `get_data_loaders()` function
3. Add download script in `scripts/download_dataset.py`
4. Test loading with dashboard or training pipeline

### Adding New Explainability Method
1. Extend `ModelExplainer` class in `src/explainability/explain_model.py`
2. Add method to `explain_image()` function
3. Update visualization in `utils/visualization.py`
4. Test with explanation pipeline

## Important Notes

- **GPU Support**: Code handles both CPU and GPU automatically
- **Dataset Fallback**: Uses `DummyDataset` if real data unavailable
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Memory Management**: Efficient handling of large datasets and models
- **Cross-platform**: Works on macOS, Linux, Windows with proper path handling

## File Organization

- **Entry Point**: `main.py` orchestrates all operations
- **Results**: All outputs go to `results/` subdirectories
- **Data**: Datasets stored in `data/` with train/test splits
- **Dashboard**: Interactive interface in `dashboard/app.py`
- **Scripts**: Utility scripts in `scripts/` directory

Always run code quality checks after modifications and test the main pipeline to ensure functionality.