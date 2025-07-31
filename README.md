# Explainable AI Quality Inspection

Automated defect detection for metal castings with PyTorch training, evaluation, explainability, and a Streamlit dashboard.

[![Python 3.8+](requirements.txt)](requirements.txt) • [![PyTorch MPS on macOS](requirements.txt)](requirements.txt)

## Overview
This repository provides an end-to-end pipeline to:
- Train a CNN classifier to detect casting defects.
- Evaluate model performance with standard metrics.
- Generate explainability overlays for model decisions.
- Explore predictions interactively via a Streamlit dashboard.

The project exposes a single CLI entry point in [main.py](main.py) with modes: `full`, `train`, `evaluate`, `explain`, and a dashboard in [dashboard/app.py](dashboard/app.py).

## Features
- Training: Configurable epochs, augmentation, checkpointing.
- Evaluation: Accuracy, precision, recall, F1, confusion matrix/ROC (depending on configuration).
- Explainability: Visual explanations (e.g., SHAP or similar) for images.
- Dashboard: Upload images, see predictions and explanations.
- CLI: One script orchestrates all workflows.

## Architecture Layout
```
.
├── main.py                      # CLI entry point: full/train/evaluate/explain
├── requirements.txt             # Python dependencies
├── dashboard/
│   └── app.py                   # Streamlit UI
├── src/
│   ├── training/
│   │   └── train_model.py
│   ├── evaluation/
│   │   └── evaluate_model.py
│   ├── explainability/
│   │   └── explain_model.py
│   └── data/
│       └── dataset.py
├── data/                        # Dataset root (user-provided or dummy auto-created)
│   ├── train/
│   │   ├── defective/
│   │   └── ok/
│   └── val/
│       ├── defective/
│       └── ok/
└── results/                     # Outputs (models, explanations, reports, experiments)
```

## Environment Setup (macOS, venv)
1) Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Optional: Verify PyTorch MPS (Apple Silicon) is usable:
```
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

## Installation
- Ensure you are at the repository root.
- Use the commands above to create a venv and install from [requirements.txt](requirements.txt).

## Dataset Structure
You can use your own dataset or rely on small dummy data auto-created when no dataset is found.

Expected structure for user-provided data:
```
data/
├── train/
│   ├── defective/*.jpg|*.jpeg|*.png
│   └── ok/*.jpg|*.jpeg|*.png
└── val/
    ├── defective/*.jpg|*.jpeg|*.png
    └── ok/*.jpg|*.jpeg|*.png
```

Note on missing download script: A dedicated downloader script such as `scripts/download_dataset.py` is not present. For now, manually place your dataset under `data/` as shown above. If `--download-data` is used and no dataset is present, the pipeline may fall back to a small dummy dataset to let you run end-to-end.

## Quick Start
Run the full pipeline (train → evaluate → explain) and then launch the dashboard:
```
# activate venv first
source venv/bin/activate

python main.py --mode full --download-data --epochs 30
streamlit run dashboard/app.py
```

## CLI Commands
All commands are run from the repository root:

- Full workflow:
```
python main.py --mode full --download-data --epochs 30
```

- Training only:
```
python main.py --mode train --epochs 30
```

- Evaluation only:
```
python main.py --mode evaluate --model-path results/models/best_model.pth
```

- Explainability only:
```
python main.py --mode explain --model-path results/models/best_model.pth
```

- Launch dashboard:
```
streamlit run dashboard/app.py
```

## macOS Notes (MPS)
- On Apple Silicon, PyTorch can accelerate via MPS. This is typically auto-detected in the code.
- If you see an error like “MPS backend not available”, enable CPU fallback:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
- Performance may vary based on PyTorch version and macOS updates. Ensure versions match those in [requirements.txt](requirements.txt).

## Expected Outputs
After running training/evaluation/explainability:
- `results/models/best_model.pth` — Best checkpoint by validation metric.
- `results/reports/evaluation_report.json` — Aggregate metrics and plot artifacts (if produced).
- `results/explanations/` — Visual explanations/overlays.
- `results/experiments/` — Logs/experiments metadata (e.g., TensorBoard if enabled).

## Troubleshooting
- Module not found (torch/streamlit):
  ```
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- MPS not available:
  ```
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- Out-of-memory during training:
  Reduce batch size or image size; adjust loaders in [src/training/train_model.py](src/training/train_model.py).
- Dashboard port conflicts:
  ```
  streamlit run dashboard/app.py --server.port 8502
  ```
- Dataset warnings/empty splits:
  Verify your files are in `data/train/{defective,ok}` and `data/val/{defective,ok}` with supported image extensions.

## License
Placeholder: Add a LICENSE file (e.g., MIT, Apache-2.0) and reference it here.

Tip: For command-line help:
```
python main.py --help