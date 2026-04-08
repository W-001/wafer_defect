# Wafer Defect Classification

A DINOv3-based hierarchical open-set classification framework for wafer defect detection.

## Architecture

```
Input Image → DINOv3 Backbone (frozen) → Shared Feature Tower
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                               Gate Head          Fine Head
                              (Binary)           (Multi-class)
                                    ↓                   ↓
                                 Nuisance         Defect Type
```

**Key Features**:
- DINOv3 Backbone (frozen)
- Shared Feature Tower + Gate/Fine Classification Heads
- Dinomaly Anomaly Detection (open-source)
- Optional 3-View Fusion

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (for GPU training)

## Installation

```bash
conda activate py310
```

## Training

```bash
# Synthetic data (quick test)
set PYTHONPATH=. && python wafer_defect/train.py --synthetic --epochs 10 --device cpu

# Real data
set PYTHONPATH=. && python wafer_defect/train.py --data_dir /path/to/wafer_data --epochs 10 --device cuda

# Full training with Dinomaly
set PYTHONPATH=. && python wafer_defect/train.py --data_dir /path/to/wafer_data --use_dinomaly --dinomaly_iters 100 --epochs 10 --device cuda
```

## Inference

```bash
set PYTHONPATH=. && python wafer_defect/inference.py --checkpoint output/best_model.pt --data_dir /path/to/test_data --device cuda --output_dir output/inference
```

## Data Format

Real data folder structure:
```
data/
├── Nuisance/
│   └── *.jpg
├── Scratch/
│   └── *.jpg
└── Particle/
    └── *.jpg
```

**3-View Mode** (with `--three_views`):
- Filenames with `I00K`/`I01K`/`I02K` represent three views of the same defect

## Project Structure

```
wafer_defect/
├── models/              # Model definitions
│   ├── backbone.py      # DINOv3 wrapper
│   ├── classification.py # Classification branch (Gate + Fine)
│   ├── dinomaly.py     # Dinomaly anomaly detection
│   ├── defect_model.py  # Main model
│   ├── fusion.py        # 3-view fusion
│   └── open_set_detector.py # Open-set detection
├── losses/              # Loss functions
│   ├── gate_loss.py     # Gate loss
│   ├── fine_loss.py     # Fine loss
│   ├── metric_loss.py   # Metric loss
│   └── combined_loss.py # Combined loss
├── engine/              # Training engine
│   ├── trainer.py        # Trainer
│   ├── sampler.py        # Long-tail sampling
│   └── collate.py       # Data collate
├── data/                # Data processing
│   ├── dataset.py        # Dataset
│   └── preprocessor.py  # Preprocessor
├── utils/               # Utilities
│   ├── metrics.py        # Evaluation metrics
│   └── data_inspector.py # Data inspector
├── inference.py          # Inference entry
└── train.py             # Training entry
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--img_size` | 224 | Image resize size |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 10 | Number of epochs |
| `--lr` | 1e-4 | Learning rate |
| `--defect_weight` | 3.0 | Gate loss weight for defect class |
| `--dinomaly_iters` | 10000 | Dinomaly decoder training iterations |

## License

This project uses the DINOv3 model which has its own license. Please refer to the original DINOv3 repository for license details.
