"""
Training engine for wafer defect classification.

Submodules:
- trainer: WaferDefectTrainer with three-phase training support
- sampler: LongTailSampler for imbalanced datasets
- collate: CollateFn for multi-view data packing

Three-Phase Training:
    Phase 1: Train classification branch (Gate + Fine heads)
    Phase 2: Train Dinomaly2 decoder
    Phase 3: Joint fine-tuning (optional)
"""

from .trainer import WaferDefectTrainer, ThreePhaseTrainer, generate_markdown_report
from .sampler import LongTailSampler, BalancedBatchSampler
from .collate import MultiViewCollateFn, PairedCollateFn

__all__ = [
    # Trainer
    "WaferDefectTrainer",
    "ThreePhaseTrainer",
    "generate_markdown_report",
    # Sampler
    "LongTailSampler",
    "BalancedBatchSampler",
    # Collate
    "MultiViewCollateFn",
    "PairedCollateFn",
]

# Training phases
PHASE_CLASSIFICATION = "classification"   # Phase 1: Gate + Fine heads
PHASE_DINOMALITY2 = "dinomaly2"           # Phase 2: Dinomaly2 decoder
PHASE_JOINT = "joint"                     # Phase 3: Joint fine-tuning

__all__ += [
    "PHASE_CLASSIFICATION",
    "PHASE_DINOMALITY2",
    "PHASE_JOINT",
]
