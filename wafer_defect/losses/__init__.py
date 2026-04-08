"""
Loss functions for wafer defect classification.

New architecture losses:

1. GateLoss: Binary classification (Nuisance vs Defect)
   - Weighted cross-entropy with defect_weight penalty

2. FineLoss: Multi-class classification (Defect types)
   - Standard cross-entropy on defect samples only

3. MetricLoss: Supervised contrastive learning
   - Pulls same-class samples together, pushes different-class samples apart

4. DinomalyLoss: Loose reconstruction for Dinomaly2
   - Computes reconstruction error between encoder and decoder features

5. CombinedLoss: Unified loss combining all above
   - L_total = L_gate + λ1*L_fine + λ2*L_metric

Example usage:
    from wafer_defect.losses import GateLoss, FineLoss, CombinedLoss

    gate_loss = GateLoss(defect_weight=3.0)
    fine_loss = FineLoss()

    combined_loss = CombinedLoss(
        gate_weight=1.0,
        fine_weight=0.5,
        metric_weight=0.1,
        defect_weight=3.0
    )
"""

from .gate_loss import GateLoss
from .fine_loss import FineLoss
from .metric_loss import MetricLoss, CenterLoss
from .dinomaly_loss import DinomalyLoss, loose_reconstruction_loss
from .combined_loss import CombinedLoss

__all__ = [
    "GateLoss",
    "FineLoss",
    "MetricLoss",
    "CenterLoss",
    "DinomalyLoss",
    "CombinedLoss",
    "loose_reconstruction_loss",
]
