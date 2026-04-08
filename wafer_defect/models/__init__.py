"""
Wafer Defect Classification Models - New Architecture.

Architecture:
- backbone.py: DINOv3 backbone (frozen)
- classification.py: Classification branch (Gate + Fine heads with shared feature tower)
- dinomaly2.py: Dinomaly2 anomaly detection branch
- defect_model.py: Complete model composition
- fusion.py: Multi-view fusion (optional)
"""

# Core exports - new architecture
from .backbone import DINOv3Backbone
from .fusion import MultiViewFusion, ViewLevelAttention
from .classification import (
    ClassificationBranch,
    GateHead,
    FineHead,
    GateToFineModulation,
    PrototypeClassifier,
)
from .dinomaly2 import (
    Dinomaly2Branch,
    Dinomaly2Loss,
    OpenSetDetector,
)
from .defect_model import (
    WaferDefectModel,
    WaferDefectModelSimple,
)

# Dinomaly head (used internally by Dinomaly2Branch)
from .dinomaly_head import (
    Dinomaly2AnomalyHead,
    NoisyBottleneck2,
    LinearAttention2,
    ViTill2,
    loose_reconstruction_loss,
)

__all__ = [
    # Backbone
    "DINOv3Backbone",
    # Fusion
    "MultiViewFusion",
    "ViewLevelAttention",
    # Classification branch
    "ClassificationBranch",
    "GateHead",
    "FineHead",
    "GateToFineModulation",
    "PrototypeClassifier",
    # Anomaly detection
    "Dinomaly2Branch",
    "Dinomaly2AnomalyHead",
    "Dinomaly2Loss",
    "OpenSetDetector",
    # Models
    "WaferDefectModel",
    "WaferDefectModelSimple",
]
