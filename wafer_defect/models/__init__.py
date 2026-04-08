"""
Wafer Defect Classification Models - New Architecture.

Architecture:
- backbone.py: DINOv3 backbone (frozen)
- classification.py: Classification branch (Gate + Fine heads with shared feature tower)
- dinomaly.py: Dinomaly anomaly detection (open-source)
- open_set_detector.py: Open-set detection for unknown defects
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
from .defect_model import (
    WaferDefectModel,
    WaferDefectModelSimple,
)
from .open_set_detector import OpenSetDetector

# Dinomaly (open-source) - anomaly detection
try:
    from .dinomaly import DinomalyAnomalyDetector
    DINOMALY_AVAILABLE = True
except ImportError as e:
    DINOMALY_AVAILABLE = False
    DinomalyAnomalyDetector = None
    print(f"[Warning] Dinomaly not available: {e}")

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
    "DinomalyAnomalyDetector",
    "DINOMALY_AVAILABLE",
    # Open-set detection
    "OpenSetDetector",
    # Models
    "WaferDefectModel",
    "WaferDefectModelSimple",
]
