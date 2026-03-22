"""
Wafer Defect Classification Package.

Based on the hierarchical open-set classification framework:
- DINOv3 backbone with multi-view fusion
- Gate head: Nuisance vs True Defect binary classification
- Fine head: Defect type multi-class classification
- Anomaly head: Unknown/novel defect detection
"""

__version__ = "0.1.0"
