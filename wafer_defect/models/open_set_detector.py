"""
Open-set defect detection for identifying unknown defect types.

Combines multiple signals:
1. Dinomaly reconstruction error
2. Distance to class centers
3. Energy score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenSetDetector(nn.Module):
    """
    Open-set defect detection combining multiple signals.

    Combines:
    1. Dinomaly reconstruction error
    2. Distance to class centers
    3. Energy score
    """

    def __init__(
        self,
        threshold_dinomaly: float = 0.5,
        threshold_center: float = 2.0,
        dinomaly_weight: float = 0.6,
        center_weight: float = 0.4,
    ):
        super().__init__()
        self.threshold_dinomaly = threshold_dinomaly
        self.threshold_center = threshold_center
        self.dinomaly_weight = dinomaly_weight
        self.center_weight = center_weight

        # Class centers for known defect types
        self.register_buffer('class_centers', torch.zeros(1, 1))  # placeholder
        self.num_classes = 0

    def set_class_centers(self, centers: torch.Tensor):
        """
        Set class centers for distance-based anomaly detection.

        Args:
            centers: [num_classes, embed_dim] class center embeddings
        """
        self.class_centers = centers
        self.num_classes = centers.shape[0]

    def compute_center_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to nearest class center.

        Args:
            features: [B, embed_dim] feature embeddings

        Returns:
            distances: [B] distance to nearest center
        """
        if self.num_classes == 0:
            return torch.zeros(features.shape[0], device=features.device)

        # Normalize features and centers
        features = F.normalize(features, dim=-1)
        centers = F.normalize(self.class_centers, dim=-1)

        # Compute cosine similarity
        sim = torch.matmul(features, centers.t())  # [B, num_classes]
        dist = 1.0 - sim.max(dim=-1)[0]  # [B] distance to nearest

        return dist

    def forward(
        self,
        anomaly_score: torch.Tensor,
        features: torch.Tensor = None,
    ) -> dict:
        """
        Determine if sample is unknown defect.

        Args:
            anomaly_score: [B] Dinomaly anomaly scores
            features: [B, embed_dim] optional features for center distance

        Returns:
            dict with:
                - is_unknown: [B] binary prediction
                - combined_score: [B] weighted combination
                - anomaly_score: original anomaly score
                - center_distance: distance to nearest center (if features provided)
        """
        result = {
            'anomaly_score': anomaly_score,
            'is_unknown': None,
            'combined_score': anomaly_score,
        }

        # Center distance
        if features is not None and self.num_classes > 0:
            center_dist = self.compute_center_distance(features)
            result['center_distance'] = center_dist

            # Combined score
            combined = (
                self.dinomaly_weight * anomaly_score +
                self.center_weight * center_dist
            )
            result['combined_score'] = combined
        else:
            combined = anomaly_score

        # Threshold
        threshold = self.threshold_dinomaly
        result['is_unknown'] = (combined > threshold).long()

        return result