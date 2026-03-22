"""
Anomaly Head: Detect unknown/novel defects based on embedding distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyHead(nn.Module):
    """
    Anomaly detection based on distances to class centers in embedding space.
    Samples far from all known classes are flagged as potential novel defects.
    """

    def __init__(self, feat_dim: int, num_classes: int, k: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.k = k

        # Class centers for known defect types
        self.register_buffer('centers', torch.zeros(num_classes, feat_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))

        # Energy score parameters
        self.energy_linear = nn.Linear(feat_dim, 1)

    def update_centers(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        momentum: float = 0.95
    ):
        """
        Update class centers with EMA.

        Args:
            feats: [B, D] features
            labels: [B] class labels
            momentum: EMA factor
        """
        # Ensure centers are on same device as feats
        centers = self.centers.to(feats.device)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_mean = feats[mask].mean(dim=0)
                centers[c] = centers[c] * momentum + class_mean * (1 - momentum)
        self.centers = centers

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Compute anomaly scores.

        Returns:
            dict with:
                - min_dist: [B] distance to nearest class center
                - energy: [B] energy score
                - anomaly_score: [B] combined anomaly score (higher = more anomalous)
        """
        # Ensure centers are on same device as feat
        centers = self.centers.to(feat.device)

        # Distance to nearest class center
        dists = torch.cdist(feat, centers)  # [B, num_classes]
        min_dist, nearest_class = dists.min(dim=1)  # [B]

        # Energy score
        energy = torch.logsumexp(self.energy_linear(feat), dim=1)  # [B]

        # Combined anomaly score (can be tuned)
        # Higher min_dist = more anomalous
        # Higher energy = more out-of-distribution
        anomaly_score = min_dist  # Simple version

        return {
            "dists": dists,
            "min_dist": min_dist,
            "nearest_class": nearest_class,
            "energy": energy,
            "anomaly_score": anomaly_score
        }

    def detect_anomaly(
        self,
        feat: torch.Tensor,
        threshold: float = None
    ) -> dict:
        """
        Detect if samples are anomalous (unknown defects).

        Args:
            feat: [B, D] features
            threshold: if min_dist > threshold, flag as anomaly

        Returns:
            dict with:
                - is_anomaly: [B] boolean, True if anomaly
                - anomaly_score: [B] raw score
        """
        output = self.forward(feat)

        if threshold is None:
            # Use statistics: more than 2 std from mean
            threshold = self.centers.std() * 2 + self.centers.mean()

        is_anomaly = output["min_dist"] > threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": output["anomaly_score"],
            "nearest_class": output["nearest_class"],
            "min_dist": output["min_dist"]
        }


class KNNDensityEstimator(nn.Module):
    """
    k-NN based density estimation for anomaly detection.
    """

    def __init__(self, feat_dim: int, k: int = 10):
        super().__init__()
        self.feat_dim = feat_dim
        self.k = k

        self.register_buffer('ref_feats', torch.zeros(1, feat_dim))
        self.ref_labels = []

    def update_reference(self, feats: torch.Tensor, labels: torch.Tensor):
        """Add features to reference set."""
        if self.ref_feats.size(0) == 1:
            self.ref_feats = feats
        else:
            self.ref_feats = torch.cat([self.ref_feats, feats], dim=0)
        self.ref_labels.extend(labels.tolist())

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute k-NN density score.

        Returns:
            density: [B] higher = more densely connected (normal)
        """
        # Compute distances to reference set
        dists = torch.cdist(feat, self.ref_feats)  # [B, N]

        # Take k nearest neighbors
        knn_dists, _ = dists.topk(self.k, largest=False, dim=1)  # [B, k]

        # Mean distance to k nearest
        density = -knn_dists.mean(dim=1)  # Negative so higher = more anomalous

        return density
