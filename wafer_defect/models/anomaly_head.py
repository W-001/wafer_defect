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

    Uses normalized combination of:
    - Distance to nearest class center
    - Energy score
    """

    def __init__(self, feat_dim: int, num_classes: int, k: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.k = k

        # Class centers for known defect types
        self.register_buffer('centers', torch.zeros(num_classes, feat_dim))
        self.register_buffer('center_std', torch.zeros(1))  # For normalization
        self.register_buffer('class_counts', torch.zeros(num_classes))

        # Energy score parameters
        self.energy_linear = nn.Linear(feat_dim, 1)

        # Statistics for normalization
        self.register_buffer('dist_mean', torch.tensor(0.0))
        self.register_buffer('dist_std', torch.tensor(1.0))
        self.register_buffer('energy_mean', torch.tensor(0.0))
        self.register_buffer('energy_std', torch.tensor(1.0))

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

        # Update center statistics for normalization
        if self.training and centers.std() > 0:
            center_std = self.center_std.to(feats.device)
            center_std = centers.std()
            self.center_std = center_std

        self.centers = centers

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Compute anomaly scores.

        Returns:
            dict with:
                - min_dist: [B] distance to nearest class center
                - energy: [B] energy score
                - anomaly_score: [B] combined normalized anomaly score
                - normalized_dist: [B] z-score of distance
                - normalized_energy: [B] z-score of energy
        """
        # Ensure centers are on same device as feat
        centers = self.centers.to(feat.device)

        # Distance to nearest class center
        dists = torch.cdist(feat, centers)  # [B, num_classes]
        min_dist, nearest_class = dists.min(dim=1)  # [B]

        # Energy score
        energy = torch.logsumexp(self.energy_linear(feat), dim=1)  # [B]

        # Normalize scores using running statistics
        dist_mean = self.dist_mean.to(feat.device)
        dist_std = self.dist_std.to(feat.device)
        energy_mean = self.energy_mean.to(feat.device)
        energy_std = self.energy_std.to(feat.device)

        # Avoid division by zero
        dist_std = torch.where(dist_std > 1e-6, dist_std, torch.ones_like(dist_std))
        energy_std = torch.where(energy_std > 1e-6, energy_std, torch.ones_like(energy_std))

        # Z-score normalization
        normalized_dist = (min_dist - dist_mean) / dist_std
        normalized_energy = (energy - energy_mean) / energy_std

        # Combined anomaly score (higher = more anomalous)
        # Weight distance more as it's more interpretable for defect detection
        anomaly_score = 0.7 * normalized_dist + 0.3 * normalized_energy

        return {
            "dists": dists,
            "min_dist": min_dist,
            "nearest_class": nearest_class,
            "energy": energy,
            "anomaly_score": anomaly_score,
            "normalized_dist": normalized_dist,
            "normalized_energy": normalized_energy
        }

    def update_statistics(self, feats: torch.Tensor, defect_mask: torch.Tensor = None):
        """
        Update running statistics for normalization.
        Should be called after training epoch on known defect samples.

        Args:
            feats: [B, D] features
            defect_mask: [B] boolean, True for known defect samples
        """
        if defect_mask is None:
            defect_mask = torch.ones(feats.size(0), dtype=torch.bool, device=feats.device)

        if defect_mask.sum() < 10:  # Need minimum samples
            return

        output = self.forward(feats[defect_mask])

        # Update running statistics with EMA
        alpha = 0.9

        dist_mean = self.dist_mean.to(feats.device)
        dist_std = self.dist_std.to(feats.device)
        energy_mean = self.energy_mean.to(feats.device)
        energy_std = self.energy_std.to(feats.device)

        dist_mean_new = output["min_dist"].mean()
        dist_std_new = output["min_dist"].std() + 1e-6
        energy_mean_new = output["energy"].mean()
        energy_std_new = output["energy"].std() + 1e-6

        self.dist_mean = dist_mean * alpha + dist_mean_new * (1 - alpha)
        self.dist_std = dist_std * alpha + dist_std_new * (1 - alpha)
        self.energy_mean = energy_mean * alpha + energy_mean_new * (1 - alpha)
        self.energy_std = energy_std * alpha + energy_std_new * (1 - alpha)

    def detect_anomaly(
        self,
        feat: torch.Tensor,
        threshold: float = None
    ) -> dict:
        """
        Detect if samples are anomalous (unknown defects).

        Args:
            feat: [B, D] features
            threshold: if anomaly_score > threshold, flag as anomaly

        Returns:
            dict with:
                - is_anomaly: [B] boolean, True if anomaly
                - anomaly_score: [B] raw score
        """
        output = self.forward(feat)

        if threshold is None:
            # Use normalized threshold (z-score > 2 means significantly different)
            threshold = 2.0

        is_anomaly = output["anomaly_score"] > threshold

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
