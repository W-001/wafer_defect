"""
Metric Loss: Supervised Contrastive Loss for better embedding structure.

Pulls same-class samples together, pushes different-class samples apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) for improved embedding structure.

    This loss encourages samples of the same class to have similar embeddings
    while samples of different classes should have dissimilar embeddings.

    Based on: "Supervised Contrastive Learning" (Khosla et al., 2020)

    Args:
        temperature: Temperature parameter for contrastive scaling (default: 0.1)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: [B, D] embedding features (will be normalized)
            labels: [B] class labels for supervised contrastive learning
            mask: [B] optional boolean mask for valid samples

        Returns:
            Scalar loss tensor. Returns 0 if no positive pairs exist.
        """
        if mask is not None:
            features = features[mask]
            labels = labels[mask]

        # Handle empty batch
        if features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        # Normalize features to unit hypersphere
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix: [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # Create positive mask: same label pairs (excluding self-contrastive)
        labels = labels.unsqueeze(0)
        same_label = (labels == labels.T).float()

        # Mask out self-contrastive pairs (diagonal)
        batch_size = labels.size(1)
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        same_label = same_label.masked_fill(self_mask, 0)

        # Compute loss: -log(exp(pos) / sum(exp))
        exp_sim = torch.exp(sim)
        # Subtract max for numerical stability
        exp_sim = exp_sim - sim.amax(dim=1, keepdim=True).detach()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        num_pos = same_label.sum(dim=1)

        # Guard: if no positive pairs exist in a batch, return 0 instead of NaN
        if num_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        loss = -(same_label * log_prob).sum(dim=1) / (num_pos + 1e-8)
        return loss.mean()

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class CenterLoss(nn.Module):
    """
    Center loss to minimize intra-class variation.

    Learns a center (mean feature) for each class and penalizes
    the distance between samples and their class centers.

    Args:
        feat_dim: Dimensionality of the embedding features
        num_classes: Number of classes
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute center loss.

        Args:
            features: [B, D] embedding features
            labels: [B] class labels

        Returns:
            Scalar loss tensor (mean squared distance to class centers)
        """
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss

    def extra_repr(self) -> str:
        return f"feat_dim={self.feat_dim}, num_classes={self.num_classes}"
