"""
Fine Head: True Defect multi-class classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineHead(nn.Module):
    """
    Second-stage classifier: specific defect type classification.
    Only applied to samples classified as True Defect by GateHead.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, D] input features
        Returns:
            dict with:
                - logits: [B, num_classes]
                - prob: [B, num_classes]
                - pred: [B] predicted class indices
        """
        logits = self.classifier(feat)
        prob = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        return {
            "logits": logits,
            "prob": prob,
            "pred": pred
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss, optionally only on masked samples.

        Args:
            logits: [B, num_classes]
            targets: [B] ground truth class indices
            mask: [B] boolean tensor, True for defect samples to train on
        """
        if mask is None:
            return F.cross_entropy(logits, targets)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        return F.cross_entropy(logits[mask], targets[mask])


class PrototypeClassifier(nn.Module):
    """
    Prototype-based classifier using class centers.
    Helps with few-shot and novel defect detection.
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Learnable class prototypes
        self.register_buffer('centers', torch.zeros(num_classes, feat_dim))

    def update_centers(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        momentum: float = 0.9
    ):
        """
        Update class centers based on current batch.

        Args:
            feats: [B, D] features
            labels: [B] class labels
            momentum: EMA update factor
        """
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                center = feats[mask].mean(dim=0)
                self.centers[c] = momentum * self.centers[c] + (1 - momentum) * center

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Compute distances to prototypes.

        Returns:
            dict with:
                - dists: [B, num_classes] distances to each prototype
                - logits: [B, num_classes] similarity scores
                - pred: [B] predicted class
        """
        # Compute squared Euclidean distances
        dists = torch.cdist(feat, self.centers)  # [B, num_classes]

        # Convert to similarity scores (higher = more similar)
        logits = -dists
        pred = logits.argmax(dim=-1)

        return {
            "dists": dists,
            "logits": logits,
            "pred": pred
        }
