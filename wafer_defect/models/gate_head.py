"""
Gate Head: Nuisance vs True Defect binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateHead(nn.Module):
    """
    First-stage classifier: Nuisance vs True Defect.

    This is the most critical business constraint - minimize misclassification
    between Nuisance (no defect) and True Defect.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 512,
        defect_weight: float = 3.0
    ):
        super().__init__()
        self.defect_weight = defect_weight

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, D] input features
        Returns:
            dict with:
                - logits: [B, 2] raw logits
                - prob: [B, 2] softmax probabilities
                - is_defect_pred: [B] predicted class (0 or 1)
        """
        logits = self.classifier(feat)
        prob = F.softmax(logits, dim=-1)

        # is_defect_pred = 1 means True Defect, 0 means Nuisance
        is_defect_pred = (prob[:, 1] > 0.5).long()

        return {
            "logits": logits,
            "prob": prob,
            "is_defect_pred": is_defect_pred
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: [B, 2]
            targets: [B] ground truth (0=Nuisance, 1=Defect)
        """
        # Create class weights with higher penalty for missing defects
        weight = torch.tensor([1.0, self.defect_weight], device=logits.device)
        return F.cross_entropy(logits, targets, weight=weight)


class UncertaintyHead(nn.Module):
    """
    Additional head to estimate prediction uncertainty.
    Helps with open-set rejection.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.uncertainty = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            uncertainty_score: [B, 1] higher = more uncertain
        """
        return torch.sigmoid(self.uncertainty(feat))
