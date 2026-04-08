"""
Gate Loss: Binary classification loss for Nuisance vs True Defect.

Uses weighted cross-entropy with higher weight for Defect class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateLoss(nn.Module):
    """
    Binary classification loss for Nuisance vs True Defect.

    Uses weighted cross-entropy with higher weight for Defect class.

    Args:
        defect_weight: Weight multiplier for Defect class (default: 3.0)
    """

    def __init__(self, defect_weight: float = 3.0):
        super().__init__()
        self.defect_weight = defect_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: [B, 2] raw logits for binary classification
            targets: [B] ground truth labels (0=Nuisance, 1=Defect)

        Returns:
            Scalar loss tensor
        """
        weight = torch.tensor(
            [1.0, self.defect_weight],
            device=logits.device,
            dtype=logits.dtype
        )
        return F.cross_entropy(logits, targets, weight=weight)

    def extra_repr(self) -> str:
        return f"defect_weight={self.defect_weight}"
