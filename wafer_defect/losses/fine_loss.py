"""
Fine Loss: Multi-class loss for defect type classification.

Only computes loss on samples labeled as True Defect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineLoss(nn.Module):
    """
    Multi-class loss for defect type classification.

    Only computes loss on samples labeled as True Defect.
    This ensures the fine-grained defect classifier only learns from
    actual defect samples while ignoring nuisance samples.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_defect: torch.Tensor = None,
        defect_labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on defect samples only.

        Args:
            logits: [B, num_classes] raw logits for defect type classification
            targets: [B] defect type labels (1~K for defect, 0~K-1 after adjustment)
            is_defect: [B] binary mask, True for defect samples (optional)
            defect_labels: [B] original defect type labels for mapping (optional)

        Returns:
            Scalar loss tensor. Returns 0 if no defect samples in batch.
        """
        if is_defect is None:
            # Train on all samples (assuming targets are 0~K where 0 may be nuisance)
            return F.cross_entropy(logits, targets)

        # Only train on defect samples
        mask = is_defect.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Return CE loss computed only on defect samples
        # Labels should be adjusted before this: defect types should map to 0~(K-1)
        return F.cross_entropy(logits[mask], targets[mask])

    def extra_repr(self) -> str:
        return "Fine-grained defect type classification loss (defect samples only)"
