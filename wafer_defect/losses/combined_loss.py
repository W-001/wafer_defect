"""
Combined Loss: Unified loss function combining Gate, Fine, and Metric losses.

L_total = gate_weight * L_gate + fine_weight * L_fine + metric_weight * L_metric

This combines:
  - Gate Loss: Binary classification (Nuisance vs Defect)
  - Fine Loss: Multi-class classification (Defect types)
  - Metric Loss: Supervised contrastive learning (optional)
"""

import torch
import torch.nn as nn

from .gate_loss import GateLoss
from .fine_loss import FineLoss
from .metric_loss import MetricLoss
from .dinomaly_loss import DinomalyLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for the full wafer defect classification model.

    Combines gate classification loss, fine-grained defect type classification
    loss, and optional supervised contrastive learning loss.

    Total loss formula:
        L_total = gate_weight * L_gate + fine_weight * L_fine + metric_weight * L_metric

    Args:
        gate_weight: Weight for gate loss (default: 1.0)
        fine_weight: Weight for fine loss (default: 0.5)
        metric_weight: Weight for metric loss (default: 0.1)
        defect_weight: Weight for Defect class in gate loss (default: 3.0)
        use_metric_loss: Whether to include metric loss (default: True)
        use_dinomaly_loss: Whether to include Dinomaly reconstruction loss (default: False)
        dinomaly_weight: Weight for Dinomaly loss (default: 0.1)
    """

    def __init__(
        self,
        gate_weight: float = 1.0,
        fine_weight: float = 0.5,
        metric_weight: float = 0.1,
        defect_weight: float = 3.0,
        use_metric_loss: bool = True,
        use_dinomaly_loss: bool = False,
        dinomaly_weight: float = 0.1,
    ):
        super().__init__()

        # Initialize individual losses
        self.gate_loss = GateLoss(defect_weight=defect_weight)
        self.fine_loss = FineLoss()

        self.use_metric_loss = use_metric_loss
        if use_metric_loss:
            self.metric_loss = MetricLoss()

        self.use_dinomaly_loss = use_dinomaly_loss
        if use_dinomaly_loss:
            self.dinomaly_loss = DinomalyLoss()

        # Loss weights
        self.gate_weight = gate_weight
        self.fine_weight = fine_weight
        self.metric_weight = metric_weight
        self.dinomaly_weight = dinomaly_weight

    def forward(
        self,
        gate_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        features: torch.Tensor,
        is_defect_target: torch.Tensor,
        defect_target: torch.Tensor,
        en_groups: list = None,
        de_groups: list = None,
        tau_percent: float = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            gate_logits: [B, 2] gate head logits (Nuisance vs Defect)
            fine_logits: [B, num_defect_types] fine head logits (Defect types)
            features: [B, D] embedding features for metric loss
            is_defect_target: [B] binary targets (0=Nuisance, 1=Defect)
            defect_target: [B] defect type labels (only valid when is_defect=True)
            en_groups: Optional list of encoder groups for Dinomaly loss
            de_groups: Optional list of decoder groups for Dinomaly loss
            tau_percent: Optional tau warmup value for Dinomaly loss

        Returns:
            Dictionary containing:
                - 'total': Combined total loss
                - 'gate': Gate loss value
                - 'fine': Fine loss value
                - 'metric': Metric loss value (if enabled)
                - 'dinomaly': Dinomaly loss value (if enabled)
        """
        # Gate loss (Nuisance vs Defect)
        loss_gate = self.gate_loss(gate_logits, is_defect_target)

        # Fine loss (defect type)
        loss_fine = self.fine_loss(
            fine_logits,
            defect_target,
            is_defect_target
        )

        # Start with gate + fine losses
        total_loss = (
            self.gate_weight * loss_gate +
            self.fine_weight * loss_fine
        )

        losses = {
            "total": total_loss,
            "gate": loss_gate,
            "fine": loss_fine,
        }

        # Metric loss (SupCon) - only on defect samples
        if self.use_metric_loss:
            loss_metric = self.metric_loss(
                features,
                defect_target,
                mask=is_defect_target.bool()
            )
            total_loss = total_loss + self.metric_weight * loss_metric
            losses["metric"] = loss_metric

        # Dinomaly loss (Loose Reconstruction)
        if self.use_dinomaly_loss and en_groups is not None and de_groups is not None:
            loss_dinomaly = self.dinomaly_loss(
                en_groups,
                de_groups,
                tau_percent=tau_percent
            )
            total_loss = total_loss + self.dinomaly_weight * loss_dinomaly
            losses["dinomaly"] = loss_dinomaly

        losses["total"] = total_loss
        return losses

    def extra_repr(self) -> str:
        parts = [
            f"gate_weight={self.gate_weight}",
            f"fine_weight={self.fine_weight}",
            f"defect_weight=3.0",
            f"use_metric_loss={self.use_metric_loss}",
        ]
        if self.use_metric_loss:
            parts.append(f"metric_weight={self.metric_weight}")
        if self.use_dinomaly_loss:
            parts.append(f"use_dinomaly_loss=True")
            parts.append(f"dinomaly_weight={self.dinomaly_weight}")
        return ", ".join(parts)
