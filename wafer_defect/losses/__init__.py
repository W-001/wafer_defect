"""
Loss functions for wafer defect classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateLoss(nn.Module):
    """
    Binary classification loss for Nuisance vs True Defect.
    Uses weighted cross-entropy with higher weight for Defect class.
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
        Args:
            logits: [B, 2] raw logits
            targets: [B] ground truth (0=Nuisance, 1=Defect)
        """
        weight = torch.tensor(
            [1.0, self.defect_weight],
            device=logits.device
        )
        return F.cross_entropy(logits, targets, weight=weight)


class FineLoss(nn.Module):
    """
    Multi-class loss for defect type classification.
    Only computes loss on samples labeled as True Defect.
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
        Args:
            logits: [B, num_classes] raw logits
            targets: [B] defect type labels (1~K)
            is_defect: [B] binary mask, True for defect samples
            defect_labels: [B] original defect type labels (for mapping)
        """
        if is_defect is None:
            # Train on all samples (assuming targets are 0~K where 0 may be nuisance)
            return F.cross_entropy(logits, targets)

        # Only train on defect samples
        mask = is_defect.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Adjust labels: defect labels start from 1, map to 0~(K-1)
        # Assuming labels are already adjusted before this
        return F.cross_entropy(logits[mask], targets[mask])


class MetricLoss(nn.Module):
    """
    Supervised Contrastive Loss for better embedding structure.
    Pulls same-class samples together, pushes different-class samples apart.
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
        Args:
            features: [B, D] embedding features
            labels: [B] class labels
            mask: [B] optional mask for valid samples
        """
        if mask is not None:
            features = features[mask]
            labels = labels[mask]

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Create labels matrix
        labels = labels.unsqueeze(0)
        same_label = (labels == labels.T).float()

        # Mask out self-contrastive
        mask = torch.eye(labels.size(1), device=features.device).bool()
        same_label = same_label.masked_fill(mask, 0)

        # Compute loss: -log(exp(pos) / sum(exp))
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        num_pos = same_label.sum(dim=1)
        loss = -(same_label * log_prob).sum(dim=1) / (num_pos + 1e-8)

        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center loss to minimize intra-class variation.
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
        Args:
            features: [B, D]
            labels: [B]
        """
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for the full model:
    L = L_gate + λ1 * L_fine + λ2 * L_metric
    """

    def __init__(
        self,
        gate_weight: float = 1.0,
        fine_weight: float = 0.5,
        metric_weight: float = 0.1,
        defect_weight: float = 3.0,
        use_metric_loss: bool = True
    ):
        super().__init__()

        self.gate_loss = GateLoss(defect_weight=defect_weight)
        self.fine_loss = FineLoss()
        self.use_metric_loss = use_metric_loss

        if use_metric_loss:
            self.metric_loss = MetricLoss()

        self.gate_weight = gate_weight
        self.fine_weight = fine_weight
        self.metric_weight = metric_weight

    def forward(
        self,
        gate_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        features: torch.Tensor,
        is_defect_target: torch.Tensor,
        defect_target: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.

        Returns:
            dict with total_loss and individual losses
        """
        # Gate loss (Nuisance vs Defect)
        loss_gate = self.gate_loss(gate_logits, is_defect_target)

        # Fine loss (defect type)
        loss_fine = self.fine_loss(
            fine_logits,
            defect_target,
            is_defect_target
        )

        total_loss = (
            self.gate_weight * loss_gate +
            self.fine_weight * loss_fine
        )

        losses = {
            "total": total_loss,
            "gate": loss_gate,
            "fine": loss_fine
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
            losses["total"] = total_loss

        return losses
