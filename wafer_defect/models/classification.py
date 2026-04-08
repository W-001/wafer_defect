"""
Classification Branch: Shared Feature Tower + Gate/Fine Classification Heads.

This module implements:
- Shared Feature Tower: LayerNorm + Linear + GELU + Dropout
- Gate Head: Nuisance vs Defect binary classification
- Fine Head: Defect type multi-class classification
- Gate-to-Fine Feature Modulation: Gate confidence modulates Fine features

Based on the refactored architecture design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateHead(nn.Module):
    """
    First-stage classifier: Nuisance vs True Defect.

    Critical business constraint - minimize misclassification
    between Nuisance (no defect) and True Defect.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        defect_weight: float = 3.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.defect_weight = defect_weight

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, D] input features
        Returns:
            dict with:
                - logits: [B, 2] raw logits
                - prob: [B, 2] softmax probabilities
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
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: [B, 2]
            targets: [B] ground truth (0=Nuisance, 1=Defect)
        """
        weight = torch.tensor([1.0, self.defect_weight], device=logits.device)
        return F.cross_entropy(logits, targets, weight=weight)


class FineHead(nn.Module):
    """
    Second-stage classifier: specific defect type classification.
    Only applied to samples classified as True Defect by GateHead.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = None
    ):
        super().__init__()
        self.num_classes = num_classes
        hidden_dim = hidden_dim or input_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
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


class GateToFineModulation(nn.Module):
    """
    Gate-to-Fine feature modulation.
    Uses Gate classification probabilities to modulate Fine features.
    """

    def __init__(self, gate_num_classes: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.gate_num_classes = gate_num_classes
        self.hidden_dim = hidden_dim

        self.modulation = nn.Sequential(
            nn.Linear(gate_num_classes, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, fine_feat: torch.Tensor, gate_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine_feat: [B, D] Fine head features
            gate_prob: [B, 2] Gate classification probabilities
        Returns:
            modulated_feat: [B, D] modulated features
        """
        gate_mod = self.modulation(gate_prob)  # [B, D]
        return fine_feat * gate_mod


class ClassificationBranch(nn.Module):
    """
    Shared Feature Tower + Task-Specific Classification Heads.

    Architecture:
        Backbone Features → Shared Feature Tower → Gate Head
                                                    ↓
                                              Gate-to-Fine Modulation
                                                    ↓
                                              Fine Head

    The shared feature tower reduces redundancy and improves feature reuse.
    Gate-to-Fine modulation allows Gate predictions to influence Fine classification.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        num_defect_classes: int = 10,
        defect_weight: float = 3.0,
        use_gate_modulation: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_defect_classes = num_defect_classes
        self.use_gate_modulation = use_gate_modulation

        # Shared Feature Tower
        self.feature_tower = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Gate Head (Nuisance vs Defect)
        self.gate_head = GateHead(
            input_dim=hidden_dim,
            num_classes=2,
            defect_weight=defect_weight
        )

        # Fine Head (Defect Type Classification)
        self.fine_head = FineHead(
            input_dim=hidden_dim,
            num_classes=num_defect_classes,
            hidden_dim=hidden_dim
        )

        # Gate-to-Fine Feature Modulation (optional)
        if use_gate_modulation:
            self.gate_to_fine_mod = GateToFineModulation(
                gate_num_classes=2,
                hidden_dim=hidden_dim
            )
        else:
            self.gate_to_fine_mod = None

    def forward(
        self,
        feat: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        Args:
            feat: [B, D] backbone features
            return_features: if True, also return shared features
        Returns:
            dict with:
                - gate: {logits, prob, pred}
                - fine: {logits, prob, pred}
                - features: [B, hidden_dim] shared features (if return_features)
        """
        # Shared Feature Tower
        shared_feat = self.feature_tower(feat)

        # Gate Classification
        gate_out = self.gate_head(shared_feat)

        # Gate-to-Fine Feature Modulation
        if self.use_gate_modulation and self.gate_to_fine_mod is not None:
            modulated_feat = self.gate_to_fine_mod(shared_feat, gate_out['prob'])
        else:
            modulated_feat = shared_feat

        # Fine Classification
        fine_out = self.fine_head(modulated_feat)

        result = {
            'gate': gate_out,
            'fine': fine_out,
        }

        if return_features:
            result['features'] = shared_feat

        return result

    def compute_losses(
        self,
        outputs: dict,
        gate_targets: torch.Tensor,
        fine_targets: torch.Tensor,
        loss_weights: dict = None
    ) -> dict:
        """
        Compute combined classification losses.

        Args:
            outputs: forward pass outputs
            gate_targets: [B] gate ground truth (0=Nuisance, 1=Defect)
            fine_targets: [B] fine ground truth class indices
            loss_weights: dict with weights for each loss component
        Returns:
            dict with individual losses and total loss
        """
        if loss_weights is None:
            loss_weights = {'gate': 1.0, 'fine': 0.5, 'metric': 0.1}

        losses = {}

        # Gate Loss
        gate_loss = self.gate_head.compute_loss(
            outputs['gate']['logits'],
            gate_targets
        )
        losses['gate'] = gate_loss

        # Fine Loss (only on defect samples)
        is_defect_mask = gate_targets == 1
        fine_loss = self.fine_head.compute_loss(
            outputs['fine']['logits'],
            fine_targets,
            mask=is_defect_mask
        )
        losses['fine'] = fine_loss

        # Total Loss
        total_loss = (
            loss_weights['gate'] * gate_loss +
            loss_weights['fine'] * fine_loss
        )
        losses['total'] = total_loss

        return losses

    def freeze_gate(self):
        """Freeze Gate head for fine-tuning Fine head only."""
        for param in self.gate_head.parameters():
            param.requires_grad = False

    def unfreeze_gate(self):
        """Unfreeze Gate head."""
        for param in self.gate_head.parameters():
            param.requires_grad = True

    def freeze_fine(self):
        """Freeze Fine head."""
        for param in self.fine_head.parameters():
            param.requires_grad = False

    def unfreeze_fine(self):
        """Unfreeze Fine head."""
        for param in self.fine_head.parameters():
            param.requires_grad = True


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
