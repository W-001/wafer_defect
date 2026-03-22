"""
Full Wafer Defect Classification Model.
"""

import torch
import torch.nn as nn

from .backbone import DINOv3Backbone
from .fusion import MultiViewFusion
from .gate_head import GateHead, UncertaintyHead
from .fine_head import FineHead, PrototypeClassifier
from .anomaly_head import AnomalyHead


class WaferDefectModel(nn.Module):
    """
    Complete wafer defect classification model with:
    - DINOv3 backbone
    - 3-view fusion
    - Gate head (Nuisance vs Defect)
    - Fine head (defect type classification)
    - Anomaly head (unknown defect detection)
    """

    def __init__(
        self,
        num_defect_classes: int,
        backbone_name: str = "dinov3_vitb16",
        pretrained_path: str = None,
        embed_dim: int = 768,
        gate_hidden_dim: int = 512,
        fine_hidden_dim: int = 512,
        defect_weight: float = 3.0,
        fusion_type: str = "attention",
        freeze_backbone: bool = False,
        use_anomaly_head: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()

        self.num_defect_classes = num_defect_classes
        self.use_anomaly_head = use_anomaly_head
        self.use_uncertainty = use_uncertainty

        # Backbone
        self.backbone = DINOv3Backbone(
            model_name=backbone_name,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim
        )

        # Multi-view fusion
        self.fusion = MultiViewFusion(
            feat_dim=embed_dim,
            fusion_type=fusion_type
        )

        # Gate head (Nuisance vs Defect)
        self.gate = GateHead(
            feat_dim=embed_dim,
            hidden_dim=gate_hidden_dim,
            defect_weight=defect_weight
        )

        # Fine head (defect type classification)
        self.fine = FineHead(
            feat_dim=embed_dim,
            num_classes=num_defect_classes,
            hidden_dim=fine_hidden_dim
        )

        # Uncertainty head
        if use_uncertainty:
            self.uncertainty = UncertaintyHead(embed_dim)

        # Anomaly head
        if use_anomaly_head:
            self.anomaly = AnomalyHead(
                feat_dim=embed_dim,
                num_classes=num_defect_classes
            )

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            images: [B, 3, C, H, W] 3-view images
            return_features: if True, also return intermediate features

        Returns:
            dict with:
                - gate_logits: [B, 2]
                - gate_prob: [B, 2]
                - is_defect_pred: [B]
                - fine_logits: [B, num_defect_classes]
                - fine_pred: [B]
                - feat: [B, D] fused features (if return_features)
                - anomaly_score: [B] (if use_anomaly_head)
                - uncertainty: [B] (if use_uncertainty)
        """
        B, V, C, H, W = images.shape

        # Reshape for backbone: [B*3, C, H, W]
        images_flat = images.view(B * V, C, H, W)

        # Extract features for each view
        view_feats = self.backbone(images_flat)  # [B*3, D]

        # Reshape back: [B, 3, D]
        view_feats = view_feats.view(B, V, -1)

        # Fuse 3 views
        fused_feat = self.fusion(view_feats)  # [B, D]

        # Gate head (Nuisance vs Defect)
        gate_out = self.gate(fused_feat)

        # Fine head (defect classification)
        fine_out = self.fine(fused_feat)

        # Uncertainty
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty(fused_feat)

        # Anomaly score
        anomaly_score = None
        if self.use_anomaly_head:
            anomaly_out = self.anomaly(fused_feat)
            anomaly_score = anomaly_out["anomaly_score"]

        result = {
            "gate_logits": gate_out["logits"],
            "gate_prob": gate_out["prob"],
            "is_defect_pred": gate_out["is_defect_pred"],
            "fine_logits": fine_out["logits"],
            "fine_prob": fine_out["prob"],
            "fine_pred": fine_out["pred"],
        }

        if return_features:
            result["feat"] = fused_feat

        if uncertainty is not None:
            result["uncertainty"] = uncertainty

        if anomaly_score is not None:
            result["anomaly_score"] = anomaly_score

        return result

    def update_anomaly_centers(self, feats: torch.Tensor, labels: torch.Tensor):
        """Update class centers for anomaly detection."""
        if self.use_anomaly_head:
            self.anomaly.update_centers(feats, labels)


class WaferDefectModelSimple(nn.Module):
    """
    Simplified version without DINOv3 for quick testing.
    Uses a simple CNN backbone.
    """

    def __init__(
        self,
        num_defect_classes: int,
        img_size: int = 224,
        feat_dim: int = 512
    ):
        super().__init__()
        self.num_defect_classes = num_defect_classes

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.ReLU(inplace=True)
        )

        self.fusion = MultiViewFusion(feat_dim, fusion_type="mean")
        self.gate = GateHead(feat_dim)
        self.fine = FineHead(feat_dim, num_defect_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False) -> dict:
        B, V, C, H, W = images.shape

        images_flat = images.view(B * V, C, H, W)
        view_feats = self.backbone(images_flat)
        view_feats = view_feats.view(B, V, -1)
        fused_feat = self.fusion(view_feats)

        gate_out = self.gate(fused_feat)
        fine_out = self.fine(fused_feat)

        result = {
            "gate_logits": gate_out["logits"],
            "gate_prob": gate_out["prob"],
            "is_defect_pred": gate_out["is_defect_pred"],
            "fine_logits": fine_out["logits"],
            "fine_prob": fine_out["prob"],
            "fine_pred": fine_out["pred"],
        }

        if return_features:
            result["feat"] = fused_feat

        return result

    def update_anomaly_centers(self, feats: torch.Tensor, labels: torch.Tensor):
        """Dummy method for compatibility with full model."""
        pass
