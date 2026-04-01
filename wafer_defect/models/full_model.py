"""
Full Wafer Defect Classification Model.
"""

import torch
import torch.nn as nn
import numpy as np

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

    Unknown defect detection supports two modes:
      - AnomalyHead: distance-to-class-center based (default)
      - RADAnomalyHead: multi-layer patch-KNN based (RAD method)

    When a sample is classified as Defect by Gate but is far from all known
    defect class centers (high anomaly score), it's marked as "unknown defect".
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
        use_uncertainty: bool = True,
        # RAD-specific options
        use_rad_anomaly: bool = False,
        rad_layer_indices: list = None,
        rad_k_image: int = 5,
        rad_bank_path: str = None,
    ):
        super().__init__()

        self.num_defect_classes = num_defect_classes
        self.use_anomaly_head = use_anomaly_head
        self.use_uncertainty = use_uncertainty
        self.use_rad_anomaly = use_rad_anomaly
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

        # Fine head (defect classification)
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
            if use_rad_anomaly:
                from .rad_head import RADAnomalyHead
                self.anomaly = RADAnomalyHead(
                    backbone=self.backbone,
                    layer_indices=rad_layer_indices or [3, 6, 9, 11],
                    k_image=rad_k_image,
                    bank_path=rad_bank_path,
                )
            else:
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
                - is_defect_pred: [B] (0=Nuisance, 1=Defect)
                - fine_logits: [B, num_defect_classes]
                - fine_prob: [B, num_defect_classes]
                - fine_pred: [B] (predicted defect type)
                - is_unknown_defect: [B] (1=unknown/ novel defect, 0=known defect)
                - anomaly_score: [B]
                - feat: [B, D] fused features (if return_features)
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

        # Anomaly detection
        anomaly_score = None
        is_unknown_defect = None
        if self.use_anomaly_head:
            if self.use_rad_anomaly:
                # RAD: only run if bank has been built
                if not getattr(self.anomaly, '_banks_built', False):
                    anomaly_score = None
                    is_unknown_defect = None
                else:
                    anomaly_out = self.anomaly(images)
                    anomaly_score = anomaly_out["anomaly_score"]
                    is_defect_mask = gate_out["is_defect_pred"] == 1
                    is_unknown_defect = torch.zeros(B, dtype=torch.long, device=images.device)
                    if is_defect_mask.sum() > 0:
                        is_unknown_defect[is_defect_mask] = anomaly_out["is_unknown_defect"][is_defect_mask]
            else:
                # Original class-center based
                anomaly_out = self.anomaly(fused_feat)
                anomaly_score = anomaly_out["anomaly_score"]
                is_defect_mask = gate_out["is_defect_pred"] == 1
                is_unknown_defect = torch.zeros(B, dtype=torch.long, device=images.device)
                dist_std = self.anomaly.dist_std.item()
                if dist_std > 1e-6:
                    is_unknown_defect[is_defect_mask] = (
                        anomaly_score[is_defect_mask] > 2.0
                    ).long()

        result = {
            "gate_logits": gate_out["logits"],
            "gate_prob": gate_out["prob"],
            "is_defect_pred": gate_out["is_defect_pred"],
            "fine_logits": fine_out["logits"],
            "fine_prob": fine_out["prob"],
            "fine_pred": fine_out["pred"],
            "is_unknown_defect": is_unknown_defect,
        }

        if return_features:
            result["feat"] = fused_feat

        if uncertainty is not None:
            result["uncertainty"] = uncertainty

        if anomaly_score is not None:
            result["anomaly_score"] = anomaly_score

        return result

    def update_anomaly_centers(self, feats: torch.Tensor, labels: torch.Tensor):
        """Update class centers for anomaly detection (non-RAD mode)."""
        if self.use_anomaly_head and not self.use_rad_anomaly:
            self.anomaly.update_centers(feats, labels)

    def build_rad_bank(
        self,
        dataloader,
        device: str = "cuda",
        save_path: str = None
    ):
        """
        Build RAD memory bank from training defect samples.

        Args:
            dataloader: DataLoader with training samples
            device: device
            save_path: optional path to save bank .pth
        """
        if not self.use_rad_anomaly:
            raise RuntimeError("RAD anomaly not enabled. Set use_rad_anomaly=True.")
        self.anomaly.build_bank(dataloader, device=device, save_path=save_path)

    def calibrate_anomaly_threshold(
        self,
        train_loader,
        device: str = "cuda",
        percentile: float = 95
    ):
        """
        Calibrate anomaly detection threshold from full training set.

        Args:
            train_loader: DataLoader with training samples
            device: Device to run on
            percentile: Percentile for threshold calibration (default 95)
        """
        if not self.use_anomaly_head:
            return

        self.eval()
        if self.use_rad_anomaly:
            print(f"[RAD Anomaly] Calibrating threshold (percentile={percentile})...")
            return self.anomaly.calibrate(train_loader, device=device, percentile=percentile)

        # Original class-center based
        all_feats = []
        defect_mask_list = []

        with torch.no_grad():
            for batch in train_loader:
                images = batch["images"].to(device)
                is_defect = batch["is_defect"].to(device)
                outputs = self.forward(images, return_features=True)
                all_feats.append(outputs["feat"].cpu())
                defect_mask_list.append((is_defect == 1).cpu())

        all_feats = torch.cat(all_feats, dim=0)
        defect_mask = torch.cat(defect_mask_list, dim=0)

        if defect_mask.sum() > 0:
            self.anomaly.update_statistics(all_feats, defect_mask)
            outputs = self.anomaly(all_feats[defect_mask])
            dists = outputs["anomaly_score"].cpu().numpy()
            threshold = np.percentile(dists, percentile)
            print(f"[Anomaly Detection] Final calibration:")
            print(f"  Known defect anomaly scores: mean={dists.mean():.4f}, std={dists.std():.4f}")
            print(f"  Threshold (percentile={percentile}): {threshold:.4f}")
            print(f"  → Samples with score > {threshold:.4f} will be flagged as unknown defects")
            return threshold


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
