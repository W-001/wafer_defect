"""
DefectModel: Complete Wafer Defect Detection Model.

This module implements the full wafer defect detection pipeline:
- DINOv3 backbone (frozen)
- Classification branch (Gate + Fine)
- Dinomaly2 anomaly detection branch
- Open-set detection (unknown defect identification)

Architecture:
    Input Image → Backbone → Classification → [Nuisance] → output
                                ↓
                           [Defect] → Anomaly Detection → [heatmap, score]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone
from .classification import ClassificationBranch, GateHead, FineHead, PrototypeClassifier
from .dinomaly2 import Dinomaly2Branch, OpenSetDetector


class WaferDefectModel(nn.Module):
    """
    Complete wafer defect detection model.

    Combines:
    - DINOv3 backbone for feature extraction
    - ClassificationBranch for gate and fine classification
    - Dinomaly2Branch for anomaly detection
    - OpenSetDetector for unknown defect identification
    """

    def __init__(
        self,
        num_defect_classes: int,
        backbone_name: str = "dinov3_vitl16",
        pretrained_path: str = None,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        defect_weight: float = 3.0,
        use_gate_modulation: bool = True,
        use_dinomaly2: bool = True,
        dinomaly2_config: dict = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.num_defect_classes = num_defect_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_dinomaly2 = use_dinomaly2

        # 1. Backbone (frozen by default)
        self.backbone = DINOv3Backbone(
            model_name=backbone_name,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
        )

        # 2. Classification Branch
        self.classification = ClassificationBranch(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_defect_classes=num_defect_classes,
            defect_weight=defect_weight,
            use_gate_modulation=use_gate_modulation,
        )

        # 3. Dinomaly2 Anomaly Detection
        if use_dinomaly2:
            if dinomaly2_config is None:
                dinomaly2_config = {}
            self.dinomaly2 = Dinomaly2Branch(
                backbone=self.backbone,
                embed_dim=embed_dim,
                img_size=dinomaly2_config.get('img_size', 392),
                layer_indices=dinomaly2_config.get('layer_indices', [3, 4, 5, 6, 7, 8, 9, 10]),
                num_heads=dinomaly2_config.get('num_heads', 16),
                num_decoder_blocks=dinomaly2_config.get('num_decoder_blocks', 8),
                training_iters=dinomaly2_config.get('training_iters', 40000),
                lr=dinomaly2_config.get('lr', 2e-3),
                dropout=dinomaly2_config.get('dropout', 0.2),
                grad_factor=dinomaly2_config.get('grad_factor', 0.1),
                tau_warmup_iters=dinomaly2_config.get('tau_warmup_iters', 1000),
            )
        else:
            self.dinomaly2 = None

        # 4. Open-set Detection
        self.open_set_detector = OpenSetDetector()

        # 5. Class Centers for anomaly detection
        self.class_centers = nn.Parameter(
            torch.zeros(num_defect_classes, hidden_dim),
            requires_grad=False
        )

    def forward(
        self,
        images: torch.Tensor,
        mode: str = 'all',
        return_features: bool = False,
        return_heatmap: bool = True,
    ) -> dict:
        """
        Complete forward pass.

        Args:
            images: [B, 3, H, W] input images (RGB)
            mode: 'all' | 'classification' | 'anomaly'
            return_features: if True, return intermediate features
            return_heatmap: if True, return anomaly heatmap

        Returns:
            dict with:
                - is_defect: [B] 0=Nuisance, 1=Defect
                - defect_type: [B] predicted defect type index
                - gate_prob: [B, 2] gate probabilities
                - fine_prob: [B, num_classes] fine probabilities
                - anomaly_score: [B] anomaly score (if mode includes 'anomaly')
                - heatmap: [B, 1, H, W] anomaly heatmap (if return_heatmap)
                - is_unknown: [B] 1=unknown defect, 0=known defect
                - features: [B, hidden_dim] shared features (if return_features)
        """
        B = images.shape[0]

        # 1. Backbone feature extraction
        feat = self.backbone(images)  # [B, D]

        # 2. Classification
        cls_outputs = self.classification(feat, return_features=return_features)
        gate_out = cls_outputs['gate']
        fine_out = cls_outputs['fine']

        is_defect = gate_out['pred']  # [B]
        defect_type = fine_out['pred']  # [B]

        result = {
            'is_defect': is_defect,
            'defect_type': defect_type,
            'gate_prob': gate_out['prob'],
            'fine_prob': fine_out['prob'],
        }

        if return_features and 'features' in cls_outputs:
            result['features'] = cls_outputs['features']

        # 3. Anomaly Detection (only for defect samples)
        if mode in ['all', 'anomaly'] and self.use_dinomaly2 and self.dinomaly2 is not None:
            if self.dinomaly2.is_trained():
                anomaly_out = self.dinomaly2(images, return_heatmap=return_heatmap)
                anomaly_score = anomaly_out['anomaly_score']  # [B]

                # Open-set detection
                features = cls_outputs.get('features', feat)
                open_set_out = self.open_set_detector(
                    anomaly_score=anomaly_score,
                    features=features if is_defect.any() else None,
                )

                result['anomaly_score'] = anomaly_score
                result['is_unknown'] = open_set_out['is_unknown']

                if return_heatmap:
                    result['heatmap'] = anomaly_out.get('heatmap')

        # If Dinomaly not trained, use class center distance
        elif mode in ['all', 'anomaly'] and is_defect.any():
            features = cls_outputs.get('features', feat)
            center_dist = self.open_set_detector.compute_center_distance(features)
            result['anomaly_score'] = center_dist
            result['is_unknown'] = (center_dist > 2.0).long()

        return result

    def update_class_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update class centers for anomaly detection.

        Args:
            features: [B, hidden_dim] feature embeddings
            labels: [B] class labels
        """
        for c in range(self.num_defect_classes):
            mask = labels == c
            if mask.sum() > 0:
                center = features[mask].mean(dim=0)
                self.class_centers[c] = center

        # Update open-set detector
        self.open_set_detector.set_class_centers(self.class_centers)

    def train_dinomaly2(
        self,
        defect_loader,
        device: str = 'cuda',
        save_path: str = None,
        log_interval: int = 500,
    ):
        """
        Train Dinomaly2 decoder on defect samples.

        Args:
            defect_loader: DataLoader with defect samples
            device: device
            save_path: optional path to save trained decoder
            log_interval: print frequency
        """
        if not self.use_dinomaly2 or self.dinomaly2 is None:
            raise RuntimeError("Dinomaly2 is not enabled")

        self.dinomaly2.train_decoder(
            defect_loader=defect_loader,
            device=device,
            save_path=save_path,
            log_interval=log_interval,
        )

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: path to save checkpoint
        """
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        checkpoint = {
            'classification': self.classification.state_dict(),
            'class_centers': self.class_centers.data,
        }
        torch.save(checkpoint, path)

        # Save Dinomaly2 separately if trained
        if self.use_dinomaly2 and self.dinomaly2 is not None and self.dinomaly2.is_trained():
            dinomaly_path = path.replace('.pt', '_dinomaly2.pt')
            self.dinomaly2.save(dinomaly_path)

    def load(self, path: str, device: str = 'cuda', load_dinomaly: bool = True):
        """
        Load model checkpoint.

        Args:
            path: path to checkpoint
            device: device
            load_dinomaly: whether to load Dinomaly2 decoder
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.classification.load_state_dict(checkpoint['classification'])
        self.class_centers.data = checkpoint['class_centers'].to(self.class_centers.device)

        # Update open-set detector
        self.open_set_detector.set_class_centers(self.class_centers)

        # Load Dinomaly2
        if load_dinomaly:
            dinomaly_path = path.replace('.pt', '_dinomaly2.pt')
            if os.path.exists(dinomaly_path):
                self.dinomaly2.load(dinomaly_path, device=device)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_classification(self):
        """Freeze classification branch."""
        for param in self.classification.parameters():
            param.requires_grad = False

    def unfreeze_classification(self):
        """Unfreeze classification branch."""
        for param in self.classification.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append((name, param))
        return trainable


class WaferDefectModelSimple(nn.Module):
    """
    Simplified model without DINOv3 for quick testing.

    Uses a simple CNN backbone for fast iteration.
    """

    def __init__(
        self,
        num_defect_classes: int,
        img_size: int = 392,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.num_defect_classes = num_defect_classes
        self.img_size = img_size  # Store for heatmap generation
        self.use_dinomaly2 = False

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.embed_dim = feat_dim

        # Classification
        self.gate_head = GateHead(input_dim=feat_dim, defect_weight=3.0)
        self.fine_head = FineHead(input_dim=feat_dim, num_classes=num_defect_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False, return_heatmap: bool = True) -> dict:
        """
        Forward pass.

        Args:
            images: [B, 3, H, W] input images
            return_features: whether to return intermediate features
            return_heatmap: whether to return anomaly heatmap (returns None for simple model)

        Returns:
            dict with classification outputs and optionally features/heatmap
        """
        feat = self.backbone(images)
        gate_out = self.gate_head(feat)
        fine_out = self.fine_head(feat)

        result = {
            'gate_logits': gate_out['logits'],
            'fine_logits': fine_out['logits'],
            'is_defect': gate_out['pred'],
            'defect_type': fine_out['pred'],
            'gate_prob': gate_out['prob'],
            'fine_prob': fine_out['prob'],
        }

        if return_features:
            result['feat'] = feat

        if return_heatmap:
            # Simple model doesn't produce heatmaps, return zeros
            B = images.shape[0]
            result['heatmap'] = torch.zeros(B, 1, self.img_size, self.img_size, device=images.device)
            result['anomaly_score'] = torch.zeros(B, device=images.device)

        return result
