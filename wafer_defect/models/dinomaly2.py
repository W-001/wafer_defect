"""
Dinomaly2 Anomaly Detection Branch.

This module provides a clean interface for Dinomaly2-based anomaly detection.

Based on arXiv:2510.17611v2: "One Dinomaly2 Detect Them All"
Reference: https://github.com/cnulab/Dinomaly

Key components:
- Noisy Bottleneck: 3-layer MLP + Dropout (prevents over-generalization)
- Context-Aware Recentering: patch - CLS (resolves multi-class confusion)
- Loose Reconstruction: grouped layer reconstruction (prevents identity mapping)
- Linear Attention: unfocused attention (low-pass filter effect)

This module wraps the existing implementation in dinomaly_head.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from existing implementation
from .dinomaly_head import (
    Dinomaly2AnomalyHead,
    NoisyBottleneck2,
    LinearAttention2,
    ViTill2,
    VitBlock,  # Note: named VitBlock, not ViBlock
    bMlp,
    get_gaussian_kernel,
    WarmCosineScheduler,
    loose_reconstruction_loss,
)


class Dinomaly2Branch(nn.Module):
    """
    Dinomaly2 Anomaly Detection Branch.

    A clean wrapper around Dinomaly2AnomalyHead that integrates
    with the new WaferDefectModel architecture.

    Usage:
        dinomaly = Dinomaly2Branch(backbone=backbone, embed_dim=1024)
        outputs = dinomaly(images)  # Returns anomaly_score + heatmap
    """

    def __init__(
        self,
        backbone,
        embed_dim: int = 1024,
        layer_indices: list = None,
        num_heads: int = 16,
        num_decoder_blocks: int = 8,
        training_iters: int = 40000,
        lr: float = 2e-3,
        dropout: float = 0.2,
        grad_factor: float = 0.1,
        tau_warmup_iters: int = 1000,
        img_size: int = 392,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self._trained = False

        # Layer indices for ViT-L/16: [3,4,5,6,7,8,9,10] (8 middle layers)
        self.layer_indices = layer_indices or [3, 4, 5, 6, 7, 8, 9, 10]

        # Delegate to existing implementation
        self._anomaly_head = Dinomaly2AnomalyHead(
            backbone=backbone,
            layer_indices=self.layer_indices,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_blocks=num_decoder_blocks,
            training_iters=training_iters,
            lr=lr,
            dropout=dropout,
            grad_factor=grad_factor,
            tau_warmup_iters=tau_warmup_iters,
            img_size=img_size,
        )

    def forward(self, images: torch.Tensor, return_heatmap: bool = True) -> dict:
        """
        Compute anomaly scores and optionally return heatmap.

        Args:
            images: [B, 3, H, W] input images
            return_heatmap: whether to return spatial anomaly map

        Returns:
            dict with:
                - anomaly_score: [B] image-level anomaly score
                - heatmap: [B, 1, H, W] spatial anomaly map (if return_heatmap)
                - is_unknown_defect: [B] binary prediction
        """
        return self._anomaly_head.forward(images, return_heatmap=return_heatmap)

    def train_decoder(
        self,
        defect_loader,
        device: str = 'cuda',
        save_path: str = None,
        log_interval: int = 500,
    ):
        """
        Train the Dinomaly2 decoder on defect samples.

        Args:
            defect_loader: DataLoader with defect samples
            device: device
            save_path: optional path to save trained decoder
            log_interval: print frequency
        """
        self._anomaly_head.train_decoder(
            defect_loader=defect_loader,
            device=device,
            save_path=save_path,
            log_interval=log_interval,
        )
        self._trained = True

    def is_trained(self) -> bool:
        """Check if decoder has been trained."""
        return self._anomaly_head.is_trained()

    def save(self, path: str):
        """Save trained decoder."""
        self._anomaly_head.save(path)

    def load(self, path: str, device: str = 'cuda'):
        """Load trained decoder."""
        self._anomaly_head.load(path, device=device)
        self._trained = self._anomaly_head.is_trained()

    def calibrate_threshold(
        self,
        dataloader,
        device: str = 'cuda',
        percentile: float = 95,
    ) -> float:
        """
        Calibrate anomaly threshold from known defect samples.

        Args:
            dataloader: DataLoader with known defect samples
            device: device
            percentile: percentile for threshold (default 95)

        Returns:
            threshold: calibrated threshold value
        """
        self.eval()
        all_scores = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(device)
                if images.dim() == 5 and images.shape[1] >= 3:
                    images = images[:, 1, :, :, :]  # Center view
                elif images.dim() == 5 and images.shape[1] == 1:
                    images = images[:, 0, :, :, :]

                scores = self.forward(images)["anomaly_score"]
                all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=0).float()
        threshold = torch.quantile(all_scores, percentile / 100).item()

        print(f"[Dinomaly2Branch] Calibration:")
        print(f"  Score: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")
        print(f"  Threshold (p{percentile}): {threshold:.4f}")

        return threshold


class OpenSetDetector(nn.Module):
    """
    Open-set defect detection combining multiple signals.

    Combines:
    1. Dinomaly2 reconstruction error
    2. Distance to class centers
    3. Energy score
    """

    def __init__(
        self,
        threshold_dinomaly: float = 0.5,
        threshold_center: float = 2.0,
        dinomaly_weight: float = 0.6,
        center_weight: float = 0.4,
    ):
        super().__init__()
        self.threshold_dinomaly = threshold_dinomaly
        self.threshold_center = threshold_center
        self.dinomaly_weight = dinomaly_weight
        self.center_weight = center_weight

        # Class centers for known defect types
        self.register_buffer('class_centers', torch.zeros(1, 1))  # placeholder
        self.num_classes = 0

    def set_class_centers(self, centers: torch.Tensor):
        """
        Set class centers for distance-based anomaly detection.

        Args:
            centers: [num_classes, embed_dim] class center embeddings
        """
        self.class_centers = centers
        self.num_classes = centers.shape[0]

    def compute_center_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to nearest class center.

        Args:
            features: [B, embed_dim] feature embeddings

        Returns:
            distances: [B] distance to nearest center
        """
        if self.num_classes == 0:
            return torch.zeros(features.shape[0], device=features.device)

        # Normalize features and centers
        features = F.normalize(features, dim=-1)
        centers = F.normalize(self.class_centers, dim=-1)

        # Compute cosine similarity
        sim = torch.matmul(features, centers.t())  # [B, num_classes]
        dist = 1.0 - sim.max(dim=-1)[0]  # [B] distance to nearest

        return dist

    def forward(
        self,
        anomaly_score: torch.Tensor,
        features: torch.Tensor = None,
    ) -> dict:
        """
        Determine if sample is unknown defect.

        Args:
            anomaly_score: [B] Dinomaly2 anomaly scores
            features: [B, embed_dim] optional features for center distance

        Returns:
            dict with:
                - is_unknown: [B] binary prediction
                - combined_score: [B] weighted combination
                - anomaly_score: original anomaly score
                - center_distance: distance to nearest center (if features provided)
        """
        result = {
            'anomaly_score': anomaly_score,
            'is_unknown': None,
            'combined_score': anomaly_score,
        }

        # Center distance
        if features is not None and self.num_classes > 0:
            center_dist = self.compute_center_distance(features)
            result['center_distance'] = center_dist

            # Combined score
            combined = (
                self.dinomaly_weight * anomaly_score +
                self.center_weight * center_dist
            )
            result['combined_score'] = combined
        else:
            combined = anomaly_score

        # Threshold
        threshold = self.threshold_dinomaly
        result['is_unknown'] = (combined > threshold).long()

        return result


class Dinomaly2Loss(nn.Module):
    """
    Dinomaly2 reconstruction loss wrapper.

    Implements Loose Reconstruction Loss from the paper.
    """

    def __init__(
        self,
        p: float = 0.9,
        grad_factor: float = 0.1,
        tau_percent: float = None,
    ):
        super().__init__()
        self.p = p
        self.grad_factor = grad_factor
        self.tau_percent = tau_percent

    def forward(self, en_groups, de_groups) -> torch.Tensor:
        """
        Compute Dinomaly2 loose reconstruction loss.

        Args:
            en_groups: list of 2 [B, D] summed encoder group features
            de_groups: list of 2 [B, D] summed decoder group features

        Returns:
            loss: scalar loss value
        """
        return loose_reconstruction_loss(
            en_groups, de_groups,
            p=self.p,
            factor=self.grad_factor,
            tau_percent=self.tau_percent,
        )
