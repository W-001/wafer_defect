"""
Multi-view fusion module for 3-view wafer defect samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewFusion(nn.Module):
    """
    Fuse features from 3 views of the same defect sample.

    Supports:
    - Mean pooling
    - Attention-weighted pooling
    - Gating mechanism
    """

    def __init__(self, feat_dim: int, fusion_type: str = "attention"):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 4, 1)
            )
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(feat_dim * 3, feat_dim),
                nn.Sigmoid()
            )
            self.transform = nn.Sequential(
                nn.Linear(feat_dim * 3, feat_dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, 3, D] features from 3 views
        Returns:
            fused: [B, D] fused features
        """
        if self.fusion_type == "mean":
            return feats.mean(dim=1)

        elif self.fusion_type == "attention":
            # Compute attention weights
            weights = self.attention(feats)  # [B, 3, 1]
            weights = F.softmax(weights, dim=1)
            return (feats * weights).sum(dim=1)  # [B, D]

        elif self.fusion_type == "gated":
            # Flatten and apply gate
            flat = feats.view(feats.size(0), -1)  # [B, 3*D]
            gate = self.gate(flat)  # [B, D]
            transformed = self.transform(flat)  # [B, D]
            return gate * transformed

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class ViewLevelAttention(nn.Module):
    """Learn to weight each view based on feature quality."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(feat_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, 3, D]
        Returns:
            weights: [B, 3, 1] softmax weights
        """
        scores = self.attention_weights(feats)  # [B, 3, 1]
        return F.softmax(scores, dim=1)
