"""
Multi-view fusion module — supports single-view passthrough and 3-view fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewFusion(nn.Module):
    """
    Fuse features from 3 views of the same defect sample.

    When use_three_views=False, acts as a pass-through (single-view mode).
    When use_three_views=True, supports:
      - Mean pooling
      - Attention-weighted pooling
      - Gating mechanism
    """

    def __init__(self, feat_dim: int, fusion_type: str = "attention",
                 use_three_views: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type
        self.use_three_views = use_three_views

        if use_three_views and fusion_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 4, 1)
            )
        elif use_three_views and fusion_type == "gated":
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
            feats: [B, 3, D] features from 3 views (3-view mode)
                  or [B, D] features (single-view mode, use_three_views=False)
        Returns:
            fused: [B, D] fused features
        """
        # Single-view mode: pass through directly
        if not self.use_three_views:
            # Accept both [B, D] and [B, 1, D] shapes
            if feats.dim() == 3:
                return feats.squeeze(1)
            return feats

        # ── 3-view fusion ────────────────────────────────────────────────────
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
