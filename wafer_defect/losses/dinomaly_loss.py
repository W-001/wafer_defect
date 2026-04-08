"""
Dinomaly Loss: Loose Reconstruction Loss for Dinomaly2 anomaly detection.

Based on the Dinomaly2 paper (arXiv 2510.17611):
  "One Dinomaly2 Detect Them All: A Unified Framework for
   Full-Spectrum Unsupervised Anomaly Detection"

Key features:
  - Loose Reconstruction: Group encoder/decoder features by semantic layers
  - Gradient scaling: Easy patches get reduced gradient (factor=0.1)
  - tau warmup: Ramp tau from 0% to 90% over first N iterations
"""

import torch
import torch.nn as nn


def _loose_grad_scale(x, mask, factor=0.1):
    """
    Dinomaly2: Gradient scaling for easy patches.

    Easy patches (mask=True): scale gradient to `factor` (0.1 -> 10% of original).
    Hard patches (mask=False): keep full gradient.

    Unlike v1's complete gradient stop, this uses SOFT scaling to avoid instability.

    Args:
        x: Input tensor to scale
        mask: Boolean mask where True indicates easy patches
        factor: Scaling factor for easy patches (default 0.1)

    Returns:
        Scaled tensor with gradient modulation
    """
    return x * (1.0 - factor * mask.float())


def loose_reconstruction_loss(
    en_groups: list,
    de_groups: list,
    p: float = 0.9,
    factor: float = 0.1,
    tau_percent: float = None
) -> torch.Tensor:
    """
    Dinomaly2 Loose Reconstruction Loss.

    From the paper:
      L_loose = (1/|G|) * sum_{k in G} d_cos(F(g_k), F(^g'_k))
    where ^g'_k undergoes selective gradient modulation:
      ^g'_k(n) = sg_{0.1}(^g_k(n))  if d_cos(g_k(n), ^g_k(n)) > tau_k
                 ^g_k(n)            otherwise
    where tau_k = 90th percentile within the batch (default).

    Args:
        en_groups: List of 2 [B, D] summed encoder group features
        de_groups: List of 2 [B, D] summed decoder group features
        p: Top percentile threshold for hard patches (default 0.9 -> 90th pct)
        factor: Gradient scaling for easy patches (default 0.1)
        tau_percent: Current tau warmup value, or None for final (0.9)

    Returns:
        Scalar loss tensor (mean cosine distance across groups)
    """
    cos_loss = nn.CosineSimilarity()
    loss = 0.0

    # tau_k = k-th percentile (90% by default)
    tau = tau_percent if tau_percent is not None else p

    for g_idx, (g_enc, g_dec) in enumerate(zip(en_groups, de_groups)):
        # g_enc, g_dec: [B, D]
        cos_sim = cos_loss(g_enc, g_dec)  # [B]
        point_dist = 1.0 - cos_sim  # [B], scalar per sample

        # tau percentile threshold within this batch
        if tau > 0:
            thresh = torch.quantile(point_dist, q=tau)
        else:
            thresh = point_dist.max()

        # Easy patches: d < thresh -> scale gradient to factor
        mask = (point_dist < thresh).float()  # [B]

        # Forward loss: mean cosine distance across batch
        loss += torch.mean(point_dist)

        # Gradient modulation: easy patches scale to `factor`
        # Use hook on g_dec with scaling
        def grad_hook(grad, m=mask, fac=factor):
            return grad * (1.0 - fac * m.unsqueeze(-1))

        g_dec.register_hook(grad_hook)

    loss = loss / len(en_groups)
    return loss


class DinomalyLoss(nn.Module):
    """
    Dinomaly2 Loose Reconstruction Loss wrapper.

    This module provides a torch.nn.Module interface for the Loose Reconstruction
    loss, making it compatible with standard training loops.

    Args:
        p: Top percentile threshold for hard patches (default 0.9)
        factor: Gradient scaling for easy patches (default 0.1)
    """

    def __init__(self, p: float = 0.9, factor: float = 0.1):
        super().__init__()
        self.p = p
        self.factor = factor

    def forward(
        self,
        en_groups: list,
        de_groups: list,
        tau_percent: float = None
    ) -> torch.Tensor:
        """
        Compute Loose Reconstruction loss.

        Args:
            en_groups: List of 2 [B, D] summed encoder group features
            de_groups: List of 2 [B, D] summed decoder group features
            tau_percent: Current tau warmup value (optional)

        Returns:
            Scalar loss tensor
        """
        return loose_reconstruction_loss(
            en_groups=en_groups,
            de_groups=de_groups,
            p=self.p,
            factor=self.factor,
            tau_percent=tau_percent
        )

    def extra_repr(self) -> str:
        return f"p={self.p}, factor={self.factor}"
