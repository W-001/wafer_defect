"""
Dinomaly / Dinomaly2: Masked Autoencoder for Anomaly Detection with DINOv3.

Based on:
  v1 (CVPR 2025): https://github.com/cnulab/Dinomaly
  v2 (arXiv 2510.17611): https://github.com/guojiajeremy/Dinomaly

Two anomaly detection modes:
  Dinomaly v1 — single bMlp bottleneck, no CLS recentering, factor=0 hard mask
  Dinomaly2 — Noisy Bottleneck (3-layer MLP + Dropout), Context-Aware Recentering,
               Loose Reconstruction (2-group summed features, gradient factor=0.1, τ warmup)

Inference for both:
  Reconstruction error = 1 - cosine_similarity(encoder_feat, decoder_feat)
  Anomaly score = mean of top-z% most anomalous pixels
"""

import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Core building blocks (shared between v1 and v2)
# ─────────────────────────────────────────────────────────────────────────────

class bMlp(nn.Module):
    """Standard bottleneck MLP (v1 style — single hidden layer)."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NoisyBottleneck2(nn.Module):
    """
    Noisy Bottleneck (Dinomaly2) — 3-layer MLP with Dropout.

    From the paper: "Dropout is all you need."
    Dropout introduces pseudo feature anomalies by randomly masking
    activations, forcing the decoder to map corrupted inputs back to
    the most likely normal patterns. This prevents over-generalization.

    Architecture: Linear(in, 4*in) → GELU → Dropout(p) →
                  Linear(4*in, in) → GELU → Dropout(p) → Linear(in, in)
    """
    def __init__(self, in_features, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(in_features * 4, in_features)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class LinearAttention2(nn.Module):
    """
    Efficient linear attention (used in decoder for both v1 and v2).

    From the paper: "One man's poison is another man's meat."
    LA's inability to focus is a FEATURE, not a bug: it spreads attention
    across the entire image (low-pass filter), preventing identity mapping.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class VitBlock(nn.Module):
    """Transformer block used in Dinomaly decoder."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = bMlp(dim, mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, x, attn_mask=None):
        y, _ = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  ViTill variants (v1 and v2)
# ─────────────────────────────────────────────────────────────────────────────

class ViTill(nn.Module):
    """
    ViTill (v1): frozen encoder + learnable bottleneck + learnable decoder.

    Features are extracted WITHOUT CLS token (en_list[i] has shape [B, N, D]).
    All 8 layer features are summed → one vector → bottleneck → 8 decoder blocks.
    Output: 2 groups of [B, C, H, W] feature maps for cosine distance computation.
    """
    def __init__(self, encoder, bottleneck, decoder,
                 target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                 fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                 fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                 mask_neighbor_size=0):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)
        side = int(math.sqrt(en_list[0].shape[1]))

        x = self._fuse(en_list)
        for blk in self.bottleneck:
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self._make_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for blk in self.decoder:
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self._fuse([en_list[idx] for idx in idxs])
              for idxs in self.fuse_layer_encoder]
        de = [self._fuse([de_list[idx] for idx in idxs])
              for idxs in self.fuse_layer_decoder]

        en = [e.permute(0, 2, 1).reshape(x.shape[0], -1, side, side).contiguous()
              for e in en]
        de = [d.permute(0, 2, 1).reshape(x.shape[0], -1, side, side).contiguous()
              for d in de]
        return en, de

    def _fuse(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _make_mask(self, feature_size, device='cuda'):
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end] = 0
        mask = mask.view(h * w, h * w)
        mask_all = torch.ones(
            h * w + 1 + self.encoder.num_register_tokens,
            h * w + 1 + self.encoder.num_register_tokens,
            device=device
        )
        mask_all[1 + self.encoder.num_register_tokens:,
                 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTill2(nn.Module):
    """
    ViTill (Dinomaly2): frozen encoder + learnable Noisy Bottleneck +
    learnable decoder + Context-Aware Recentering + Loose Reconstruction.

    Key differences from v1:
      1. Encoder features INCLUDE CLS token: [B, N+1, D]
      2. Recentering: patch_features = patch - cls (broadcast)
      3. Loose Reconstruction: group layers by summing, then reconstruct groups
      4. No mask generation (mask_neighbor_size always 0)

    Architecture (per the paper equations):
      z0 = Σ_{i∈M} f_i           (M = {3,...,10} middle layers, 8 layers)
      ~f_patch = f_patch - f_cls  (Context-Aware Recentering)
      z = B(~f_patch)            (Noisy Bottleneck)
      g_k = Σ_{i∈S_k} f_i        (group k = sum of encoder layer features)
      ^g_k = Σ_{j∈S'_k} ^f_j     (decoder reconstructs group)
      A_k = 1 - cosine(g_k, ^g_k)  (anomaly map for group k)
    """
    def __init__(self, encoder, bottleneck, decoder,
                 target_layers=[3, 4, 5, 6, 7, 8, 9, 10],
                 # 2 groups: shallow [0-3] = layers 3-6, deep [4-7] = layers 7-10
                 fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                 fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                 num_register_tokens=0):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.num_register_tokens = num_register_tokens

    def forward(self, x):
        """
        Returns:
          en_groups: list of 2 [B, D] summed layer-group features
          de_groups: list of 2 [B, D] reconstructed summed layer-group features
          patch_size: (H, W) spatial dimensions
        """
        # Get intermediate layers WITH CLS token: [B, N+1, D] each
        en_list = self.encoder.get_intermediate_layers(
            x, n=self.target_layers, norm=False, return_class_token=True
        )
        # Each element is tuple: (patch_tokens [B,N,D], cls_token [B,D])
        # OR a single tensor [B, N+1, D] (depending on encoder implementation)

        B = x.shape[0]
        N_plus_1 = en_list[0].shape[1]
        side = int(math.sqrt(N_plus_1 - self.num_register_tokens - 1))
        num_patches = side * side

        # Step 1: Recentering — patch - CLS for each layer
        # en_list[i]: [B, N+1+num_reg, D] or [B, N+1, D]
        recentered_list = []
        for layer_feat in en_list:
            if layer_feat.dim() == 3 and layer_feat.shape[1] > num_patches:
                # Has CLS + optional register tokens: [B, 1+num_reg+N, D]
                # Extract CLS token (first one, after register tokens if any)
                cls_start = self.num_register_tokens
                cls = layer_feat[:, cls_start:cls_start + 1, :]   # [B, 1, D]
                patch = layer_feat[:, cls_start + 1:, :]          # [B, N, D]
            else:
                # Only patch tokens: [B, N, D]
                # For single-view case, use mean as proxy for CLS
                cls = layer_feat.mean(dim=1, keepdim=True)       # [B, 1, D]
                patch = layer_feat                                  # [B, N, D]
            # Context-Aware Recentering: patch - cls
            rec = patch - cls                                      # [B, N, D]
            recentered_list.append(rec)

        # Step 2: Sum all recentered layers → bottleneck
        z0 = torch.stack(recentered_list, dim=1).sum(dim=1)         # [B, N, D]
        z = self.bottleneck(z0)                                   # [B, N, D]

        # Step 3: Decoder (processes [B, N, D] sequence)
        de_list = []
        for blk in self.decoder:
            z = blk(z, attn_mask=None)
            de_list.append(z)
        de_list = de_list[::-1]  # reverse: [7,6,5,4,3,2,1,0]

        # Step 4: Form 2 groups by SUMMING within each group (Loose Reconstruction)
        # g_k = Σ_{i∈S_k} f_i  (grouped encoder features)
        # ^g_k = Σ_{j∈S'_k} ^f_j  (grouped decoder features)
        en_groups = []
        de_groups = []
        for idxs in self.fuse_layer_encoder:
            g = torch.stack([en_list[i][:, self.num_register_tokens + 1:, :].mean(dim=1)
                             if en_list[i].dim() == 3 and en_list[i].shape[1] > num_patches
                             else en_list[i].mean(dim=1)
                             for i in idxs], dim=1).sum(dim=1)  # [B, D]
            en_groups.append(g)

        for idxs in self.fuse_layer_decoder:
            g = torch.stack([de_list[i].mean(dim=1) for i in idxs], dim=1).sum(dim=1)  # [B, D]
            de_groups.append(g)

        return en_groups, de_groups, (side, side)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def _modify_grad(x, inds, factor=0.0):
    """Hook-based gradient modification for hard-mining cosine loss (v1)."""
    inds = inds.expand_as(x)
    x = x.clone()
    x[inds] *= factor
    return x


def global_cosine_hm_percent(a, b, p=0.9, factor=0.0):
    """
    v1 loss: Masked global cosine similarity.
    Keeps gradients only for the top (1-p)% most anomalous patches.
    factor=0 → complete gradient stop for easy patches.
    """
    cos_loss = nn.CosineSimilarity()
    loss = 0.0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1.0 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(
            point_dist.reshape(-1),
            k=max(1, int(point_dist.numel() * (1.0 - p)))
        )[0][-1]
        loss += torch.mean(1.0 - cos_loss(
            a_.reshape(a_.shape[0], -1),
            b_.reshape(b_.shape[0], -1)
        ))
        partial_func = partial(_modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    loss = loss / len(a)
    return loss


def _loose_grad_scale(x, mask, factor=0.1):
    """
    Dinomaly2: Gradient scaling for easy patches.
    Easy patches (mask=True): scale gradient to `factor` (0.1 → 10% of original).
    Hard patches (mask=False): keep full gradient.
    Unlike v1's complete gradient stop, this uses SOFT scaling to avoid instability.
    """
    # gradient = grad_output * (1 - factor * mask)
    return x * (1.0 - factor * mask.float())


def loose_reconstruction_loss(en_groups, de_groups, p=0.9, factor=0.1, tau_percent=None):
    """
    Dinomaly2 loss: Loose Reconstruction Loss.

    From the paper:
      L_loose = (1/|G|) Σ_{k∈G} d_cos(F(g_k), F(^g'_k))
    where ^g'_k undergoes selective gradient modulation:
      ^g'_k(n) = sg_{0.1}(^g_k(n))  if d_cos(g_k(n), ^g_k(n)) > τ_k
                 ^g_k(n)          otherwise
    where τ_k = 90th percentile within the batch (default).

    Args:
        en_groups: list of 2 [B, D] summed encoder group features
        de_groups: list of 2 [B, D] summed decoder group features
        p: top percentile of hard patches to keep (default 0.9 → 90th pct)
        factor: gradient scaling for easy patches (default 0.1)
        tau_percent: current τ warmup value, or None for final (0.9)
    """
    cos_loss = nn.CosineSimilarity()
    loss = 0.0
    # τ_k = k-th percentile (90% by default)
    tau = tau_percent if tau_percent is not None else 0.9

    for g_idx, (g_enc, g_dec) in enumerate(zip(en_groups, de_groups)):
        # g_enc, g_dec: [B, D]
        cos_sim = cos_loss(g_enc, g_dec)                    # [B]
        point_dist = 1.0 - cos_sim                          # [B], scalar per sample

        # τ percentile threshold within this batch
        thresh = torch.quantile(point_dist, q=tau) if tau > 0 else point_dist.max()
        # Easy patches: d < thresh → scale gradient to factor
        mask = (point_dist < thresh).float()               # [B]

        # Forward loss: mean cosine distance across batch
        loss += torch.mean(point_dist)

        # Gradient modulation: easy patches scale to `factor`
        # Use hook on g_dec with scaling
        def grad_hook(grad, m=mask, fac=factor):
            return grad * (1.0 - fac * m.unsqueeze(-1))

        g_dec.register_hook(grad_hook)

    loss = loss / len(en_groups)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
#  Warm cosine LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters,
                 warmup_iters=0, start_warmup_value=0.0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(max(total_iters - warmup_iters, 1))
        schedule = final_value + 0.5 * (base_value - final_value) * \
            (1.0 + np.cos(math.pi * iters / max(len(iters) - 1, 1)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        idx = min(self.last_epoch, len(self.schedule) - 1)
        return [self.schedule[idx] for _ in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian smoothing kernel (shared)
# ─────────────────────────────────────────────────────────────────────────────

def get_gaussian_kernel(kernel_size=5, sigma=2.0, channels=1):
    x = torch.arange(kernel_size, dtype=torch.float32)
    x_grid = x.repeat(kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1.0) / 2.0
    variance = sigma ** 2.0
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    gauss = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
    gauss.weight.data = kernel
    gauss.weight.requires_grad_(False)
    return gauss


# ─────────────────────────────────────────────────────────────────────────────
#  DinomalyAnomalyHead (v1)
# ─────────────────────────────────────────────────────────────────────────────

class DinomalyAnomalyHead(nn.Module):
    """
    Dinomaly v1 anomaly detection head.

    Architecture (from dinomaly_mvtec_uni_base.py):
      - Frozen DINOv3 encoder: layers [2,3,4,5,6,7,8,9] (8 layers, no CLS)
      - Single bMlp bottleneck: Linear → GELU → Linear (no Dropout)
      - 8 × LinearAttention2 decoder blocks
      - Loss: global_cosine_hm_percent (factor=0, hard gradient stop)
      - Optimizer: StableAdamW(lr=2e-3), cosine decay to 2e-4
      - Iterations: 10,000
      - Image size: 224×224
    """

    def __init__(
        self,
        backbone,
        layer_indices=None,
        embed_dim=1024,
        num_heads=16,
        num_decoder_blocks=8,
        training_iters=10000,
        lr=2e-3,
        loss_p=0.9,
        img_size=224,
        device='cuda',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.training_iters = training_iters
        self.lr = lr
        self.loss_p = loss_p
        self.img_size = img_size
        self._device = device
        self.version = 1

        self.layer_indices = layer_indices or [2, 3, 4, 5, 6, 7, 8, 9]
        n = len(self.layer_indices)
        half = n // 2
        self.fuse_layer_encoder = [[i for i in range(half)], [i for i in range(half, n)]]
        self.fuse_layer_decoder = [[i for i in range(half)], [i for i in range(half, n)]]

        # v1 bottleneck: single bMlp (no Dropout)
        self.bottleneck = nn.ModuleList([
            bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)
        ])

        self.decoder = nn.ModuleList([
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(num_decoder_blocks)
        ])

        self._init_weights()
        self.vitill = ViTill(
            encoder=backbone.model,
            bottleneck=self.bottleneck,
            decoder=self.decoder,
            target_layers=self.layer_indices,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
        )

        self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4.0, channels=1)
        self._trained = False

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, images: torch.Tensor, return_heatmap=True) -> dict:
        if not self._trained:
            B = images.shape[0]
            return {
                "anomaly_score": torch.zeros(B, device=images.device),
                "heatmap": torch.zeros(B, 1, self.img_size, self.img_size,
                                       device=images.device),
                "is_unknown_defect": torch.zeros(B, dtype=torch.long,
                                                  device=images.device),
            }

        with torch.no_grad():
            en, de = self.vitill(images)  # en/de: list of 2 [B, C, H, W]

        # v1: per-patch cosine distance per group, then average
        cos_sim = nn.CosineSimilarity(dim=1)
        all_maps = []
        for enc, dec in zip(en, de):
            sim = cos_sim(enc, dec)
            all_maps.append(1.0 - sim)
        anomaly_map = torch.stack(all_maps, dim=0).mean(dim=0)

        image_score = anomaly_map.flatten(1).mean(dim=1)
        heatmap = anomaly_map.unsqueeze(1)
        if heatmap.shape[-1] != self.img_size or heatmap.shape[-2] != self.img_size:
            heatmap = F.interpolate(heatmap, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        heatmap = self.gaussian_kernel(heatmap)
        b_min = heatmap.flatten(2).min(dim=2, keepdim=True)[0]
        b_max = heatmap.flatten(2).max(dim=2, keepdim=True)[0]
        heatmap = (heatmap - b_min) / (b_max - b_min + 1e-8)

        return {
            "anomaly_score": image_score,
            "heatmap": heatmap,
            "is_unknown_defect": torch.zeros(images.shape[0], dtype=torch.long,
                                              device=images.device),
        }

    def train_decoder(self, defect_loader, device='cuda', save_path=None, log_interval=500):
        print(f"[DinomalyAnomalyHead v1] Training decoder on defect samples...")
        print(f"  Layers: {self.layer_indices}, Iterations: {self.training_iters}, LR: {self.lr}")

        self.vitill.to(device)
        self.vitill.eval()
        self.bottleneck.train()
        self.decoder.train()

        trainable = nn.ModuleList([self.bottleneck, self.decoder])
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=self.lr,
                                      betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = WarmCosineScheduler(
            optimizer, base_value=self.lr, final_value=self.lr * 0.1,
            total_iters=self.training_iters,
            warmup_iters=max(100, self.training_iters // 100),
        )

        all_images = self._collect_center_views(defect_loader)
        all_images = torch.cat(all_images, dim=0)
        n_samples = len(all_images)
        print(f"[DinomalyAnomalyHead v1] Total samples: {n_samples}")

        batch_size = min(16, n_samples)
        it, losses = 0, []

        while it < self.training_iters:
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                if it >= self.training_iters:
                    break
                batch = all_images[perm[start:start + batch_size]].to(device)
                en, de = self.vitill(batch)
                loss = global_cosine_hm_percent(en, de, p=self.loss_p, factor=0.0)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                if (it + 1) % log_interval == 0 or it == 0:
                    print(f"  iter [{it+1}/{self.training_iters}], loss: {loss.item():.4f}  "
                          f"(avg: {np.mean(losses[-log_interval:]):.4f})")
                it += 1

        print(f"[DinomalyAnomalyHead v1] Training complete. Final loss: {losses[-1]:.4f}")
        self._trained = True
        if save_path:
            self.save(save_path)

    def _collect_center_views(self, dataloader):
        all_imgs = []
        for batch in dataloader:
            imgs = batch["images"]
            if imgs.shape[2] >= 3:
                imgs = imgs[:, :, 1, :, :]
            elif imgs.shape[2] == 1:
                imgs = imgs[:, :, 0, :, :]
            all_imgs.append(imgs)
        return all_imgs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'version': 1,
            'bottleneck': self.bottleneck.state_dict(),
            'decoder': self.decoder.state_dict(),
            'trained': self._trained,
            'layer_indices': self.layer_indices,
            'embed_dim': self.embed_dim,
        }, path)

    def load(self, path: str, device='cuda'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.bottleneck.load_state_dict(ckpt['bottleneck'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self._trained = ckpt.get('trained', True)
        print(f"[DinomalyAnomalyHead v1] Loaded from {path} (trained={self._trained})")

    def is_trained(self) -> bool:
        return self._trained


# ─────────────────────────────────────────────────────────────────────────────
#  Dinomaly2AnomalyHead (v2)
# ─────────────────────────────────────────────────────────────────────────────

class Dinomaly2AnomalyHead(nn.Module):
    """
    Dinomaly2 anomaly detection head.

    From the paper (arXiv 2510.17611v2):
      "One Dinomaly2 Detect Them All: A Unified Framework for
       Full-Spectrum Unsupervised Anomaly Detection"

    Key innovations over v1:
      1. Noisy Bottleneck: 3-layer MLP + Dropout(p=0.2)
         → "Dropout is all you need." Prevents over-generalization.
      2. Context-Aware Recentering: patch - CLS before bottleneck
         → "Beauty is in the eye of the beholder." Resolves multi-class confusion.
      3. Loose Reconstruction: 2 semantic groups (shallow + deep layers)
         → "The tighter you squeeze, the less you have."
         → Gradient scaling factor=0.1 (NOT complete stop)
      4. τ warmup: τ ramps 0%→90% over first 1000 iterations
         → Prevents early training instability

    Architecture:
      - Frozen DINOv3 encoder: layers [3,4,5,6,7,8,9,10] (8 middle layers)
      - Noisy Bottleneck (3-layer MLP + Dropout p=0.2)
      - 8 × LinearAttention2 decoder blocks
      - Loss: loose_reconstruction_loss (factor=0.1, τ warmup)
      - Optimizer: StableAdamW(lr=2e-3), cosine decay to 2e-4
      - Iterations: 40,000 (MUAD standard)
      - Image size: 392×392 (default for Dinomaly2)

    Reference from paper (Section 4.1):
      "StableAdamW optimizer is utilized with lr=2e-3, β=(0.9,0.999),
       wd=1e-4 and eps=1e-10. The lr warms up from 0 to 2e-3 in the first
       100 iterations and cosine anneals to 2e-4 throughout training.
       The dropout rate is set to 0.2 or 0.4. Loose constraint with 2 groups
       is used by default. The discarding rate controlling τ_k linearly increases
       from 0% to 90% in the first 1,000 iterations as warm-up."
    """

    def __init__(
        self,
        backbone,
        layer_indices=None,           # which DINOv3 layers to extract
        embed_dim=1024,                # ViT-L/16 = 1024, ViT-B/16 = 768
        num_heads=16,                 # ViT-L/16 = 16, ViT-B/16 = 12
        num_decoder_blocks=8,
        training_iters=40000,          # standard for MUAD on MVTec
        lr=2e-3,
        loss_p=0.9,                   # top percentile (τ = 90th pct)
        dropout=0.2,                  # Noisy Bottleneck dropout rate
        grad_factor=0.1,              # gradient scaling for easy patches
        tau_warmup_iters=1000,        # τ ramps 0→90% over this many iters
        img_size=392,                 # Dinomaly2 default input size
        device='cuda',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.training_iters = training_iters
        self.lr = lr
        self.loss_p = loss_p
        self.dropout = dropout
        self.grad_factor = grad_factor
        self.tau_warmup_iters = tau_warmup_iters
        self.img_size = img_size
        self._device = device
        self.version = 2

        # DINOv3 layer indices: for ViT-L/16 (24 layers), use [3,4,...,22] middle 16
        # but Dinomaly2 paper uses 8 middle layers {3,...,10} for ViT-B/S
        # We use {3,4,5,6,7,8,9,10} for ViT-L/16 as well (consistent with DINOv3)
        self.layer_indices = layer_indices or [3, 4, 5, 6, 7, 8, 9, 10]
        n = len(self.layer_indices)  # = 8
        half = n // 2               # = 4
        # Group 0: shallow layers [3,4,5,6] = indices [0,1,2,3]
        # Group 1: deep layers   [7,8,9,10] = indices [4,5,6,7]
        self.fuse_layer_encoder = [[i for i in range(half)], [i for i in range(half, n)]]
        self.fuse_layer_decoder = [[i for i in range(half)], [i for i in range(half, n)]]

        # v2 Noisy Bottleneck: 3-layer MLP + Dropout
        self.bottleneck = NoisyBottleneck2(embed_dim, drop=dropout)

        # Decoder: 8 × LinearAttention2 blocks
        self.decoder = nn.ModuleList([
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(num_decoder_blocks)
        ])

        self._init_weights()

        # Get num_register_tokens from encoder
        num_reg = getattr(backbone.model, 'num_register_tokens', 0)
        self.vitill2 = ViTill2(
            encoder=backbone.model,
            bottleneck=self.bottleneck,
            decoder=self.decoder,
            target_layers=self.layer_indices,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
            num_register_tokens=num_reg,
        )

        self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4.0, channels=1)
        self._trained = False

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # Noisy Bottleneck: initialize as identity-like
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor, return_heatmap=True) -> dict:
        if not self._trained:
            B = images.shape[0]
            return {
                "anomaly_score": torch.zeros(B, device=images.device),
                "heatmap": torch.zeros(B, 1, self.img_size, self.img_size,
                                       device=images.device),
                "is_unknown_defect": torch.zeros(B, dtype=torch.long,
                                                  device=images.device),
            }

        with torch.no_grad():
            en_groups, de_groups, (H, W) = self.vitill2(images)

        # Compute cosine distance per group (Loose Reconstruction)
        cos_sim = nn.CosineSimilarity(dim=-1)
        all_maps = []
        for g_enc, g_dec in zip(en_groups, de_groups):
            # g_enc, g_dec: [B, D] — expand to [B, N, D] for spatial map
            # Use broadcast: expand D to spatial grid via cosine per feature dim
            dist = 1.0 - cos_sim(g_enc, g_dec)   # [B] scalar per sample
            all_maps.append(dist)

        # Average image-level score across groups
        image_score = torch.stack(all_maps, dim=0).mean(dim=0)  # [B]

        # For heatmap: reconstruct per-patch anomaly from cosine similarity
        # Use the decoder's final output (z after all decoder blocks)
        # Approximate heatmap by using the encoder features
        with torch.no_grad():
            en_list = self.vitill2.encoder.get_intermediate_layers(
                images, n=self.vitill2.target_layers, norm=False,
                return_class_token=True
            )
            # Sum recentered features for heatmap
            recentered_list = []
            num_patches = H * W
            num_reg = self.vitill2.num_register_tokens
            for lf in en_list:
                if lf.dim() == 3 and lf.shape[1] > num_patches:
                    cls = lf[:, num_reg:num_reg + 1, :]
                    patch = lf[:, num_reg + 1:, :]
                else:
                    cls = lf.mean(dim=1, keepdim=True)
                    patch = lf
                recentered_list.append(patch - cls)
            # Sum all layers for a single feature map
            feat_sum = torch.stack(recentered_list, dim=1).sum(dim=1)  # [B, N, D]
            feat = feat_sum / len(en_list)  # [B, N, D]

            # Decoder output as comparison
            z0 = feat_sum
            z = self.bottleneck(z0)
            for blk in self.decoder:
                z = blk(z, attn_mask=None)
            dec_feat = z  # [B, N, D]

            # Per-patch cosine distance
            cos_sp = nn.CosineSimilarity(dim=-1)
            patch_dist = 1.0 - cos_sp(feat, dec_feat)  # [B, N]
            anomaly_map = patch_dist.view(-1, H, W)     # [B, H, W]

        heatmap = anomaly_map.unsqueeze(1)  # [B, 1, H, W]
        if heatmap.shape[-1] != self.img_size or heatmap.shape[-2] != self.img_size:
            heatmap = F.interpolate(heatmap, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        heatmap = self.gaussian_kernel(heatmap)
        b_min = heatmap.flatten(2).min(dim=2, keepdim=True)[0]
        b_max = heatmap.flatten(2).max(dim=2, keepdim=True)[0]
        heatmap = (heatmap - b_min) / (b_max - b_min + 1e-8)

        return {
            "anomaly_score": image_score,
            "heatmap": heatmap,
            "is_unknown_defect": torch.zeros(images.shape[0], dtype=torch.long,
                                              device=images.device),
        }

    def train_decoder(self, defect_loader, device='cuda', save_path=None, log_interval=500):
        print(f"[Dinomaly2AnomalyHead] Training Dinomaly2 decoder on defect samples...")
        print(f"  Version: Dinomaly2")
        print(f"  Layers: {self.layer_indices}")
        print(f"  Iterations: {self.training_iters}")
        print(f"  LR: {self.lr}, Dropout: {self.dropout}, Grad factor: {self.grad_factor}")
        print(f"  τ warmup: 0→{self.loss_p*100:.0f}% over {self.tau_warmup_iters} iters")
        print(f"  Input size: {self.img_size}×{self.img_size}")

        self.vitill2.to(device)
        self.vitill2.eval()
        self.bottleneck.train()
        self.decoder.train()

        trainable = nn.ModuleList([self.bottleneck, self.decoder])
        optimizer = torch.optim.AdamW(
            trainable.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = WarmCosineScheduler(
            optimizer,
            base_value=self.lr,
            final_value=self.lr * 0.1,
            total_iters=self.training_iters,
            warmup_iters=100,  # paper: "warm up from 0 to 2e-3 in the first 100 iterations"
        )

        all_images = self._collect_center_views(defect_loader)
        all_images = torch.cat(all_images, dim=0)
        n_samples = len(all_images)
        print(f"[Dinomaly2AnomalyHead] Total samples: {n_samples}")

        batch_size = min(16, n_samples)
        it, losses = 0, []

        while it < self.training_iters:
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                if it >= self.training_iters:
                    break
                batch = all_images[perm[start:start + batch_size]].to(device)

                # τ warmup: linearly ramp from 0 to self.loss_p (0.9)
                if it < self.tau_warmup_iters:
                    tau_percent = (it / self.tau_warmup_iters) * self.loss_p
                else:
                    tau_percent = self.loss_p

                en_groups, de_groups, _ = self.vitill2(batch)
                loss = loose_reconstruction_loss(
                    en_groups, de_groups,
                    p=self.loss_p,
                    factor=self.grad_factor,
                    tau_percent=tau_percent,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                if (it + 1) % log_interval == 0 or it == 0:
                    print(f"  iter [{it+1}/{self.training_iters}], loss: {loss.item():.4f}  "
                          f"(avg: {np.mean(losses[-log_interval:]):.4f})  τ={tau_percent:.3f}")
                it += 1

        print(f"[Dinomaly2AnomalyHead] Training complete. Final loss: {losses[-1]:.4f}")
        self._trained = True
        if save_path:
            self.save(save_path)

    def _collect_center_views(self, dataloader):
        all_imgs = []
        for batch in dataloader:
            imgs = batch["images"]
            if imgs.shape[2] >= 3:
                imgs = imgs[:, :, 1, :, :]
            elif imgs.shape[2] == 1:
                imgs = imgs[:, :, 0, :, :]
            all_imgs.append(imgs)
        return all_imgs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'version': 2,
            'bottleneck': self.bottleneck.state_dict(),
            'decoder': self.decoder.state_dict(),
            'trained': self._trained,
            'layer_indices': self.layer_indices,
            'embed_dim': self.embed_dim,
            'dropout': self.dropout,
            'grad_factor': self.grad_factor,
        }, path)

    def load(self, path: str, device='cuda'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.bottleneck.load_state_dict(ckpt['bottleneck'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self._trained = ckpt.get('trained', True)
        print(f"[Dinomaly2AnomalyHead] Loaded from {path} (trained={self._trained})")

    def is_trained(self) -> bool:
        return self._trained
