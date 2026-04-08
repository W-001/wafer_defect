"""
Dinomaly Wrapper for wafer_defect project.

This module provides integration with the open-source Dinomaly project
(https://github.com/cnulab/Dinomaly) for anomaly detection.

Based on the Dinomaly paper:
  "One Dinomaly Detect Them All: A Unified Framework for
   Unified Industrial Anomaly Detection and Reasoning"
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

# Add dinomaly_tmp to path
dinomaly_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dinomaly_tmp')
if dinomaly_path not in sys.path:
    sys.path.insert(0, dinomaly_path)

from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dinov1.utils import trunc_normal_
from utils import WarmCosineScheduler, global_cosine_hm_percent


class DinomalyAnomalyDetector(nn.Module):
    """
    Dinomaly-based anomaly detection for wafer defect detection.

    Uses DINOv3 as encoder and a lightweight decoder with bottleneck.
    """

    def __init__(
        self,
        img_size: int = 392,
        target_layers: list = None,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_decoder_blocks: int = 8,
        dropout: float = 0.2,
        lr: float = 2e-3,
        iters: int = 10000,
        **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.target_layers = target_layers or [3, 4, 5, 6, 7, 8, 9, 10]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.dropout = dropout
        self.lr = lr
        self.iters = iters

        # Fuse layer configuration (2 groups: shallow + deep)
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        # Load DINOv3 encoder
        encoder_name = 'dinov3_vitl16'
        encoder_weight = os.path.join(
            os.path.dirname(__file__), '..', '..', 'dinov3', 'weights',
            'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        )

        self.encoder = load_dinov3_model(
            encoder_name,
            layers_to_extract_from=self.target_layers,
            pretrained_weight_path=encoder_weight
        )

        # Build bottleneck and decoder
        self._build_bottleneck_decoder()

        # Build ViTill model
        self.model = ViTill(
            encoder=self.encoder,
            bottleneck=self.bottleneck,
            decoder=self.decoder,
            target_layers=self.target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder
        )

        self._trained = False

    def _build_bottleneck_decoder(self):
        """Build bottleneck and decoder modules."""
        # Bottleneck: reduces feature dimension
        self.bottleneck = nn.ModuleList([
            bMlp(self.embed_dim, self.embed_dim * 4, self.embed_dim, drop=self.dropout)
        ])

        # Decoder: 8 Transformer blocks with linear attention
        self.decoder = nn.ModuleList([
            VitBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2
            )
            for _ in range(self.num_decoder_blocks)
        ])

        # Initialize weights
        for m in self.bottleneck.modules():
            self._init_weights(m)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def train_decoder(self, train_images, device='cuda', save_path=None, log_interval=100):
        """
        Train the Dinomaly decoder.

        Args:
            train_images: Tensor of shape [N, 3, H, W] containing training images
            device: device to train on
            save_path: optional path to save trained model
            log_interval: logging interval
        """
        print(f"[Dinomaly] Training decoder...")
        print(f"  Images: {train_images.shape}")
        print(f"  Iterations: {self.iters}")
        print(f"  LR: {self.lr}")

        self.model = self.model.to(device)
        self.model.train()
        self.model.encoder.eval()  # Keep encoder frozen

        trainable = nn.ModuleList([self.bottleneck, self.decoder])

        # StableAdamW optimizer
        from optimizers import StableAdamW
        optimizer = StableAdamW(
            [{'params': trainable.parameters()}],
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=False,
            eps=1e-10
        )

        total_iters = self.iters
        # Adaptive warmup: use 10% of total_iters for warmup, min 10
        warmup_iters = max(10, int(total_iters * 0.1))
        lr_scheduler = WarmCosineScheduler(
            optimizer,
            base_value=self.lr,
            final_value=self.lr * 0.1,
            total_iters=total_iters,
            warmup_iters=warmup_iters
        )

        batch_size = min(12, len(train_images))
        # Adaptive log interval: log every 10% of iterations, min 5
        log_interval = max(5, int(total_iters * 0.1))
        it = 0

        while it < total_iters:
            perm = torch.randperm(len(train_images))
            for start in range(0, len(train_images), batch_size):
                if it >= total_iters:
                    break

                batch = train_images[perm[start:start + batch_size]].to(device)

                # Forward
                en, de = self.model(batch)

                # Loss with hard mining (p=0.9 keeps top 90% hardest patches)
                loss = global_cosine_hm_percent(en, de, p=0.9, factor=0.1)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()

                if (it + 1) % log_interval == 0:
                    print(f"  iter [{it+1}/{total_iters}], loss: {loss.item():.4f}", flush=True)

                it += 1

        print(f"[Dinomaly] Training complete. Final loss: {loss.item():.4f}")
        self._trained = True

        if save_path:
            self.save(save_path, device)

    def predict(self, images, return_heatmap=True):
        """
        Predict anomaly scores.

        Args:
            images: [B, 3, H, W] input images
            return_heatmap: whether to return anomaly heatmap

        Returns:
            dict with anomaly_score [B] and optionally heatmap [B, 1, H, W]
        """
        self.model.eval()

        with torch.no_grad():
            en, de = self.model(images)

            # Compute anomaly scores from encoder-decoder features
            cos_loss = nn.CosineSimilarity()
            scores = []
            for e, d in zip(en, de):
                # e, d: [B, C, H, W]
                score = 1 - cos_loss(e.flatten(2).transpose(1, 2).contiguous(),
                                     d.flatten(2).transpose(1, 2).contiguous())
                scores.append(score.mean(dim=-1))  # [B]

            anomaly_score = torch.stack(scores).mean(dim=0)  # [B]

            result = {
                'anomaly_score': anomaly_score,
            }

            if return_heatmap:
                # Create heatmap from cosine distance
                heatmaps = []
                for e, d in zip(en, de):
                    dist = 1 - nn.CosineSimilarity()(e, d)  # [B, H, W]
                    heatmaps.append(dist.unsqueeze(1))

                heatmap = torch.stack(heatmaps).mean(dim=0)  # [B, 1, H, W]
                heatmap = F.interpolate(heatmap, size=(self.img_size, self.img_size),
                                       mode='bilinear', align_corners=False)

                # Normalize to [0, 1]
                b_min = heatmap.flatten(2).min(dim=2, keepdim=True)[0]
                b_max = heatmap.flatten(2).max(dim=2, keepdim=True)[0]
                heatmap = (heatmap - b_min) / (b_max - b_min + 1e-8)

                result['heatmap'] = heatmap

            return result

    def forward(self, images, return_heatmap=True):
        """Forward method - alias for predict()."""
        return self.predict(images, return_heatmap=return_heatmap)

    def is_trained(self):
        return self._trained

    def save(self, path, device='cuda'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'bottleneck': self.bottleneck.state_dict(),
            'decoder': self.decoder.state_dict(),
            'trained': self._trained,
        }, path)
        print(f"[Dinomaly] Saved to {path}")

    def load(self, path, device='cuda'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.bottleneck.load_state_dict(ckpt['bottleneck'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self._trained = ckpt.get('trained', True)
        print(f"[Dinomaly] Loaded from {path} (trained={self._trained})")
