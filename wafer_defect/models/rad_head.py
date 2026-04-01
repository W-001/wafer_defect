"""
RAD: Multi-Layer Patch-KNN Anomaly Detection.

Based on "RAD: Unleashing Masked Autoencoder for Anomaly Detection with DINOv3"
https://github.com/longkukuhi/RAD

Key idea: Build a multi-layer patch feature memory bank from known defect samples,
then score test patches via KNN against the bank.

Architecture:
  - Extract intermediate layer features from DINOv3 (e.g. layers [3, 6, 9, 11])
  - Bank: cls_banks[layer] [N,C] + patch_banks[layer] [N,L,C]
  - Inference: for each layer, find top-k nearest training images (cls similarity),
    then compute patch-level anomaly score (1 - max cosine sim to neighbor patches).
  - Fuse scores across layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional
import os
import numpy as np


def get_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> nn.Conv2d:
    """Create a 2D Gaussian smoothing kernel."""
    kernel = torch.tensor([
        [i * j for j in range(kernel_size)]
        for i in range(kernel_size)
    ], dtype=torch.float32)
    kernel = torch.exp(-kernel / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    ch = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
    ch.weight.data = kernel
    ch.requires_grad_(False)
    return ch


class RADAnomalyHead(nn.Module):
    """
    RAD-based anomaly detection head using multi-layer Patch-KNN.

    For 3-view wafer images: extract features per view, fuse at layer level,
    build bank from known defect training samples.

    Inference:
      1. Extract multi-layer features for test sample
      2. Find top-k nearest neighbor training images (CLS similarity)
      3. Compute patch-level KNN score (1 - max cosine sim)
      4. Multi-layer linear fusion + Gaussian smoothing
      5. High anomaly score → unknown/novel defect
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: List[int] = None,
        k_image: int = 5,
        layer_weights: List[float] = None,
        resize_mask: int = 224,
        anomaly_threshold: float = 2.0,
        bank_path: str = None,
        use_positional_bank: bool = False,
        pos_radius: int = 1,
    ):
        """
        Args:
            backbone: DINOv3Backbone wrapper (must have get_intermediate_layers)
            layer_indices: 0-based layer indices to extract (e.g. [3, 6, 9, 11])
            k_image: top-K nearest neighbor training images
            layer_weights: fusion weights per layer (default: uniform)
            resize_mask: spatial size to resize anomaly map to
            anomaly_threshold: z-score threshold for flagging unknown defects
            bank_path: path to pre-built memory bank .pth
            use_positional_bank: whether to use position-aware patch KNN
            pos_radius: neighborhood radius for positional bank
        """
        super().__init__()
        self.backbone = backbone
        self.layer_indices = layer_indices or [3, 6, 9, 11]
        self.k_image = k_image
        self.layer_weights = layer_weights
        self.resize_mask = resize_mask
        self.anomaly_threshold = anomaly_threshold
        self.bank_path = bank_path
        self.use_positional_bank = use_positional_bank
        self.pos_radius = pos_radius

        # Banks will be built or loaded
        self._banks_built = False
        self.register_buffer("_cls_banks", torch.zeros(1))
        self.register_buffer("_patch_banks", torch.zeros(1))

        # Normalization stats for z-score threshold
        self.register_buffer("_score_mean", torch.tensor(0.0))
        self.register_buffer("_score_std", torch.tensor(1.0))

        # Gaussian kernel for smoothing
        self._gauss = None

    @property
    def cls_banks(self):
        return self._cls_banks

    @property
    def patch_banks(self):
        return self._patch_banks

    def _ensure_gauss(self, device):
        if self._gauss is None:
            self._gauss = get_gaussian_kernel(5, 1.0).to(device)
        return self._gauss

    # ─────────────────────────────────────────────────────────────────────────
    #  Bank building
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def build_bank(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        save_path: str = None,
    ):
        """
        Build multi-layer feature memory bank from training defect samples.

        Args:
            dataloader: DataLoader yielding {'images': [B,3,V,H,W], ...}
            device: target device
            save_path: optional path to save bank .pth
        """
        self.backbone.eval()
        num_layers = len(self.layer_indices)
        cls_feats = [[] for _ in range(num_layers)]
        patch_feats = [[] for _ in range(num_layers)]
        labels_list = []

        total = len(dataloader)
        for step, batch in enumerate(dataloader):
            images = batch["images"].to(device)  # [B, V, C, H, W]  V=3 views, C=3 channels
            defect_types = batch.get("defect_type", torch.zeros(len(images), dtype=torch.long, device=device))
            batch_size = images.shape[0]

            # Extract center view (index 1) for feature extraction
            # images[:, 1, :, :, :] → [B, C, H, W]
            imgs = images[:, 1, :, :, :]

            inter = self.backbone.get_intermediate_layers(
                imgs,
                n=self.layer_indices,
                return_class_token=True,
                norm=True,
            )

            for li, (patch_tok, cls_tok) in enumerate(inter):
                cls_feats[li].append(cls_tok.cpu())
                patch_feats[li].append(patch_tok.cpu())

            labels_list.append(defect_types.cpu())

            if step % 50 == 0:
                print(f"[RAD Bank] step={step}/{total}")

        # Concatenate
        cls_banks = [torch.cat(cf, dim=0) for cf in cls_feats]
        patch_banks = [torch.cat(pf, dim=0) for pf in patch_feats]
        labels = torch.cat(labels_list, dim=0)

        self._cls_banks = cls_banks
        self._patch_banks = patch_banks
        self._defect_labels = labels
        self._banks_built = True

        print(f"[RAD Bank] Built bank: {num_layers} layers")
        for li, (cls_b, patch_b) in enumerate(zip(cls_banks, patch_banks)):
            print(f"  Layer {self.layer_indices[li]}: "
                  f"cls={tuple(cls_b.shape)}, patch={tuple(patch_b.shape)}")

        # Calibrate threshold stats from bank
        self._calibrate_threshold(cls_banks, patch_banks)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({
                "layer_indices": self.layer_indices,
                "cls_banks": cls_banks,
                "patch_banks": patch_banks,
                "defect_labels": labels,
                "layer_weights": self.layer_weights,
                "k_image": self.k_image,
            }, save_path)
            print(f"[RAD Bank] Saved to {save_path}")

    def _calibrate_threshold(self, cls_banks, patch_banks):
        """Calibrate anomaly score normalization from bank statistics."""
        device = cls_banks[0].device if isinstance(cls_banks[0], torch.Tensor) else "cpu"

        # Use CLS similarity distribution to estimate normal score range
        cls_bank0 = F.normalize(cls_banks[0].to(device), dim=-1)
        # Sample some pairs for speed
        n = min(1000, cls_bank0.shape[0])
        idx = torch.randperm(cls_bank0.shape[0])[:n]
        sample_cls = F.normalize(cls_bank0[idx], dim=-1)
        sim = torch.matmul(sample_cls, sample_cls.t())
        sim = sim.flatten()
        sim = sim[sim < 0.999]  # exclude self-similarity
        self._score_mean = sim.mean()
        self._score_std = sim.std() + 1e-6
        print(f"[RAD] Score calibration: mean={self._score_mean:.4f}, std={self._score_std:.4f}")

    @torch.no_grad()
    def load_bank(self, path: str):
        """Load a pre-built memory bank."""
        ckpt = torch.load(path, map_location="cpu")
        self.layer_indices = ckpt["layer_indices"]
        self._cls_banks = ckpt["cls_banks"]
        self._patch_banks = ckpt["patch_banks"]
        self._defect_labels = ckpt.get("defect_labels", None)
        if ckpt.get("layer_weights"):
            self.layer_weights = ckpt["layer_weights"]
        if ckpt.get("k_image"):
            self.k_image = ckpt["k_image"]
        self._banks_built = True
        print(f"[RAD Bank] Loaded from {path}")
        for li, (cb, pb) in enumerate(zip(self._cls_banks, self._patch_banks)):
            print(f"  Layer {self.layer_indices[li]}: cls={tuple(cb.shape)}, patch={tuple(pb.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Forward (inference)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        return_map: bool = False
    ) -> dict:
        """
        Compute RAD anomaly scores for a batch.

        Args:
            images: torch.Tensor,  # [B, V, C, H, W]  V=3 views
            return_map: whether to return per-patch anomaly maps

        Returns:
            dict with:
              - anomaly_score: [B] per-sample anomaly score
              - is_unknown_defect: [B] binary (1=unknown, 0=known)
              - anomaly_map: [B, H, W] spatial anomaly map (if return_map)
              - patch_scores: [B, L] per-patch scores (if return_map)
        """
        if not self._banks_built:
            raise RuntimeError(
                "RAD bank not built! Call build_bank() or load_bank() first."
            )

        B, V, C, H, W = images.shape  # [B, V, C, H, W]  V=3 views
        device = images.device

        # Use center view (index 1) for feature extraction
        imgs = images[:, 1, :, :, :]  # [B, C, H, W]

        # Extract multi-layer features
        inter = self.backbone.get_intermediate_layers(
            imgs,
            n=self.layer_indices,
            return_class_token=True,
            norm=True,
        )

        patch_list = []
        cls_list = []
        for (patch_tok, cls_tok) in inter:
            patch_list.append(patch_tok)    # [B, L, D]
            cls_list.append(cls_tok)         # [B, D]

        num_layers = len(self.layer_indices)

        # Move banks to device
        cls_banks_dev = [cb.to(device) for cb in self._cls_banks]
        patch_banks_dev = [pb.to(device) for pb in self._patch_banks]

        # Image-level nearest neighbors using highest layer CLS
        cls_bank_global = F.normalize(cls_banks_dev[-1], dim=-1)  # [N_bank, D]
        cls_x_global = F.normalize(cls_list[-1], dim=-1)          # [B, D]
        sim_img = torch.matmul(cls_x_global, cls_bank_global.t())  # [B, N_bank]
        _, topk_idx = torch.topk(sim_img, min(self.k_image, sim_img.shape[-1]), dim=-1)  # [B, k]

        # Spatial resolution
        L = patch_list[0].shape[1]  # number of patches
        h = w = int(L ** 0.5)

        # Layer weights
        if self.layer_weights is None or len(self.layer_weights) != num_layers:
            layer_weights = [1.0 / num_layers] * num_layers
        else:
            s = sum(self.layer_weights)
            layer_weights = [w / s for w in self.layer_weights]

        # Multi-layer patch-KNN
        patch_scores_batch = []
        for b in range(B):
            neigh_indices = topk_idx[b]  # [k]

            scores_per_layer = []
            for li in range(num_layers):
                q_feat = F.normalize(patch_list[li][b], dim=-1)      # [L, D]
                bank_l = F.normalize(patch_banks_dev[li][neigh_indices], dim=-1)  # [k, L_bank, D]

                if self.use_positional_bank:
                    # Position-aware KNN
                    patch_scores_l = []
                    for j in range(L):
                        r, c = j // w, j % w
                        r_min = max(0, r - self.pos_radius)
                        r_max = min(h - 1, r + self.pos_radius)
                        c_min = max(0, c - self.pos_radius)
                        c_max = min(w - 1, c + self.pos_radius)
                        idx_list = []
                        for rr in range(r_min, r_max + 1):
                            for cc in range(c_min, c_max + 1):
                                idx_list.append(rr * w + cc)
                        idx = torch.tensor(idx_list, device=device, dtype=torch.long)
                        neigh_local = bank_l[:, idx, :].reshape(-1, q_feat.shape[-1])  # [k*Kpos, D]
                        if neigh_local.numel() == 0:
                            patch_scores_l.append(torch.tensor(1.0, device=device))
                            continue
                        sim = torch.matmul(q_feat[j:j+1], neigh_local.t())  # [1, k*Kpos]
                        nn_sim = sim.max(dim=-1)[0]  # [1]
                        patch_scores_l.append((1.0 - nn_sim.squeeze(0)))
                    patch_score_l = torch.stack(patch_scores_l, dim=0)
                else:
                    # Global patch-KNN
                    bank_flat = bank_l.reshape(-1, q_feat.shape[-1])  # [k*L_bank, D]
                    sim = torch.matmul(q_feat, bank_flat.t())         # [L, k*L_bank]
                    nn_sim = sim.max(dim=-1)[0]                       # [L]
                    patch_score_l = 1.0 - nn_sim                      # [L]

                scores_per_layer.append(patch_score_l)

            # Multi-layer linear fusion
            fused = torch.zeros_like(scores_per_layer[0])
            for li in range(num_layers):
                fused = fused + layer_weights[li] * scores_per_layer[li]
            patch_scores_batch.append(fused)

        patch_scores_batch = torch.stack(patch_scores_batch, dim=0)  # [B, L]

        # Reshape to spatial map and upsample
        patch_maps = patch_scores_batch.view(B, 1, h, w)  # [B, 1, h, w]
        anomaly_maps = F.interpolate(
            patch_maps,
            size=self.resize_mask,
            mode="bilinear",
            align_corners=False
        )  # [B, 1, H', W']

        # Gaussian smoothing
        gauss = self._ensure_gauss(device)
        anomaly_maps = gauss(anomaly_maps)

        # Per-sample anomaly score: max over spatial map
        sample_scores = anomaly_maps.flatten(1).max(dim=1)[0]  # [B]

        # Z-score normalization (_score_mean/_score_std are registered buffers, always tensors)
        score_mean = self._score_mean.to(device)
        score_std = self._score_std.to(device)
        z_scores = (sample_scores - score_mean) / (score_std + 1e-8)

        # Unknown defect decision
        is_unknown = (z_scores > self.anomaly_threshold).long()

        result = {
            "anomaly_score": sample_scores,
            "z_score": z_scores,
            "is_unknown_defect": is_unknown,
        }

        if return_map:
            result["anomaly_map"] = anomaly_maps  # [B, 1, H', W']
            result["patch_scores"] = patch_scores_batch  # [B, L]

        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        percentile: float = 95
    ):
        """
        Calibrate anomaly threshold from known defect samples.

        Args:
            dataloader: DataLoader with known defect samples
            device: device
            percentile: percentile for threshold (default 95)
        """
        self.eval()
        all_scores = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(device)
                scores = self.forward(images)["anomaly_score"]
                all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=0).numpy()
        threshold = np.percentile(all_scores, percentile)
        z_threshold = (threshold - all_scores.mean()) / (all_scores.std() + 1e-8)

        print(f"[RAD Calibration] Score: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")
        print(f"[RAD Calibration] Threshold (p{percentile}): {threshold:.4f}")
        print(f"[RAD Calibration] Z-score threshold: {z_threshold:.4f}")

        self.anomaly_threshold = z_threshold
        self._score_mean = torch.tensor(all_scores.mean(), device="cpu")
        self._score_std = torch.tensor(all_scores.std(), device="cpu")

        return z_threshold
