"""
Inference Demo — Anomaly Localization Heatmap Visualization.

Supports both anomaly detection modes:
  1. RAD (multi-layer patch-KNN): uses per-patch scores → heatmap
  2. Class-center (distance-to-center): uses integrated-gradients → heatmap

Usage:
  # RAD mode (after training with --use_rad_anomaly):
  python wafer_defect/demo_inference.py \
      --checkpoint output/best_model.pt \
      --rad_bank output/rad_bank.pth \
      --data_dir /path/to/wafer_data \
      --use_dinov3 --use_rad_anomaly \
      --output_dir output/demo

  # Class-center mode (after training with default anomaly):
  python wafer_defect/demo_inference.py \
      --checkpoint output/best_model.pt \
      --data_dir /path/to/wafer_data \
      --use_dinov3 \
      --output_dir output/demo

  # Synthetic data (quick test, no real data needed):
  python wafer_defect/demo_inference.py \
      --synthetic --num_samples 20 --num_defect_classes 3 \
      --checkpoint output/best_model.pt \
      --output_dir output/demo
"""

import argparse
import csv
import os
import random
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from wafer_defect.data.dataset import (
    generate_synthetic_dataset, create_dataloaders,
    create_real_dataloaders, RealWaferDataset,
)
from wafer_defect.models import WaferDefectModel, WaferDefectModelSimple


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap rendering utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_heatmap(gray: np.ndarray, colormap='jet') -> np.ndarray:
    """
    Convert a 2D grayscale array to a 3-channel heatmap RGB image.
    gray: [H, W] in range [0, 1]
    Returns: [H, W, 3] uint8
    """
    import matplotlib.cm as cm
    default_cmap = {
        'jet': cm.get_cmap('jet'),
        'hot': cm.get_cmap('hot'),
        'inferno': cm.get_cmap('inferno'),
        'magma': cm.get_cmap('magma'),
    }
    cmap_fn = default_cmap.get(colormap, cm.get_cmap('jet'))
    rgba = cmap_fn(np.clip(gray, 0, 1))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """
    Overlay a heatmap on top of an RGB image.
    img_rgb: [H, W, 3] uint8
    heatmap: [H, W, 3] uint8
    Returns: [H, W, 3] uint8
    """
    out = img_rgb.copy().astype(np.float32)
    out = out * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a [3, H, W] or [H, W] torch.Tensor to PIL Image.
    Handles both [0,1] float and [0,255] uint8.
    """
    if img_tensor.dim() == 3:
        arr = img_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        arr = img_tensor.cpu().numpy()

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)

    # If single channel, convert to RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)

    return Image.fromarray(arr)


def draw_label(draw: ImageDraw.Draw, text: str, x: int, y: int,
                font_size: int = 14, fill: str = 'white',
                bg_fill: str = 'black', padding: int = 4):
    """Draw a text label with background box."""
    bbox = draw.textbbox((x, y), text)
    w = bbox[2] - bbox[0] + padding * 2
    h = bbox[3] - bbox[1] + padding * 2
    draw.rectangle([x - padding, y - padding, x + w, y + h], fill=bg_fill)
    draw.text((x + padding // 2, y), text, fill=fill)


# ─────────────────────────────────────────────────────────────────────────────
#  Class-center anomaly heatmap via Integrated Gradients approximation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_center_heatmap(model, images: torch.Tensor,
                                 device: str) -> torch.Tensor:
    """
    Class-center anomaly heatmap: compute per-patch distance to class center.

    Uses the CLS token as the class representative, then measures how far each
    patch is from the CLS direction (inverse cosine similarity as anomaly score).
    Falls back to uniform for CNN-backbone models (WaferDefectModelSimple).

    Returns [B, 1, H, W] spatial heatmap.
    """
    model.eval()
    B, V, C, H, W = images.shape

    with torch.no_grad():
        imgs = images[:, 1, :, :, :]  # center view [B, C, H, W]

        # ── ViT backbone: use CLS-to-patch distance ─────────────────────────
        if hasattr(model.backbone, 'get_intermediate_layers'):
            last_layer_idx = model.backbone.model.n_blocks - 1
            inter = model.backbone.get_intermediate_layers(
                imgs,
                n=[last_layer_idx],
                return_class_token=True,
                norm=True,
            )
            # inter is tuple of (patch_tokens, cls_token) per layer
            patch_tok, cls_tok = inter[0]   # [B, L, D], [B, D]

            L = patch_tok.shape[1]
            side = int(math.isqrt(L))
            if side * side != L:
                # Non-square: fallback to uniform
                heatmap = torch.ones(B, 1, side, side, device=device) / (side * side)
            else:
                # Cosine similarity per patch to CLS: anomaly = 1 - sim
                patch_norm = F.normalize(patch_tok, dim=-1)   # [B, L, D]
                cls_norm = F.normalize(cls_tok, dim=-1)       # [B, D]
                sim_to_cls = (patch_norm * cls_norm.unsqueeze(1)).sum(dim=-1)  # [B, L]
                anomaly_per_patch = 1.0 - sim_to_cls          # [B, L]
                heatmap = anomaly_per_patch.view(B, 1, side, side)  # [B, 1, side, side]

        # ── CNN backbone: use spatial feature variance ───────────────────────
        else:
            # Extract intermediate conv features before GlobalAvgPool
            x = imgs  # [B, C, H, W]
            feat_maps = []
            hook_handle = None

            def hook_fn(module, input, output):
                feat_maps.append(output)

            # Register hook on the last conv layer
            conv_layers = [m for m in model.backbone.modules()
                           if isinstance(m, nn.Conv2d)]
            if conv_layers:
                hook_handle = conv_layers[-1].register_forward_hook(hook_fn)
                _ = model.backbone.forward(images.view(B * V, C, H, W))
                if hook_handle:
                    hook_handle.remove()

            if feat_maps:
                fm = feat_maps[-1]  # [B, C', H', W']
                fm = fm.view(B, V, *fm.shape[-3:])[:, 1]  # [B, C', H', W']
                var_map = fm.var(dim=1, keepdim=True)    # [B, 1, H', W']
                heatmap = F.interpolate(var_map, size=(H, W),
                                        mode='bilinear', align_corners=False)
            else:
                heatmap = torch.zeros(B, 1, H, W, device=device)

        # Upsample to image resolution
        heatmap = F.interpolate(
            heatmap,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Per-image min-max normalize to [0, 1]
        for i in range(heatmap.size(0)):
            mn = heatmap[i].min()
            mx = heatmap[i].max()
            heatmap[i] = (heatmap[i] - mn) / (mx - mn + 1e-8)

        return heatmap


# ─────────────────────────────────────────────────────────────────────────────
#  Build anomaly heatmap per sample
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap_for_batch(model, images: torch.Tensor, device: str,
                              use_rad: bool, rad_head=None) -> torch.Tensor:
    """
    Returns [B, 1, H, W] spatial anomaly heatmap per sample.
    For RAD: use per-patch RAD scores (already computed).
    For class-center: use CLS-to-patch distance heatmap.
    """
    B, V, C, H, W = images.shape

    if use_rad and rad_head is not None:
        # RAD: forward with return_map=True
        rad_out = rad_head.forward(images, return_map=True)
        heatmap = rad_out["anomaly_map"]  # [B, 1, H', W']
        # Resize to match input image dimensions
        heatmap = F.interpolate(heatmap, size=(H, W),
                                mode='bilinear', align_corners=False)
        # Per-image min-max normalize to [0, 1]
        for i in range(heatmap.size(0)):
            mn = heatmap[i].min()
            mx = heatmap[i].max()
            heatmap[i] = (heatmap[i] - mn) / (mx - mn + 1e-8)
        return heatmap

    else:
        # Class-center / non-RAD: CLS-to-patch distance heatmap
        return compute_class_center_heatmap(model, images, device)


# ─────────────────────────────────────────────────────────────────────────────
#  Render a single sample result as a 3-panel image
# ─────────────────────────────────────────────────────────────────────────────

def render_sample(img_center: Image.Image,
                  heatmap_np: np.ndarray,
                  pred_is_defect: int,
                  pred_defect_type: int,
                  anomaly_score: float,
                  is_unknown: int,
                  true_label: str,
                  class_names: list,
                  gate_threshold: float = 0.5,
                  view_index: int = 1) -> Image.Image:
    """
    Render one sample as a horizontal 3-panel image:
      [Original view] [Anomaly heatmap] [Overlay]
    with info strip at the bottom.
    """
    W, H = img_center.size
    panel_h = H
    panel_w = W
    strip_h = 56
    gap = 4
    total_w = panel_w * 3 + gap * 2
    total_h = panel_h + strip_h

    canvas = Image.new('RGB', (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # ── Panel 1: Original image ──────────────────────────────────────────────
    canvas.paste(img_center, (0, 0))

    # ── Panel 2: Heatmap ──────────────────────────────────────────────────────
    heatmap_pil = Image.fromarray(heatmap_np)
    heatmap_pil = heatmap_pil.resize((panel_w, panel_h), Image.BILINEAR)
    canvas.paste(heatmap_pil, (panel_w + gap, 0))

    # ── Panel 3: Overlay ──────────────────────────────────────────────────────
    img_arr = np.array(img_center)
    overlay = overlay_heatmap(img_arr, heatmap_np)
    canvas.paste(Image.fromarray(overlay), (panel_w * 2 + gap * 2, 0))

    # ── Info strip ────────────────────────────────────────────────────────────
    defect_label = class_names[pred_defect_type] if pred_defect_type < len(class_names) else f"Unknown({pred_defect_type})"
    gate_status = "DEFECT" if pred_is_defect else "NUISANCE"
    gate_color = '#ff4444' if pred_is_defect else '#44ff88'
    unk_text = " | UNKNOWN DEFECT" if is_unknown else ""

    status = (
        f"[{gate_status}] {defect_label}"
        f"  |  Score={anomaly_score:.4f}{unk_text}"
        f"  |  True={true_label}"
    )

    draw.rectangle([0, panel_h, total_w, total_h], fill=(15, 15, 15))
    draw.text((6, panel_h + 6), status, fill='white')

    # Panel labels
    draw.text((4, 4), "Original", fill='white')
    draw.text((panel_w + gap + 4, 4), "Anomaly Heatmap", fill='white')
    draw.text((panel_w * 2 + gap * 2 + 4, 4), "Overlay", fill='white')

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  Grid layout — collect N samples into one big image
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(images: list, cols: int = 3, border: int = 2) -> Image.Image:
    """Arrange a list of PIL Images into a grid."""
    if not images:
        return Image.new('RGB', (10, 10))
    W, H = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB',
                     (cols * W + (cols + 1) * border,
                      rows * H + (rows + 1) * border),
                     (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * (W + border) + border
        y = r * (H + border) + border
        grid.paste(img, (x, y))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
#  Main demo
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_demo(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.synthetic or args.data_dir is None:
        print("[Demo] Using synthetic data...")
        train_samples, val_samples = generate_synthetic_dataset(
            num_samples=args.num_samples,
            num_defect_classes=args.num_defect_classes,
            nuisance_ratio=args.nuisance_ratio,
            seed=42
        )
        train_loader, val_loader = create_dataloaders(
            train_samples, val_samples,
            batch_size=args.batch_size,
            num_workers=0,
            crop_footer=True,
            footer_pixels=args.crop_bottom
        )
        real_dataset = None
    else:
        print(f"[Demo] Loading real data from: {args.data_dir}")
        real_num_workers = 0
        train_loader, val_loader = create_real_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=real_num_workers,
            img_size=args.img_size,
            crop_bottom=args.crop_bottom,
            nuisance_name=args.nuisance_name
        )
        real_dataset = train_loader.dataset.parent

    _ds = real_dataset if real_dataset else val_loader.dataset
    class_names = _ds.get_class_names() if hasattr(_ds, 'get_class_names') else None
    if class_names is None:
        n = args.num_defect_classes + 1
        class_names = [f"class_{i}" for i in range(n)]

    print(f"Class names: {class_names}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # ── 2. Build model ─────────────────────────────────────────────────────────
    print("\nBuilding model...")
    num_defect_classes = args.num_defect_classes if args.synthetic else (real_dataset.num_classes - 1)

    if args.use_dinov3:
        model = WaferDefectModel(
            num_defect_classes=num_defect_classes,
            backbone_name=args.backbone,
            pretrained_path=args.pretrained_path,
            embed_dim=args.embed_dim,
            use_rad_anomaly=args.use_rad_anomaly,
            rad_layer_indices=args.rad_layer_indices,
            rad_bank_path=args.rad_bank,
        )
    else:
        model = WaferDefectModelSimple(
            num_defect_classes=num_defect_classes,
            img_size=args.img_size,
            feat_dim=512
        )

    # ── 3. Load checkpoint ────────────────────────────────────────────────────
    use_rad = args.use_rad_anomaly

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        # Restore anomaly stats
        if hasattr(model, 'anomaly') and not use_rad:
            an = model.anomaly
            if 'anomaly_score_mean' in ckpt and hasattr(an, 'dist_mean'):
                an.dist_mean = torch.tensor(ckpt['anomaly_score_mean'])
            if 'anomaly_score_std' in ckpt and hasattr(an, 'dist_std'):
                an.dist_std = torch.tensor(ckpt['anomaly_score_std'])

        loaded_class_names = ckpt.get('class_names', None)
        if loaded_class_names:
            if isinstance(loaded_class_names, dict):
                class_names = loaded_class_names
            else:
                # list → dict
                class_names = {i: str(n) for i, n in enumerate(loaded_class_names)}
        print(f"Class names from checkpoint: {class_names}")
    else:
        if not args.checkpoint:
            print("[Demo] No checkpoint provided — running with random weights (visual only)")
        else:
            print(f"[Demo] Checkpoint not found: {args.checkpoint} — using random weights")

    model = model.to(device)
    model.eval()

    # ── 4. Build RAD bank if needed ───────────────────────────────────────────
    rad_head = None
    if use_rad:
        if args.rad_bank and os.path.exists(args.rad_bank):
            print(f"\nLoading RAD bank from: {args.rad_bank}")
            model.anomaly.load_bank(args.rad_bank)
        elif args.checkpoint and os.path.exists(args.checkpoint):
            # Try to load bank from checkpoint dir
            bank_alt = os.path.join(os.path.dirname(args.checkpoint), 'rad_bank.pth')
            if os.path.exists(bank_alt):
                model.anomaly.load_bank(bank_alt)
        else:
            if hasattr(model, 'build_rad_bank'):
                print("\nBuilding RAD bank from training data...")
                model.build_rad_bank(train_loader, device=device,
                                     save_path=os.path.join(args.output_dir, 'rad_bank.pth'))
            else:
                print("\n[Demo] Model does not support RAD bank (WaferDefectModelSimple)")

        rad_head = model.anomaly if hasattr(model, 'anomaly') else None

    # ── 5. Inference on validation set ───────────────────────────────────────
    print(f"\nRunning inference on {args.num_demo} samples...")
    batch = next(iter(val_loader))
    images = batch['images'].to(device)
    labels = batch['label'].cpu()
    is_defects = batch['is_defect'].cpu()
    defect_types = batch['defect_type'].cpu()

    # Limit to requested number
    n_show = min(args.num_demo, images.size(0))
    images = images[:n_show]
    labels = labels[:n_show]
    is_defects = is_defects[:n_show]
    defect_types = defect_types[:n_show]

    # Forward
    outputs = model(images, return_features=True)

    gate_prob = torch.softmax(outputs['gate_logits'], dim=1)
    fine_prob = torch.softmax(outputs['fine_logits'], dim=1)
    pred_is_defect = outputs['is_defect_pred'].cpu()
    pred_defect_type = outputs['fine_pred'].cpu()

    # Anomaly score + unknown defect
    anomaly_scores_raw = None
    is_unknown_list = [0] * n_show

    if use_rad and rad_head is not None:
        rad_out = rad_head.forward(images, return_map=True)
        anomaly_maps_raw = rad_out['anomaly_map']  # [B, 1, H', W']
        # Per-sample score
        sample_scores = rad_out['anomaly_score'].cpu()
        z_thresh = rad_head.anomaly_threshold if hasattr(rad_head, 'anomaly_threshold') else 2.0
        for i in range(n_show):
            z = (sample_scores[i].item() - rad_head._score_mean.item()) / \
                (rad_head._score_std.item() + 1e-8)
            is_unknown_list[i] = 1 if z > z_thresh else 0
        anomaly_scores_raw = sample_scores.numpy()
    elif hasattr(model, 'anomaly'):
        an = model.anomaly
        feats = outputs['feat']
        an_out = an(feats)
        anomaly_scores_raw = an_out['anomaly_score'].cpu().numpy()
        # is_unknown from z-score
        z_thresh = getattr(an, 'anomaly_threshold', 2.0)
        for i in range(n_show):
            z = anomaly_scores_raw[i]
            is_unknown_list[i] = 1 if z > z_thresh else 0

    # Build heatmaps
    heatmap_tensor = build_heatmap_for_batch(
        model, images, device, use_rad, rad_head
    )  # [B, 1, H, W]

    # ── 6. Render per-sample images ───────────────────────────────────────────
    rendered = []
    for i in range(n_show):
        # Get center view image
        img_center_np = images[i, 1].cpu().numpy()
        if img_center_np.max() <= 1.0:
            img_center_np = (img_center_np * 255).astype(np.uint8)
        else:
            img_center_np = img_center_np.astype(np.uint8)
        if img_center_np.shape[0] != 3:
            img_center_np = np.transpose(img_center_np, (1, 2, 0))
        img_center_np = np.clip(img_center_np, 0, 255).astype(np.uint8)
        img_center_pil = Image.fromarray(img_center_np)

        # Heatmap
        h_map = heatmap_tensor[i, 0].cpu().numpy()  # [H, W], already [0,1]
        heatmap_rgb = apply_heatmap(h_map, colormap='inferno')

        # True label name
        true_label = labels[i].item()
        true_name = class_names.get(true_label, f"class_{true_label}") \
            if isinstance(class_names, dict) else \
            (class_names[true_label] if true_label < len(class_names) else f"class_{true_label}")

        # Anomaly score
        a_score = float(anomaly_scores_raw[i]) if anomaly_scores_raw is not None else 0.0

        # fine_pred is 0~(K-1); dataset labels are 1~K, so +1 for display
        pd_type = pred_defect_type[i].item() + 1

        panel = render_sample(
            img_center=img_center_pil,
            heatmap_np=heatmap_rgb,
            pred_is_defect=pred_is_defect[i].item(),
            pred_defect_type=pd_type,
            anomaly_score=a_score,
            is_unknown=is_unknown_list[i],
            true_label=true_name,
            class_names=class_names if isinstance(class_names, list) else
                        [class_names.get(k, f"class_{k}") for k in sorted(class_names)],
        )
        rendered.append(panel)

    # ── 7. Save output ────────────────────────────────────────────────────────
    cols = min(args.cols, n_show)
    grid = make_grid(rendered, cols=cols)
    grid_path = os.path.join(args.output_dir, 'anomaly_heatmap_grid.png')
    grid.save(grid_path)
    print(f"\n[Demo] Heatmap grid saved → {grid_path}")

    # Also save per-sample images
    for i, panel in enumerate(rendered):
        out_path = os.path.join(args.output_dir, f'sample_{i:03d}.png')
        panel.save(out_path)

    # Save CSV summary
    csv_path = os.path.join(args.output_dir, 'inference_results.csv')
    cls_list = class_names if isinstance(class_names, list) else \
                [class_names.get(k, f"class_{k}") for k in sorted(class_names)]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'true_label', 'pred_gate', 'pred_defect_type',
                    'gate_defect_prob', 'anomaly_score', 'is_unknown', 'ok'])
        for i in range(n_show):
            a = float(anomaly_scores_raw[i]) if anomaly_scores_raw is not None else 0.0
            pd_idx = pred_defect_type[i].item() + 1  # fine pred 0~(K-1) → dataset 1~K
            ok = (is_defects[i].item() == pred_is_defect[i].item())
            w.writerow([
                i,
                cls_list[labels[i].item()] if labels[i].item() < len(cls_list) else labels[i].item(),
                'Defect' if pred_is_defect[i].item() else 'Nuisance',
                cls_list[pd_idx] if pd_idx < len(cls_list) else pd_idx,
                f"{gate_prob[i, 1].item():.4f}",
                f"{a:.4f}",
                'YES' if is_unknown_list[i] else 'NO',
                '✓' if ok else '✗'
            ])
    print(f"[Demo] Results CSV saved → {csv_path}")

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    print(f"{'#':>3}  {'True':>15}  {'Pred':>15}  {'Score':>8}  {'Unknown':>8}  {'OK'}")
    print("-" * 70)
    cls_list = class_names if isinstance(class_names, list) else \
                [class_names.get(k, f"class_{k}") for k in sorted(class_names)]
    for i in range(n_show):
        tn = cls_list[labels[i].item()] if labels[i].item() < len(cls_list) else f"class_{labels[i].item()}"
        pd_idx = pred_defect_type[i].item() + 1  # fine pred 0~(K-1) → dataset 1~K
        pn = cls_list[pd_idx] if pd_idx < len(cls_list) else f"class_{pd_idx}"
        a = f"{anomaly_scores_raw[i]:.4f}" if anomaly_scores_raw is not None else "N/A"
        unk = "YES" if is_unknown_list[i] else "no"
        ok = "✓" if (is_defects[i].item() == pred_is_defect[i].item()) else "✗"
        print(f"{i:>3}  {tn:>15}  {pn:>15}  {a:>8}  {unk:>8}  {ok}")
    print("=" * 70)
    print(f"\nGrid saved: {grid_path}")
    print(f"CSV saved:  {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Wafer Defect — Anomaly Heatmap Demo")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--rad_bank", type=str, default=None,
                        help="Path to RAD memory bank (.pth)")

    # Data source
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no real data needed)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real wafer data folder")
    parser.add_argument("--num_defect_classes", type=int, default=3)
    parser.add_argument("--nuisance_ratio", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--crop_bottom", type=int, default=40)
    parser.add_argument("--nuisance_name", type=str, default="Nuisance")

    # Model
    parser.add_argument("--use_dinov3", action="store_true")
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16")
    parser.add_argument("--pretrained_path", type=str,
                        default="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--embed_dim", type=int, default=1024)

    # RAD
    parser.add_argument("--use_rad_anomaly", action="store_true")
    parser.add_argument("--rad_layer_indices", nargs='+', type=int,
                        default=[3, 6, 9, 11])

    # Demo output
    parser.add_argument("--num_demo", type=int, default=12,
                        help="Number of samples to visualize")
    parser.add_argument("--cols", type=int, default=3,
                        help="Columns in output grid")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="output/demo")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)
