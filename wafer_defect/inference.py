"""
Inference Module for Wafer Defect Detection.

Unified inference interface for:
- Classification: Gate (Nuisance/Defect) + Fine (Defect Type)
- Anomaly Detection: Dinomaly2 reconstruction-based detection
- Visualization: Anomaly heatmap generation

Usage:
    from wafer_defect.inference import WaferDefectInferencer

    inferencer = WaferDefectInferencer(
        model_path='output/best_model.pt',
        device='cuda',
    )

    result = inferencer.predict(image)
    # result: {is_defect, defect_type, confidence, anomaly_score, heatmap}
"""

import argparse
import os
from pathlib import Path
from typing import Union, Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wafer_defect.data.dataset import (
    RealWaferDataset,
    create_real_dataloaders,
    generate_synthetic_dataset,
    create_dataloaders,
)
from wafer_defect.data.preprocessor import WaferPreprocessor, get_preprocessor
from wafer_defect.models.defect_model import WaferDefectModel, WaferDefectModelSimple


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_heatmap(gray: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Convert a 2D grayscale array to a 3-channel heatmap RGB image.
    gray: [H, W] in range [0, 1]
    Returns: [H, W, 3] uint8
    """
    try:
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
    except ImportError:
        # Fallback: simple grayscale
        return np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
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
    """Convert a [3, H, W] or [H, W] torch.Tensor to PIL Image."""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.cpu()
        if img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1, 2, 0)
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    return Image.fromarray(img_np)


# ─────────────────────────────────────────────────────────────────────────────
#  Main inferencer class
# ─────────────────────────────────────────────────────────────────────────────

class WaferDefectInferencer:
    """
    Unified inference interface for wafer defect detection.

    Supports:
    - Single image inference
    - Batch inference from DataLoader
    - Visualization with anomaly heatmaps
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        model_path: str = None,
        device: str = 'cuda',
        class_names: List[str] = None,
        use_dinomaly2: bool = True,
        use_synthetic: bool = False,
    ):
        self.device = device
        self.class_names = class_names or []

        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path, use_synthetic=use_synthetic)
        else:
            raise ValueError("Either model or model_path must be provided")

        self.model.to(device)
        self.model.eval()

        # Load Dinomaly2 if available
        if use_dinomaly2 and hasattr(self.model, 'dinomaly2'):
            dinomaly_path = model_path.replace('.pt', '_dinomaly2.pt') if model_path else None
            if dinomaly_path and os.path.exists(dinomaly_path):
                self.model.dinomaly2.load(dinomaly_path, device=device)

    def _load_model(self, model_path: str, use_synthetic: bool = False) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if use_synthetic:
            model = WaferDefectModelSimple(num_defect_classes=len(self.class_names))
        else:
            # Detect configuration from checkpoint
            num_classes = checkpoint.get('num_classes', len(self.class_names))
            model = WaferDefectModel(
                num_defect_classes=num_classes,
                backbone_name='dinov3_vitl16',
                pretrained_path='dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                use_dinomaly2=True,
            )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'classification' in checkpoint:
            model.classification.load_state_dict(checkpoint['classification'])

        return model

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        return_heatmap: bool = True,
    ) -> Dict:
        """
        Predict on a single image.

        Args:
            image: PIL Image, numpy array, or torch tensor
            return_heatmap: whether to return anomaly heatmap

        Returns:
            dict with:
                - is_defect: bool
                - defect_type: str (class name)
                - defect_type_idx: int
                - confidence: float
                - anomaly_score: float (if available)
                - heatmap: np.ndarray (if return_heatmap and available)
        """
        # Preprocess
        if isinstance(image, Image.Image):
            pass  # Already PIL
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get preprocessor
        preprocessor = get_preprocessor(img_size=392, crop_bottom=40)
        image_tensor = preprocessor(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Forward
        outputs = self.model(image_tensor, return_heatmap=return_heatmap)

        # Parse results
        is_defect = outputs['is_defect'].item() == 1
        defect_type_idx = outputs['defect_type'].item()
        gate_prob = outputs['gate_prob'][0].cpu().numpy()
        fine_prob = outputs['fine_prob'][0].cpu().numpy()

        result = {
            'is_defect': is_defect,
            'defect_type_idx': defect_type_idx,
            'defect_type': self.class_names[defect_type_idx] if self.class_names else str(defect_type_idx),
            'confidence': float(fine_prob.max()),
            'gate_prob_nuisance': float(gate_prob[0]),
            'gate_prob_defect': float(gate_prob[1]),
        }

        # Nuisance: return early
        if not is_defect:
            return result

        # Defect: add anomaly info
        if 'anomaly_score' in outputs and outputs['anomaly_score'] is not None:
            result['anomaly_score'] = float(outputs['anomaly_score'].item())

        if return_heatmap and 'heatmap' in outputs and outputs['heatmap'] is not None:
            heatmap = outputs['heatmap'][0, 0].cpu().numpy()  # [H, W]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            result['heatmap'] = heatmap

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_heatmap: bool = True,
    ) -> List[Dict]:
        """
        Predict on a batch of images from DataLoader.

        Args:
            dataloader: DataLoader yielding batches
            return_heatmap: whether to return anomaly heatmaps

        Returns:
            List of result dicts
        """
        results = []
        for batch in dataloader:
            images = batch['images'].to(self.device)
            outputs = self.model(images, return_heatmap=return_heatmap)

            B = images.shape[0]
            for i in range(B):
                is_defect = outputs['is_defect'][i].item() == 1
                defect_type_idx = outputs['defect_type'][i].item()

                result = {
                    'is_defect': is_defect,
                    'defect_type_idx': defect_type_idx,
                    'defect_type': self.class_names[defect_type_idx] if self.class_names else str(defect_type_idx),
                    'confidence': float(outputs['fine_prob'][i].max().item()),
                }

                if not is_defect:
                    results.append(result)
                    continue

                if 'anomaly_score' in outputs and outputs['anomaly_score'] is not None:
                    result['anomaly_score'] = float(outputs['anomaly_score'][i].item())

                if return_heatmap and 'heatmap' in outputs and outputs['heatmap'] is not None:
                    heatmap = outputs['heatmap'][i, 0].cpu().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    result['heatmap'] = heatmap

                results.append(result)

        return results

    def visualize(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        result: Dict = None,
        save_path: str = None,
        colormap: str = 'jet',
        show: bool = False,
    ) -> np.ndarray:
        """
        Visualize inference result with anomaly heatmap overlay.

        Args:
            image: input image
            result: prediction result (if None, will run predict)
            save_path: path to save visualization
            colormap: heatmap colormap
            show: whether to display

        Returns:
            visualization as numpy array
        """
        if result is None:
            result = self.predict(image, return_heatmap=True)

        # Get original image
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            img_rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        else:
            img_rgb = np.array(tensor_to_pil(image))

        # Resize if needed
        if 'heatmap' in result and result['heatmap'] is not None:
            heatmap = result['heatmap']
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
            heatmap_pil = heatmap_pil.resize((img_rgb.shape[1], img_rgb.shape[0]), Image.BILINEAR)
            heatmap = np.array(heatmap_pil) / 255.0

            # Generate heatmap visualization
            heatmap_color = apply_heatmap(heatmap, colormap=colormap)

            # Overlay on image
            vis = overlay_heatmap(img_rgb, heatmap_color, alpha=0.55)
        else:
            vis = img_rgb

        # Add text overlay
        from PIL import ImageFont, ImageDraw
        vis_pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis_pil)

        # Text info
        info_lines = []
        if result['is_defect']:
            info_lines.append(f"Type: {result['defect_type']}")
            info_lines.append(f"Conf: {result['confidence']:.2f}")
            if 'anomaly_score' in result:
                info_lines.append(f"Anomaly: {result['anomaly_score']:.4f}")
        else:
            info_lines.append("Nuisance")
            info_lines.append(f"Conf: {result['gate_prob_nuisance']:.2f}")

        # Draw text (basic)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        y_offset = 10
        for line in info_lines:
            draw.text((10, y_offset), line, fill=(255, 255, 0), font=font)
            y_offset += 22

        vis = np.array(vis_pil)

        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            Image.fromarray(vis).save(save_path)

        # Show
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.imshow(vis)
                plt.axis('off')
                plt.show()
            except:
                pass

        return vis


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Wafer Defect Inference')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, help='Real data directory')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--num_defect_classes', type=int, default=5, help='Number of defect classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='output/inference', help='Output directory')
    parser.add_argument('--colormap', type=str, default='jet', help='Heatmap colormap')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Prepare data
    if args.synthetic:
        print(f"[Inference] Generating synthetic data...")
        generate_synthetic_dataset(
            output_dir=args.output_dir + '/synthetic',
            num_samples=args.num_samples,
            num_defect_classes=args.num_defect_classes,
        )
        train_loader, val_loader = create_dataloaders(
            data_dir=args.output_dir + '/synthetic',
            batch_size=8,
            synthetic=True,
        )
        class_names = [f'Class_{i}' for i in range(args.num_defect_classes)]
        class_names.insert(0, 'Nuisance')
        val_loader_iter = iter(val_loader)
    else:
        if args.data_dir is None:
            raise ValueError("--data_dir required for real data")
        train_loader, val_loader = create_real_dataloaders(
            data_dir=args.data_dir,
            batch_size=8,
        )
        class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
        val_loader_iter = iter(val_loader)

    print(f"[Inference] Classes: {class_names}")

    # Load model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Inference] Loading model from {args.checkpoint}")
        model = WaferDefectModelSimple(num_defect_classes=len(class_names) - 1) if args.synthetic else None
        # For simplicity, use synthetic model
        model = WaferDefectModelSimple(num_defect_classes=len(class_names) - 1)
    else:
        print(f"[Inference] No checkpoint, using untrained model for demo")
        model = WaferDefectModelSimple(num_defect_classes=len(class_names) - 1)

    # Create inferencer
    inferencer = WaferDefectInferencer(
        model=model,
        device=device,
        class_names=class_names[1:],  # Exclude Nuisance
        use_synthetic=args.synthetic,
    )

    # Run inference
    print(f"[Inference] Running inference...")
    for i, batch in enumerate(val_loader_iter):
        if i >= args.num_samples // 8:
            break

        images = batch['images']
        results = inferencer.predict_batch({'images': images})

        for j, result in enumerate(results):
            print(f"  Sample {i*8+j}: is_defect={result['is_defect']}, "
                  f"type={result['defect_type']}, conf={result['confidence']:.2f}")

            # Save visualization
            vis = inferencer.visualize(
                images[j],
                result=result,
                save_path=os.path.join(args.output_dir, f'sample_{i*8+j}.png'),
            )

    print(f"[Inference] Done. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
