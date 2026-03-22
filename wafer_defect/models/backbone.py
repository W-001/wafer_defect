"""
DINOv3 Backbone wrapper for wafer defect classification.
"""

import torch
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone wrapper that extracts features from images.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained_path: str = None,
        freeze_backbone: bool = False,
        embed_dim: int = 768
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim

        # Load DINOv3 model from local weights
        if pretrained_path and pretrained_path.endswith('.pth'):
            self.model = self._load_dinov3_model(model_name, pretrained_path)
        else:
            raise ValueError(f"Invalid pretrained path: {pretrained_path}")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_dinov3_model(self, model_name: str, weights_path: str):
        """Load DINOv3 model from local weights."""
        import sys
        import os
        sys.path.insert(0, 'C:/Code/Work/DefectClass_dinov3/dinov3')

        # Convert to absolute path and ensure it exists
        weights_path = os.path.abspath(weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Load state dict directly
        state_dict = torch.load(weights_path, map_location="cpu")

        # Create model with exact architecture to match weights
        if "vitl16" in model_name:
            from dinov3.hub.backbones import dinov3_vitl16
            # Use the hub function to create the model with correct architecture
            model = dinov3_vitl16(pretrained=False)
        elif "vitb16" in model_name:
            from dinov3.hub.backbones import dinov3_vitb16
            model = dinov3_vitb16(pretrained=False)
        elif "vits16" in model_name:
            from dinov3.hub.backbones import dinov3_vits16
            model = dinov3_vits16(pretrained=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Load the full state dict
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            features: [B, D] flattened features
        """
        # DINOv3 returns dict with 'x_norm_patchtokens' and other features
        output = self.model(x)

        # Extract patch tokens and average them
        if isinstance(output, dict):
            # Get patch tokens [B, N, D]
            patch_tokens = output.get('x_norm_patchtokens', output.get('x_prenorm', None))
            if patch_tokens is None:
                # Try alternate key names
                for key in ['x_prenorm', 'x_norm', 'features']:
                    if key in output:
                        patch_tokens = output[key]
                        break

            if patch_tokens is not None:
                # Average pool patch tokens -> [B, D]
                feat = patch_tokens.mean(dim=1)
            else:
                # Fallback: use CLS token or raw output
                feat = output.get('x_norm_clstoken', output.get('cls_token', None))
                if feat is None:
                    feat = list(output.values())[0]
                    if len(feat.shape) == 3:
                        feat = feat.mean(dim=1)
        else:
            # Direct tensor output
            feat = output
            if len(feat.shape) == 3:  # [B, N, D]
                feat = feat.mean(dim=1)  # [B, D]

        return feat

    def get_output_dim(self) -> int:
        """Return the feature dimension."""
        return self.embed_dim
