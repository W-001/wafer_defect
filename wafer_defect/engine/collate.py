"""
Collate functions for multi-view and paired data processing.

Provides flexible collate functions for:
1. MultiViewCollateFn: Handles 3-view fusion with optional padding
2. PairedCollateFn: For paired samples (normal vs defective)
3. DynamicPaddingCollate: Variable-size inputs with padding

Used when DataLoader needs custom batch assembly logic.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data.dataloader import default_collate


def stack_views(views: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack multiple views into a single tensor.

    Args:
        views: List of [C, H, W] tensors

    Returns:
        [num_views, C, H, W] tensor
    """
    return torch.stack(views, dim=0)


def pad_to_max_size(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
    size: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Pad tensors to max size in batch or fixed size.

    Args:
        tensors: List of [C, H, W] tensors
        pad_value: Value for padding
        size: Optional fixed (H, W) size

    Returns:
        Tuple of (padded tensor [B, C, H, W], original sizes list)
    """
    if size is not None:
        target_h, target_w = size
    else:
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        target_h, target_w = max_h, max_w

    padded = []
    original_sizes = []

    for t in tensors:
        original_sizes.append((t.shape[1], t.shape[2]))
        c, h, w = t.shape

        if h == target_h and w == target_w:
            padded.append(t)
        else:
            # Pad to target size
            pad_h = target_h - h
            pad_w = target_w - w
            padded_t = F.pad(t, (0, pad_w, 0, pad_h), value=pad_value)
            padded.append(padded_t)

    return torch.stack(padded, dim=0), original_sizes


class MultiViewCollateFn:
    """
    Collate function for multi-view (3-view) wafer defect data.

    Handles:
    - Stacking multiple views into [B, num_views, C, H, W]
    - Optional dynamic padding for variable image sizes
    - Metadata preservation (labels, paths, etc.)

    Args:
        pad_to_size: Optional (H, W) to pad all images to uniform size
        pad_value: Padding value (default: 0.0)
        num_views: Expected number of views (default: 3)
    """

    def __init__(
        self,
        pad_to_size: Optional[Tuple[int, int]] = None,
        pad_value: float = 0.0,
        num_views: int = 3,
    ):
        self.pad_to_size = pad_to_size
        self.pad_value = pad_value
        self.num_views = num_views

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Expected input format (from dataset):
            {
                "images": torch.Tensor [num_views, C, H, W] or [C, H, W] (single-view)
                "label": int,
                "is_defect": int,
                "defect_type": int,
                ...
            }

        Returns:
            Collated batch:
            {
                "images": torch.Tensor [B, num_views, C, H, W] or [B, C, H, W]
                "labels": torch.Tensor [B]
                ...
            }
        """
        if len(batch) == 0:
            return {}

        # Separate images from other fields
        images_list = []
        labels_list = []
        is_defect_list = []
        defect_type_list = []
        meta_list = []

        for sample in batch:
            # Handle both 3-view [V, C, H, W] and single-view [C, H, W]
            images = sample["images"]
            if images.dim() == 4:
                # 3-view: [V, C, H, W]
                images_list.append(images)
            elif images.dim() == 3:
                # Single-view: [C, H, W] -> [1, C, H, W]
                images_list.append(images.unsqueeze(0))
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")

            labels_list.append(sample["label"])
            is_defect_list.append(sample["is_defect"])
            defect_type_list.append(sample["defect_type"])

            # Collect any additional metadata
            meta = {k: v for k, v in sample.items()
                    if k not in ["images", "label", "is_defect", "defect_type"]}
            if meta:
                meta_list.append(meta)

        # Stack images: [B, V, C, H, W]
        images_batch = torch.stack(images_list, dim=0)

        # Apply padding if specified
        if self.pad_to_size is not None:
            images_batch, original_sizes = self._pad_images(images_batch)

        # Stack labels
        labels_batch = torch.stack(labels_list, dim=0)
        is_defect_batch = torch.stack(is_defect_list, dim=0)
        defect_type_batch = torch.stack(defect_type_list, dim=0)

        # Build output dict
        result = {
            "images": images_batch,
            "label": labels_batch,
            "is_defect": is_defect_batch,
            "defect_type": defect_type_batch,
        }

        # Add metadata if present
        if meta_list:
            result["_meta"] = meta_list

        return result

    def _pad_images(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Pad images to uniform size.

        Args:
            images: [B, V, C, H, W] tensor

        Returns:
            Tuple of (padded [B, V, C, H, W], original sizes list)
        """
        B, V, C, H, W = images.shape
        target_h, target_w = self.pad_to_size

        if H == target_h and W == target_w:
            return images, [(H, W)] * B

        # Reshape to [B*V, C, H, W] for padding
        images_flat = images.view(B * V, C, H, W)

        # Pad
        pad_h = target_h - H
        pad_w = target_w - W
        padded_flat = F.pad(
            images_flat,
            (0, pad_w, 0, pad_h),
            value=self.pad_value
        )

        # Reshape back
        padded = padded_flat.view(B, V, C, target_h, target_w)
        original_sizes = [(H, W)] * B

        return padded, original_sizes


class PairedCollateFn:
    """
    Collate function for paired samples (e.g., normal + defective).

    Creates pairs of samples for contrastive learning or comparison.

    Args:
        view_collate: MultiViewCollateFn for handling individual samples
    """

    def __init__(self, view_collate: Optional[MultiViewCollateFn] = None):
        self.view_collate = view_collate or MultiViewCollateFn()

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate paired samples.

        Expected input format:
            {
                "view1": {...},  # First view/sample
                "view2": {...},  # Second view/sample
                "is_pair": bool,  # Whether this is a positive pair
                "pair_label": int,  # Class label for the pair
            }

        Or simplified:
            {
                "images": [tensor1, tensor2],  # List of two views
                "label": int,
                "is_pair": bool,
            }
        """
        if len(batch) == 0:
            return {}

        # Check format
        first = batch[0]

        if "view1" in first and "view2" in first:
            # Full paired format
            return self._collate_full_format(batch)
        elif isinstance(first.get("images"), list):
            # Simplified list format
            return self._collate_list_format(batch)
        else:
            # Fall back to standard collate
            return self.view_collate(batch)

    def _collate_full_format(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collates full paired format."""
        view1_batch = self.view_collate([s["view1"] for s in batch])
        view2_batch = self.view_collate([s["view2"] for s in batch])

        result = {
            "view1": view1_batch,
            "view2": view2_batch,
        }

        # Add pair metadata
        if "is_pair" in batch[0]:
            result["is_pair"] = default_collate([s["is_pair"] for s in batch])
        if "pair_label" in batch[0]:
            result["pair_label"] = default_collate([s["pair_label"] for s in batch])

        return result

    def _collate_list_format(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collates simplified list format."""
        # Separate views
        view1_images = [s["images"][0] for s in batch]
        view2_images = [s["images"][1] for s in batch]

        # Stack each view separately
        view1_stacked = torch.stack(view1_images, dim=0)
        view2_stacked = torch.stack(view2_images, dim=0)

        result = {
            "images": torch.stack([view1_stacked, view2_stacked], dim=0),
        }

        # Add labels
        if "label" in batch[0]:
            result["label"] = default_collate([s["label"] for s in batch])
        if "is_pair" in batch[0]:
            result["is_pair"] = default_collate([s["is_pair"] for s in batch])

        return result


class DynamicPaddingCollate:
    """
    Collate function with dynamic padding to max size in batch.

    Useful for datasets with variable image sizes.

    Args:
        pad_value: Value for padding
        min_size: Minimum size to pad to
        max_size: Maximum size to pad to (None = no limit)
    """

    def __init__(
        self,
        pad_value: float = 0.0,
        min_size: Tuple[int, int] = (224, 224),
        max_size: Optional[Tuple[int, int]] = None,
    ):
        self.pad_value = pad_value
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate with dynamic padding."""
        if len(batch) == 0:
            return {}

        # Find max size in batch
        max_h = max(s["images"].shape[-2] for s in batch)
        max_w = max(s["images"].shape[-1] for s in batch)

        # Apply min_size constraint
        target_h = max(max_h, self.min_size[0])
        target_w = max(max_w, self.min_size[1])

        # Apply max_size constraint
        if self.max_size is not None:
            target_h = min(target_h, self.max_size[0])
            target_w = min(target_w, self.max_size[1])

        # Pad and collate
        padded_images = []
        for s in batch:
            img = s["images"]
            h, w = img.shape[-2:]

            if h < target_h or w < target_w:
                pad_h = target_h - h
                pad_w = target_w - w
                img = F.pad(img, (0, pad_w, 0, pad_h), value=self.pad_value)

            padded_images.append(img)

        # Build result
        result = {
            "images": torch.stack(padded_images, dim=0),
        }

        # Add other fields
        for key in batch[0]:
            if key != "images":
                if isinstance(batch[0][key], torch.Tensor):
                    result[key] = torch.stack([s[key] for s in batch], dim=0)
                else:
                    result[key] = [s[key] for s in batch]

        return result


def create_collate_fn(
    mode: str = "default",
    **kwargs
) -> callable:
    """
    Factory function to create collate functions.

    Args:
        mode: Collate mode ("default", "multi_view", "paired", "dynamic_padding")
        **kwargs: Additional arguments for the collate function

    Returns:
        Collate function
    """
    if mode == "default":
        return default_collate
    elif mode == "multi_view":
        return MultiViewCollateFn(**kwargs)
    elif mode == "paired":
        return PairedCollateFn(**kwargs)
    elif mode == "dynamic_padding":
        return DynamicPaddingCollate(**kwargs)
    else:
        raise ValueError(f"Unknown collate mode: {mode}")
