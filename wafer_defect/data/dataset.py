"""
Wafer Defect Dataset with 3-view fusion and synthetic data generation.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class WaferDefectSample:
    """Represents a single wafer defect sample with 3 views."""

    def __init__(
        self,
        images: List[np.ndarray],  # 3 views
        label: int,  # 0=Nuisance, 1~K defect class
        is_defect: int,  # 0=nuisance, 1=defect
        defect_type: Optional[int] = None,  # specific defect class if is_defect=1
        meta: Optional[Dict] = None
    ):
        self.images = images  # list of 3 images
        self.label = label
        self.is_defect = is_defect
        self.defect_type = defect_type
        self.meta = meta or {}


class SyntheticWaferGenerator:
    """Generate synthetic wafer SEM-like images for code verification."""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.footer_height = int(img_size * 0.15)  # bottom 15% for scale bar

    def generate_texture(self, seed: int) -> np.ndarray:
        """Generate base wafer texture pattern."""
        np.random.seed(seed)
        img = np.random.randn(self.img_size, self.img_size).astype(np.float32)

        # Add low-frequency background variation
        x = np.linspace(-1, 1, self.img_size)
        y = np.linspace(-1, 1, self.img_size)
        X, Y = np.meshgrid(x, y)
        background = np.exp(-(X**2 + Y**2) / 0.5) * 0.3
        img += background

        # Add fine grain pattern (wafer texture)
        noise = np.random.randn(self.img_size, self.img_size) * 0.1
        img += noise

        return img

    def add_defect_pattern(self, img: np.ndarray, defect_type: int, seed: int) -> np.ndarray:
        """Add different defect patterns based on defect type."""
        np.random.seed(seed)

        h, w = img.shape
        main_h = h - self.footer_height

        if defect_type == 0:
            # Nuisance - clean wafer pattern
            pass
        elif defect_type == 1:
            # Scratch defect
            num_scratches = random.randint(1, 3)
            for _ in range(num_scratches):
                x1 = random.randint(0, w)
                y1 = random.randint(0, main_h)
                x2 = random.randint(0, w)
                y2 = random.randint(0, main_h)
                cv2.line(img[:main_h], (x1, y1), (x2, y2), 1.5, 2)
        elif defect_type == 2:
            # Particle defect
            num_particles = random.randint(5, 15)
            for _ in range(num_particles):
                x = random.randint(0, w)
                y = random.randint(0, main_h)
                r = random.randint(2, 8)
                cv2.circle(img[:main_h], (x, y), r, -1)
        elif defect_type == 3:
            # Crack defect
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, main_h)
            for _ in range(random.randint(5, 15)):
                dx = random.randint(-10, 10)
                dy = random.randint(-5, 5)
                x1 = max(0, min(w, x1 + dx))
                y1 = max(0, min(main_h, y1 + dy))
                cv2.circle(img[:main_h], (x1, y1), 1, -1)
        else:
            # Generic local anomaly
            num_spots = random.randint(1, 5)
            for _ in range(num_spots):
                x = random.randint(0, w)
                y = random.randint(0, main_h)
                r = random.randint(3, 12)
                cv2.circle(img[:main_h], (x, y), r, 0.8, -1)

        return img

    def add_scale_bar(self, img: np.ndarray) -> np.ndarray:
        """Add scale bar at the bottom of the image."""
        h, w = img.shape
        footer = img[h - self.footer_height:]

        # Add a simple scale bar line
        cv2.line(img, (10, h - self.footer_height // 2),
                 (50, h - self.footer_height // 2), 0.5, 2)

        return img

    def generate(
        self,
        defect_type: int,
        view_seed: int
    ) -> np.ndarray:
        """Generate a single-view image."""
        # Base texture
        img = self.generate_texture(view_seed)

        # Add defect pattern
        img = self.add_defect_pattern(img, defect_type, view_seed + 1000)

        # Add scale bar
        img = self.add_scale_bar(img)

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert to 3-channel by repeating
        img = np.stack([img] * 3, axis=-1)

        return img

    def generate_sample(
        self,
        label: int,
        is_defect: int,
        defect_type: int,
        base_seed: int
    ) -> WaferDefectSample:
        """Generate a 3-view sample."""
        images = []
        for v in range(3):
            view_seed = base_seed + v * 10000
            img = self.generate(defect_type, view_seed)
            images.append(img)

        return WaferDefectSample(
            images=images,
            label=label,
            is_defect=is_defect,
            defect_type=defect_type if is_defect else None
        )


class WaferDefectDataset(Dataset):
    """
    Wafer defect dataset for 3-view classification.

    Each sample contains:
    - 3 views of the same defect
    - label: 0=Nuisance, 1~K=defect class
    - is_defect: 0 or 1
    """

    def __init__(
        self,
        samples: List[WaferDefectSample],
        transform=None,
        crop_footer: bool = True,
        footer_ratio: float = 0.85
    ):
        self.samples = samples
        self.transform = transform
        self.crop_footer = crop_footer
        self.footer_ratio = footer_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Process each view
        views = []
        for img in sample.images:
            # img is already [H, W, 3] in [0, 1]

            # Crop footer region (scale bar area)
            if self.crop_footer:
                h = img.shape[0]
                main = img[:int(h * self.footer_ratio)]
            else:
                main = img

            # Convert to tensor [C, H, W]
            main = torch.from_numpy(main).permute(2, 0, 1).float()

            # Apply transforms if provided
            if self.transform:
                main = self.transform(main)

            views.append(main)

        # Stack views: [3, C, H, W]
        views = torch.stack(views)

        return {
            "images": views,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "is_defect": torch.tensor(sample.is_defect, dtype=torch.long),
            "defect_type": torch.tensor(
                sample.defect_type if sample.defect_type is not None else -1,
                dtype=torch.long
            )
        }


def generate_synthetic_dataset(
    num_samples: int = 100,
    num_defect_classes: int = 10,
    nuisance_ratio: float = 0.3,
    img_size: int = 224,
    seed: int = 42
) -> Tuple[List[WaferDefectSample], List[WaferDefectSample]]:
    """
    Generate synthetic wafer defect dataset for verification.

    Returns:
        train_samples, val_samples
    """
    random.seed(seed)
    np.random.seed(seed)

    generator = SyntheticWaferGenerator(img_size=img_size)

    # Calculate class distribution
    num_nuisance = int(num_samples * nuisance_ratio)
    num_defects = num_samples - num_nuisance

    # Generate nuisance samples
    nuisance_samples = []
    for i in range(num_nuisance):
        sample = generator.generate_sample(
            label=0,
            is_defect=0,
            defect_type=0,
            base_seed=seed + i
        )
        nuisance_samples.append(sample)

    # Generate defect samples (evenly distributed across classes)
    defect_samples = []
    samples_per_class = num_defects // num_defect_classes

    for defect_id in range(1, num_defect_classes + 1):
        for j in range(samples_per_class):
            sample = generator.generate_sample(
                label=defect_id,
                is_defect=1,
                defect_type=defect_id,
                base_seed=seed + 10000 + (defect_id - 1) * 1000 + j
            )
            defect_samples.append(sample)

    # Shuffle and split
    all_samples = nuisance_samples + defect_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    return train_samples, val_samples


def create_dataloaders(
    train_samples: List[WaferDefectSample],
    val_samples: List[WaferDefectSample],
    batch_size: int = 8,
    num_workers: int = 4,
    crop_footer: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""

    train_dataset = WaferDefectDataset(
        samples=train_samples,
        transform=None,  # Add transforms if needed
        crop_footer=crop_footer
    )

    val_dataset = WaferDefectDataset(
        samples=val_samples,
        transform=None,
        crop_footer=crop_footer
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
