"""
Wafer Defect Image Preprocessor.

Provides preprocessing pipeline for wafer SEM images:
- Grayscale to RGB conversion (channel copy)
- Bottom 40px crop (scale bar area)
- Resize to 392x392
- DINOv2/ImageNet normalization

The preprocessing is designed to be used with PyTorch transforms or standalone.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


# DINOv2 / ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard preprocessing settings for wafer images
DEFAULT_CROP_BOTTOM = 40
DEFAULT_IMG_SIZE = 392


class WaferPreprocessor:
    """
    Preprocessing pipeline for wafer SEM images.

    Applies the following steps:
    1. Load image and convert grayscale to RGB (copy channel 3 times)
    2. Crop bottom 40px (scale bar area)
    3. Resize to target size (default 392x392)
    4. Normalize with ImageNet/DINOv2 statistics
    """

    def __init__(
        self,
        img_size: int = DEFAULT_IMG_SIZE,
        crop_bottom: int = DEFAULT_CROP_BOTTOM,
        mean: Tuple[float, ...] = tuple(IMAGENET_MEAN),
        std: Tuple[float, ...] = tuple(IMAGENET_STD),
    ):
        """
        Args:
            img_size: Target image size (default 392 for DINOv3-L)
            crop_bottom: Number of pixels to crop from bottom (default 40)
            mean: Normalization mean values
            std: Normalization std values
        """
        self.img_size = img_size
        self.crop_bottom = crop_bottom
        self.mean = mean
        self.std = std

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """
        Load image from file and convert to RGB.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB mode
        """
        img = Image.open(str(path))
        if img.mode != 'RGB':
            # Grayscale or other mode: convert to RGB by copying channels
            img = img.convert('RGB')
        return img

    def crop_bottom_region(self, img: Image.Image) -> Image.Image:
        """
        Crop bottom scale bar area from image.

        Args:
            img: PIL Image

        Returns:
            Cropped PIL Image
        """
        w, h = img.size
        if h > self.crop_bottom:
            img = img.crop((0, 0, w, h - self.crop_bottom))
        return img

    def resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size.

        Args:
            img: PIL Image

        Returns:
            Resized PIL Image
        """
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to PyTorch tensor.

        Args:
            img: PIL Image in RGB mode

        Returns:
            Tensor of shape [3, H, W] with values in [0, 1]
        """
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return tensor

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor with ImageNet/DINOv2 statistics.

        Args:
            tensor: Tensor of shape [C, H, W] with values in [0, 1]

        Returns:
            Normalized tensor
        """
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def preprocess(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Full preprocessing pipeline for a single image.

        Args:
            path: Path to image file

        Returns:
            Preprocessed tensor of shape [3, img_size, img_size]
        """
        img = self.load_image(path)
        img = self.crop_bottom_region(img)
        img = self.resize_image(img)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        return tensor

    def preprocess_image_array(self, img_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess from numpy array (already loaded).

        Args:
            img_array: Numpy array of shape [H, W] or [H, W, C]

        Returns:
            Preprocessed tensor of shape [3, img_size, img_size]
        """
        # Convert to PIL
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        if img_array.ndim == 2:
            # Grayscale: convert to RGB
            img = Image.fromarray(img_array, mode='L').convert('RGB')
        else:
            img = Image.fromarray(img_array)

        img = self.crop_bottom_region(img)
        img = self.resize_image(img)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        return tensor

    def __call__(self, img_or_path: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Convenience method matching transform interface.

        Supports:
        - str/Path: file path to image
        - PIL.Image.Image: directly process the image
        - np.ndarray: numpy array image
        """
        if isinstance(img_or_path, (str, Path)):
            return self.preprocess(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            return self.preprocess_image(img_or_path)
        elif isinstance(img_or_path, np.ndarray):
            return self.preprocess_image_array(img_or_path)
        else:
            raise TypeError(f"Expected str, Path, PIL.Image, or np.ndarray, got {type(img_or_path)}")

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess from PIL Image.

        Args:
            img: PIL Image (RGB or grayscale)

        Returns:
            Preprocessed tensor of shape [3, img_size, img_size]
        """
        # Grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Crop bottom
        img = self.crop_bottom_region(img)

        # Resize
        img = self.resize_image(img)

        # To tensor and normalize
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        return tensor


def get_preprocessor(
    img_size: int = DEFAULT_IMG_SIZE,
    crop_bottom: int = DEFAULT_CROP_BOTTOM,
) -> WaferPreprocessor:
    """
    Factory function to create a WaferPreprocessor with standard settings.

    Args:
        img_size: Target image size
        crop_bottom: Bottom crop pixels

    Returns:
        Configured WaferPreprocessor instance
    """
    return WaferPreprocessor(img_size=img_size, crop_bottom=crop_bottom)


class ToTensor:
    """Convert PIL Image to normalized tensor."""

    def __init__(
        self,
        crop_bottom: int = DEFAULT_CROP_BOTTOM,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        self.crop_bottom = crop_bottom
        self.img_size = img_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Transform PIL Image to normalized tensor.

        Args:
            img: PIL Image (RGB or grayscale)

        Returns:
            Normalized tensor [3, img_size, img_size]
        """
        # Grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Crop bottom
        w, h = img.size
        if h > self.crop_bottom:
            img = img.crop((0, 0, w, h - self.crop_bottom))

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # To tensor and normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # Normalize with ImageNet stats
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor


class Normalize:
    """Apply ImageNet/DINOv2 normalization to a tensor."""

    def __init__(
        self,
        mean: Tuple[float, ...] = tuple(IMAGENET_MEAN),
        std: Tuple[float, ...] = tuple(IMAGENET_STD),
    ):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


# Convenience: default preprocessor instance
default_preprocessor = WaferPreprocessor()
