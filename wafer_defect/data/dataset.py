"""
Wafer Defect Dataset with 3-view fusion.
Supports both synthetic data generation and real folder-based data loading.
"""

import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class WaferDefectSample:
    """Represents a single wafer defect sample with 3 views."""

    def __init__(
        self,
        images: List[np.ndarray],  # 3 views
        label: int,  # 0=Nuisance, 1~K defect class
        is_defect: int,  # 0=nuisance, 1=defect
        defect_type: Optional[int] = None,
        meta: Optional[Dict] = None
    ):
        self.images = images
        self.label = label
        self.is_defect = is_defect
        self.defect_type = defect_type
        self.meta = meta or {}


class RealWaferDataset(Dataset):
    """
    Load real wafer SEM images from folder structure.

    Expected structure:
        root/
        ├── Nuisance/           # 正常样本
        │   ├── img001_1.jpg
        │   ├── img001_2.jpg
        │   ├── img001_3.jpg
        │   ├── img002_1.jpg
        │   └── ...
        ├── Scratch/            # 划痕缺陷
        │   ├── img101_1.jpg
        │   └── ...
        └── ...

    Each sample: 3 views (命名: xxx_1.jpg, xxx_2.jpg, xxx_3.jpg)
    Bottom 40px is scale bar area and will be cropped.
    Images are resized to a fixed size for batch processing.
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        crop_bottom: int = 40,
        transform=None,
        nuisance_name: str = "Nuisance",
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            root_dir: 数据根目录
            img_size: 统一resize到的尺寸
            crop_bottom: 底部裁剪高度 (默认40px)
            transform: 可选的额外transform
            nuisance_name: 正常/无缺陷类别的文件夹名称
            label_map: 类别名到label id的映射，不提供则自动生成
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.crop_bottom = crop_bottom
        self.transform = transform
        self.nuisance_name = nuisance_name

        self.samples = []  # List of (img_paths, label, is_defect)
        self.class_info = {}

        self._scan_folders(nuisance_name, label_map)
        self._scan_three_views()

    def _scan_folders(self, nuisance_name: str, label_map: Optional[Dict[str, int]]):
        """扫描根目录下的所有类别文件夹"""
        class_dirs = [d for d in self.root_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')]
        # Do NOT sort alphabetically — label assignment must follow filesystem order
        # so that the user's folder arrangement determines the label sequence.
        # The nuisance_name check ensures Nuisance is always label=0 regardless of
        # its position in the list; defect folders get labels 1,2,3... in iteration order.
        # For deterministic ordering, pass an explicit label_map.

        if label_map is None:
            self.label_map = {}
            current_label = 1  # 从1开始，0留给Nuisance
            for d in class_dirs:
                if d.name == nuisance_name:
                    self.label_map[d.name] = 0
                else:
                    self.label_map[d.name] = current_label
                    current_label += 1
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

        for class_name, label_id in self.label_map.items():
            is_defect = 0 if class_name == nuisance_name else 1
            self.class_info[label_id] = {
                'name': class_name,
                'is_defect': is_defect
            }

    def _scan_three_views(self):
        """扫描每个类别的图片，识别三视角组合"""
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            if class_dir.name not in self.label_map:
                continue

            label = self.label_map[class_dir.name]
            is_defect = 0 if class_dir.name == self.nuisance_name else 1

            img_files = [f for f in class_dir.iterdir()
                         if f.suffix.lower() in self.IMG_EXTENSIONS]
            img_files.sort(key=lambda x: x.name)

            view_groups = self._group_three_views(img_files)

            for group in view_groups:
                self.samples.append({
                    'paths': group,
                    'label': label,
                    'is_defect': is_defect
                })

    def _group_three_views(self, img_files: List[Path]) -> List[List[Path]]:
        """
        将图片分组为三视角组合。

        命名格式: D234569@123456W1234567890F12345678I00K12345678
        K前面的数字(00/01/02)表示同一defect的第几张图

        Returns:
            List of [path1, path2, path3] groups
        """
        import re

        groups = []
        base_patterns = {}

        for f in img_files:
            name = f.stem

            # 模式: Dxxxxx...IxxKxxxxxx (K前面2位数字表示视角序号)
            match = re.match(r'(.*I)(\d{2})(K.*)', name)
            if match:
                base = match.group(1) + match.group(3)  # Dxxxxx...I + Kxxxxxx
                view_idx = int(match.group(2))  # 00, 01, 02
            else:
                # Fallback: 单张图片
                base = name
                view_idx = 0

            if base not in base_patterns:
                base_patterns[base] = {}
            base_patterns[base][view_idx] = f

        for base, views_dict in base_patterns.items():
            if len(views_dict) == 3 and set(views_dict.keys()) == {0, 1, 2}:
                # 完整的3视角，按顺序排列
                groups.append([views_dict[0], views_dict[1], views_dict[2]])
            elif len(views_dict) == 1:
                # 单张图片，复制为三视角
                single_path = list(views_dict.values())[0]
                groups.append([single_path, single_path, single_path])

        return groups

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_preprocess(self, img_path: Path) -> np.ndarray:
        """加载并预处理单张图片"""
        img = Image.open(str(img_path)).convert('RGB')

        # 裁剪底部尺度区域
        if self.crop_bottom > 0:
            w, h = img.size
            if h > self.crop_bottom:
                img = img.crop((0, 0, w, h - self.crop_bottom))

        # resize到统一尺寸
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 归一化到 [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0

        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        views = []
        for path in sample['paths']:
            img = self._load_and_preprocess(path)
            img = torch.from_numpy(img).permute(2, 0, 1).float()

            if self.transform:
                img = self.transform(img)
            views.append(img)

        views = torch.stack(views)

        label = sample['label']
        is_defect = sample['is_defect']
        defect_type = label if is_defect else -1

        return {
            "images": views,
            "label": torch.tensor(label, dtype=torch.long),
            "is_defect": torch.tensor(is_defect, dtype=torch.long),
            "defect_type": torch.tensor(defect_type, dtype=torch.long)
        }

    def get_sample(self, idx: int) -> Dict:
        """Get sample info by index (for debugging)."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        return self.samples[idx]

    def get_class_names(self) -> List[str]:
        """返回类别名称列表 (按label排序)"""
        return [self.class_info[i]['name'] for i in sorted(self.class_info.keys())]


class SyntheticWaferGenerator:
    """Generate synthetic wafer SEM-like images for code verification."""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.footer_height = int(img_size * 0.15)

    def generate_texture(self, seed: int) -> np.ndarray:
        """Generate base wafer texture pattern."""
        np.random.seed(seed)
        img = np.random.randn(self.img_size, self.img_size).astype(np.float32)

        x = np.linspace(-1, 1, self.img_size)
        y = np.linspace(-1, 1, self.img_size)
        X, Y = np.meshgrid(x, y)
        background = np.exp(-(X**2 + Y**2) / 0.5) * 0.3
        img += background

        noise = np.random.randn(self.img_size, self.img_size) * 0.1
        img += noise

        return img

    def add_defect_pattern(self, img: np.ndarray, defect_type: int, seed: int) -> np.ndarray:
        """Add different defect patterns based on defect type."""
        np.random.seed(seed)

        h, w = img.shape
        main_h = h - self.footer_height

        if defect_type == 0:
            pass  # Nuisance - clean
        elif defect_type == 1:
            num_scratches = random.randint(1, 3)
            pil_img = Image.fromarray((img[:main_h] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            for _ in range(num_scratches):
                x1, y1 = random.randint(0, w - 1), random.randint(0, main_h - 1)
                x2, y2 = random.randint(0, w - 1), random.randint(0, main_h - 1)
                draw.line([(x1, y1), (x2, y2)], fill=round(1.5 * 255), width=2)
            img[:main_h] = np.array(pil_img, dtype=np.float32) / 255.0
        elif defect_type == 2:
            num_particles = random.randint(5, 15)
            pil_img = Image.fromarray((img[:main_h] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            for _ in range(num_particles):
                x, y = random.randint(0, w - 1), random.randint(0, main_h - 1)
                r = random.randint(2, 8)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            img[:main_h] = np.array(pil_img, dtype=np.float32) / 255.0
        elif defect_type == 3:
            x1, y1 = random.randint(0, w // 2), random.randint(0, main_h - 1)
            pil_img = Image.fromarray((img[:main_h] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            for _ in range(random.randint(5, 15)):
                dx, dy = random.randint(-10, 10), random.randint(-5, 5)
                x1 = max(0, min(w - 1, x1 + dx))
                y1 = max(0, min(main_h - 1, y1 + dy))
                draw.point([(x1, y1)], fill=255)
            img[:main_h] = np.array(pil_img, dtype=np.float32) / 255.0
        else:
            num_spots = random.randint(1, 5)
            pil_img = Image.fromarray((img[:main_h] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            for _ in range(num_spots):
                x, y = random.randint(0, w - 1), random.randint(0, main_h - 1)
                r = random.randint(3, 12)
                fill = round(0.8 * 255)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            img[:main_h] = np.array(pil_img, dtype=np.float32) / 255.0

        return img

    def add_scale_bar(self, img: np.ndarray) -> np.ndarray:
        """Add scale bar at the bottom."""
        h = img.shape[0]
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        draw.line([(10, h - self.footer_height // 2), (50, h - self.footer_height // 2)],
                  fill=round(0.5 * 255), width=2)
        img = np.array(pil_img, dtype=np.float32) / 255.0
        return img

    def generate(self, defect_type: int, view_seed: int) -> np.ndarray:
        """Generate a single-view image."""
        img = self.generate_texture(view_seed)
        img = self.add_defect_pattern(img, defect_type, view_seed + 1000)
        img = self.add_scale_bar(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.stack([img] * 3, axis=-1)
        return img

    def generate_sample(self, label: int, is_defect: int, defect_type: int,
                        base_seed: int) -> WaferDefectSample:
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
    Wafer defect dataset for 3-view classification (synthetic data).
    """

    def __init__(
        self,
        samples: List[WaferDefectSample],
        transform=None,
        crop_footer: bool = True,
        footer_pixels: int = 40
    ):
        self.samples = samples
        self.transform = transform
        self.crop_footer = crop_footer
        self.footer_pixels = footer_pixels
        self._build_class_info()

    def _build_class_info(self):
        """Infer class_info from the samples (same layout as RealWaferDataset)."""
        label_ids = sorted({s.label for s in self.samples})
        self.class_info = {}
        for lid in label_ids:
            if lid == 0:
                name = "Nuisance"
            else:
                name = f"Defect_{lid}"
            self.class_info[lid] = {"name": name, "is_defect": 0 if lid == 0 else 1}

    def get_class_names(self) -> List[str]:
        """Return class names in label order (index = label id, matches RealWaferDataset)."""
        return [self.class_info[i]["name"] for i in sorted(self.class_info.keys())]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        views = []
        for img in sample.images:
            if self.crop_footer:
                h = img.shape[0]
                main = img[:h - self.footer_pixels] if h > self.footer_pixels else img
            else:
                main = img

            main = torch.from_numpy(main.copy()).permute(2, 0, 1).float()

            if self.transform:
                main = self.transform(main)

            views.append(main)

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
    """Generate synthetic wafer defect dataset for verification."""
    random.seed(seed)
    np.random.seed(seed)

    generator = SyntheticWaferGenerator(img_size=img_size)

    num_nuisance = int(num_samples * nuisance_ratio)
    num_defects = num_samples - num_nuisance

    nuisance_samples = []
    for i in range(num_nuisance):
        sample = generator.generate_sample(
            label=0, is_defect=0, defect_type=0, base_seed=seed + i
        )
        nuisance_samples.append(sample)

    defect_samples = []
    samples_per_class = num_defects // num_defect_classes

    for defect_id in range(1, num_defect_classes + 1):
        for j in range(samples_per_class):
            sample = generator.generate_sample(
                label=defect_id, is_defect=1, defect_type=defect_id,
                base_seed=seed + 10000 + (defect_id - 1) * 1000 + j
            )
            defect_samples.append(sample)

    all_samples = nuisance_samples + defect_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.8)
    return all_samples[:split_idx], all_samples[split_idx:]


def create_dataloaders(
    train_samples: List[WaferDefectSample],
    val_samples: List[WaferDefectSample],
    batch_size: int = 8,
    num_workers: int = 4,
    crop_footer: bool = True,
    footer_pixels: int = 40
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders for synthetic data."""

    train_dataset = WaferDefectDataset(
        samples=train_samples,
        transform=None,
        crop_footer=crop_footer,
        footer_pixels=footer_pixels
    )

    val_dataset = WaferDefectDataset(
        samples=val_samples,
        transform=None,
        crop_footer=crop_footer,
        footer_pixels=footer_pixels
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def create_real_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    crop_bottom: int = 40,
    train_split: float = 0.8,
    nuisance_name: str = "Nuisance"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders from real folder structure.

    Args:
        data_dir: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 统一resize尺寸
        crop_bottom: 底部裁剪像素数
        train_split: 训练集比例
        nuisance_name: 正常类别文件夹名称

    Returns:
        train_loader, val_loader
    """
    full_dataset = RealWaferDataset(
        root_dir=data_dir,
        img_size=img_size,
        crop_bottom=crop_bottom,
        nuisance_name=nuisance_name
    )

    total = len(full_dataset)
    indices = list(range(total))
    random.shuffle(indices)

    split_idx = int(total * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    class _SubsetDataset(Dataset):
        def __init__(self, parent: Dataset, indices: List[int]):
            self.parent = parent
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return self.parent[self.indices[idx]]

    train_loader = torch.utils.data.DataLoader(
        _SubsetDataset(full_dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        _SubsetDataset(full_dataset, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
