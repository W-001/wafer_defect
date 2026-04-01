"""
Dataset inspection utility.
Run this before training to understand your data distribution.

Usage:
    python -c "from wafer_defect.utils.data_inspector import inspect; inspect('/path/to/data')"
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import re


def inspect(data_dir: str, crop_bottom: int = 40, nuisance_name: str = "Nuisance"):
    """
    Inspect dataset and print statistics.

    Args:
        data_dir: Path to data directory
        crop_bottom: Bottom crop pixels (for reporting)
        nuisance_name: Name of nuisance/normal class folder
    """
    print("=" * 60)
    print("Dataset Inspection")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Nuisance class name: {nuisance_name}")
    print(f"Bottom crop: {crop_bottom}px")
    print()

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return

    # Scan folders
    class_dirs = [d for d in data_path.iterdir()
                  if d.is_dir() and not d.name.startswith('.')]
    class_dirs.sort(key=lambda x: x.name)

    print(f"Found {len(class_dirs)} classes:")
    print("-" * 40)

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    class_stats = []
    total_samples = 0
    total_images = 0
    incomplete_groups = []

    for class_dir in class_dirs:
        img_files = [f for f in class_dir.iterdir()
                     if f.suffix.lower() in IMG_EXTENSIONS]

        # Group by defect (using IxxK pattern)
        groups = _group_three_views(img_files)

        is_nuisance = class_dir.name == nuisance_name
        class_type = "Nuisance" if is_nuisance else "Defect"

        class_stats.append({
            'name': class_dir.name,
            'type': class_type,
            'num_groups': len(groups),
            'num_images': len(img_files)
        })

        total_samples += len(groups)
        total_images += len(img_files)

        # Check for incomplete groups
        for group in groups:
            if len(group) < 3:
                incomplete_groups.append({
                    'class': class_dir.name,
                    'file': group[0].name if group else 'unknown',
                    'views': len(group)
                })

        # Also detect 2-view patterns that were skipped
        for base, views_dict in _get_base_patterns(img_files).items():
            if len(views_dict) == 2:
                incomplete_groups.append({
                    'class': class_dir.name,
                    'file': list(views_dict.values())[0].name,
                    'views': 2,
                    'note': '2-view sample ignored (needs 3 views)'
                })

        print(f"  [{class_type:8}] {class_dir.name}: {len(groups)} samples ({len(img_files)} images)")

    print()
    print(f"Total: {total_samples} samples, {total_images} images")
    print("-" * 40)

    # Check naming pattern
    sample_files = []
    for d in class_dirs:
        sample_files.extend([f for f in d.iterdir()
                           if f.suffix.lower() in IMG_EXTENSIONS][:5])

    if sample_files:
        print("\nSample file names:")
        for f in sample_files[:5]:
            print(f"  {f.name}")

        # Check IxxK pattern
        i_pattern_count = sum(1 for f in sample_files
                             if re.search(r'I\d{2}K', f.name))
        print(f"\nNaming pattern check:")
        print(f"  Files with 'IxxK' pattern: {i_pattern_count}/{len(sample_files)}")

    # Report issues
    print()
    print("=" * 60)
    print("Potential Issues")
    print("=" * 60)

    issues = []

    # Check for incomplete groups
    if incomplete_groups:
        issues.append(f"⚠️  {len(incomplete_groups)} incomplete 3-view groups (will be duplicated)")

    # Check for very small classes
    min_samples = min((s['num_groups'] for s in class_stats), default=0)
    if min_samples < 5:
        issues.append(f"⚠️  Some classes have very few samples (min: {min_samples})")

    # Check class imbalance
    if class_stats:
        sample_counts = [s['num_groups'] for s in class_stats]
        max_count = max(sample_counts)
        min_count = min(sample_counts)
        if max_count > min_count * 5:
            issues.append(f"⚠️  Class imbalance detected (max: {max_count}, min: {min_count})")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No obvious issues found")

    # Summary report
    report = {
        'data_dir': str(data_path),
        'nuisance_name': nuisance_name,
        'total_samples': total_samples,
        'total_images': total_images,
        'num_classes': len(class_dirs),
        'classes': class_stats,
        'issues': issues
    }

    return report


def _get_base_patterns(img_files):
    """Extract base patterns and view indices from image files."""
    base_patterns = {}
    for f in img_files:
        name = f.stem
        match = re.match(r'(.*I)(\d{2})(K.*)', name)
        if match:
            base = match.group(1) + match.group(3)
            view_idx = int(match.group(2))
        else:
            base = name
            view_idx = 0
        if base not in base_patterns:
            base_patterns[base] = {}
        base_patterns[base][view_idx] = f
    return base_patterns


def _group_three_views(img_files):
    """Group images by defect (same logic as RealWaferDataset)."""
    base_patterns = {}

    for f in img_files:
        name = f.stem

        match = re.match(r'(.*I)(\d{2})(K.*)', name)
        if match:
            base = match.group(1) + match.group(3)
            view_idx = int(match.group(2))
        else:
            base = name
            view_idx = 0

        if base not in base_patterns:
            base_patterns[base] = {}
        base_patterns[base][view_idx] = f

    groups = []
    for base, views_dict in base_patterns.items():
        if len(views_dict) == 3 and set(views_dict.keys()) == {0, 1, 2}:
            groups.append([views_dict[0], views_dict[1], views_dict[2]])
        elif len(views_dict) == 1:
            # Single view: duplicate to 3 views (same as RealWaferDataset)
            groups.append(list(views_dict.values()) * 3)
        # 2-view or other incomplete groups are ignored (consistent with RealWaferDataset)

    return groups


def save_report(data_dir: str, output_file: str = "dataset_report.json"):
    """Save inspection report to JSON file."""
    report = inspect(data_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output_file}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect wafer defect dataset")
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--nuisance", default="Nuisance", help="Nuisance class folder name")
    parser.add_argument("--crop", type=int, default=40, help="Bottom crop pixels")
    parser.add_argument("--save", help="Save report to JSON file")

    args = parser.parse_args()

    if args.save:
        save_report(args.data_dir, args.save)
    else:
        inspect(args.data_dir, args.crop, args.nuisance)
