"""
Training engine for wafer defect classification.
Includes misclassification tracking for dataset debugging.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from ..losses import CombinedLoss
from ..utils.metrics import GateMetrics, FineMetrics


class MisclassificationTracker:
    """Track misclassified samples for debugging."""

    def __init__(self, class_names=None, num_classes=10):
        self.class_names = class_names or {}
        self.num_classes = num_classes

        # Gate errors (Nuisance vs Defect confusion)
        self.gate_errors = []  # {'path': str, 'true': int, 'pred': int, 'prob': float}

        # Fine errors (Defect type confusion)
        self.fine_errors = []  # {'path': str, 'true': int, 'pred': int, 'probs': list}

        # Confusion tracking
        self.gate_confusion = defaultdict(int)  # (true, pred) -> count
        self.fine_confusion = defaultdict(int)  # (true, pred) -> count

        # Per-class error counts
        self.gate_per_class_errors = defaultdict(int)
        self.fine_per_class_errors = defaultdict(int)

    def add_gate_error(self, path: str, true_label: int, pred_label: int,
                       pred_prob: float, defect_prob: float):
        """Add a gate (Nuisance vs Defect) misclassification."""
        error = {
            'path': path,
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'true_name': self.class_names.get(true_label, f"class_{true_label}"),
            'pred_name': self.class_names.get(pred_label, f"class_{pred_label}"),
            'pred_prob': float(pred_prob),
            'defect_prob': float(defect_prob),
            'error_type': self._get_gate_error_type(true_label, pred_label)
        }
        self.gate_errors.append(error)
        self.gate_confusion[(int(true_label), int(pred_label))] += 1
        self.gate_per_class_errors[int(true_label)] += 1

    def add_fine_error(self, path: str, true_label: int, pred_label: int,
                       all_probs: list):
        """Add a fine (Defect type) misclassification."""
        error = {
            'path': path,
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'true_name': self.class_names.get(true_label, f"class_{true_label}"),
            'pred_name': self.class_names.get(pred_label, f"class_{pred_label}"),
            'probs': [float(p) for p in all_probs],
            'pred_prob': float(all_probs[pred_label] if pred_label < len(all_probs) else 0)
        }
        self.fine_errors.append(error)
        self.fine_confusion[(int(true_label), int(pred_label))] += 1
        self.fine_per_class_errors[int(true_label)] += 1

    def _get_gate_error_type(self, true_label, pred_label):
        """Get human-readable error type."""
        if true_label == 0 and pred_label != 0:
            return "defect_as_nuisance"  # 把缺陷误认为正常（漏检）
        elif true_label != 0 and pred_label == 0:
            return "nuisance_as_defect"  # 把正常误认为缺陷（误报）
        else:
            return "other"

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'gate_total_errors': len(self.gate_errors),
            'fine_total_errors': len(self.fine_errors),
            'gate_errors_by_type': {
                'defect_as_nuisance': sum(1 for e in self.gate_errors if e['error_type'] == 'defect_as_nuisance'),
                'nuisance_as_defect': sum(1 for e in self.gate_errors if e['error_type'] == 'nuisance_as_defect')
            },
            'top_gate_confusion': sorted(
                [{'true': k[0], 'pred': k[1], 'count': v} for k, v in self.gate_confusion.items()],
                key=lambda x: -x['count']
            )[:10],
            'top_fine_confusion': sorted(
                [{'true': k[0], 'pred': k[1], 'count': v} for k, v in self.fine_confusion.items()],
                key=lambda x: -x['count']
            )[:10],
            'most_confused_classes': sorted(
                [{'class': k, 'errors': v} for k, v in self.gate_per_class_errors.items()],
                key=lambda x: -x['errors']
            )[:5]
        }

    def save_report(self, output_dir: str, prefix: str = "misclassified"):
        """Save detailed misclassification report."""
        os.makedirs(output_dir, exist_ok=True)

        # Summary
        summary = self.get_summary()
        with open(os.path.join(output_dir, f"{prefix}_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Gate errors
        if self.gate_errors:
            # 按错误类型分组
            gate_report = {
                'defect_as_nuisance': [],  # 漏检：把缺陷误认为正常
                'nuisance_as_defect': [],   # 误报：把正常误认为缺陷
                'other': []
            }
            for e in self.gate_errors:
                gate_report[e['error_type']].append(e)

            with open(os.path.join(output_dir, f"{prefix}_gate_errors.json"), 'w', encoding='utf-8') as f:
                json.dump(gate_report, f, ensure_ascii=False, indent=2)

        # Fine errors
        if self.fine_errors:
            # 按真实类别分组
            fine_by_true = defaultdict(list)
            for e in self.fine_errors:
                fine_by_true[e['true_name']].append(e)

            fine_report = {
                'by_true_class': {k: list(v) for k, v in fine_by_true.items()},
                'all_errors': self.fine_errors
            }
            with open(os.path.join(output_dir, f"{prefix}_fine_errors.json"), 'w', encoding='utf-8') as f:
                json.dump(fine_report, f, ensure_ascii=False, indent=2)

        # CSV for easy inspection
        if self.gate_errors:
            import csv
            with open(os.path.join(output_dir, f"{prefix}_gate_errors.csv"), 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'true_name', 'pred_name', 'error_type', 'defect_prob'])
                writer.writeheader()
                for e in self.gate_errors:
                    writer.writerow({
                        'path': e['path'],
                        'true_name': e['true_name'],
                        'pred_name': e['pred_name'],
                        'error_type': e['error_type'],
                        'defect_prob': f"{e['defect_prob']:.4f}"
                    })

        if self.fine_errors:
            import csv
            with open(os.path.join(output_dir, f"{prefix}_fine_errors.csv"), 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'true_name', 'pred_name', 'pred_prob'])
                writer.writeheader()
                for e in self.fine_errors:
                    writer.writerow({
                        'path': e['path'],
                        'true_name': e['true_name'],
                        'pred_name': e['pred_name'],
                        'pred_prob': f"{e['pred_prob']:.4f}"
                    })

        return summary


class WaferDefectTrainer:
    """
    Trainer for wafer defect classification model.
    Includes misclassification tracking for dataset debugging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        gate_weight: float = 1.0,
        fine_weight: float = 0.5,
        metric_weight: float = 0.1,
        defect_weight: float = 3.0,
        output_dir: str = "output",
        class_names: dict = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gate_weight = gate_weight
        self.fine_weight = fine_weight
        self.metric_weight = metric_weight
        self.defect_weight = defect_weight
        self.output_dir = output_dir
        self.class_names = class_names or {}

        self.criterion = CombinedLoss(
            gate_weight=gate_weight,
            fine_weight=fine_weight,
            metric_weight=metric_weight,
            defect_weight=defect_weight
        )

        self.gate_metrics = GateMetrics()
        self.fine_metrics = FineMetrics(num_classes=model.num_defect_classes)

        # Misclassification tracking
        self.tracker = MisclassificationTracker(
            class_names=class_names,
            num_classes=model.num_defect_classes
        )

        os.makedirs(output_dir, exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> dict:
        """Train for one epoch."""
        self.model.train()

        gate_metrics = GateMetrics()
        fine_metrics = FineMetrics(num_classes=self.model.num_defect_classes)

        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            images = batch["images"].to(self.device)
            label = batch["label"].to(self.device)
            is_defect = batch["is_defect"].to(self.device)
            defect_type = batch["defect_type"].to(self.device)

            # Adjust defect_type: map 1~K to 0~(K-1) for fine head
            defect_type_adjusted = defect_type.clone()
            mask = defect_type > 0
            defect_type_adjusted[mask] = defect_type[mask] - 1

            # Forward
            outputs = self.model(images, return_features=True)

            # Compute loss
            losses = self.criterion(
                gate_logits=outputs["gate_logits"],
                fine_logits=outputs["fine_logits"],
                features=outputs["feat"],
                is_defect_target=is_defect,
                defect_target=defect_type_adjusted
            )

            loss = losses["total"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            gate_metrics.update(outputs["gate_logits"], is_defect)

            # Fine metrics only for defect samples
            defect_mask = is_defect == 1
            if defect_mask.sum() > 0:
                fine_metrics.update(
                    outputs["fine_logits"][defect_mask],
                    defect_type_adjusted[defect_mask]
                )

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Update anomaly centers at end of epoch
        with torch.no_grad():
            self._update_anomaly_centers(train_loader)

        return {
            "train_loss": total_loss / num_batches,
            "gate_metrics": gate_metrics.compute(),
            "fine_metrics": fine_metrics.compute()
        }

    @torch.no_grad()
    def _update_anomaly_centers(self, loader: DataLoader):
        """Update class centers for anomaly detection."""
        all_feats = []
        all_labels = []

        for batch in loader:
            images = batch["images"].to(self.device)
            defect_type = batch["defect_type"].to(self.device)

            outputs = self.model(images, return_features=True)

            all_feats.append(outputs["feat"].cpu())
            all_labels.append(defect_type.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Only use defect samples for centers
        mask = all_labels > 0
        if mask.sum() > 0:
            self.model.update_anomaly_centers(all_feats[mask], all_labels[mask])

            # Update normalization statistics
            if hasattr(self.model, 'anomaly') and hasattr(self.model.anomaly, 'update_statistics'):
                self.model.anomaly.update_statistics(all_feats[mask])

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, save_errors: bool = True) -> dict:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            save_errors: Whether to track and save misclassified samples

        Returns:
            dict with validation metrics and misclassification summary
        """
        self.model.eval()

        gate_metrics = GateMetrics()
        fine_metrics = FineMetrics(num_classes=self.model.num_defect_classes)

        total_loss = 0
        num_batches = 0

        # Reset tracker for validation
        tracker = MisclassificationTracker(
            class_names=self.class_names,
            num_classes=self.model.num_defect_classes
        )

        # Try to get dataset for path info
        dataset = getattr(val_loader, 'dataset', None)
        parent_dataset = None
        if hasattr(dataset, 'parent'):
            parent_dataset = dataset.parent
        elif hasattr(dataset, 'samples'):
            parent_dataset = dataset

        sample_idx_offset = 0

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"].to(self.device)
            label = batch["label"].to(self.device)
            is_defect = batch["is_defect"].to(self.device)
            defect_type = batch["defect_type"].to(self.device)
            batch_size = images.size(0)

            defect_type_adjusted = defect_type.clone()
            mask = defect_type > 0
            defect_type_adjusted[mask] = defect_type[mask] - 1

            outputs = self.model(images, return_features=True)

            losses = self.criterion(
                gate_logits=outputs["gate_logits"],
                fine_logits=outputs["fine_logits"],
                features=outputs["feat"],
                is_defect_target=is_defect,
                defect_target=defect_type_adjusted
            )

            loss = losses["total"]

            gate_metrics.update(outputs["gate_logits"], is_defect)

            defect_mask = is_defect == 1
            if defect_mask.sum() > 0:
                fine_metrics.update(
                    outputs["fine_logits"][defect_mask],
                    defect_type_adjusted[defect_mask]
                )

            # Track misclassifications
            if save_errors and parent_dataset is not None:
                gate_probs = torch.softmax(outputs["gate_logits"], dim=1)
                fine_probs = torch.softmax(outputs["fine_logits"], dim=1)

                gate_preds = outputs["is_defect_pred"]
                fine_preds = outputs["fine_pred"]

                for i in range(batch_size):
                    sample_idx = sample_idx_offset + i
                    sample_info = self._get_sample_info(parent_dataset, sample_idx)

                    # Gate errors
                    true_is_defect = is_defect[i].item()
                    pred_is_defect = gate_preds[i].item()
                    if true_is_defect != pred_is_defect:
                        tracker.add_gate_error(
                            path=sample_info.get('path', f"sample_{sample_idx}"),
                            true_label=true_is_defect,
                            pred_label=pred_is_defect,
                            pred_prob=gate_probs[i][pred_is_defect].item(),
                            defect_prob=gate_probs[i][1].item()
                        )

                    # Fine errors (only for defect samples)
                    if defect_mask[i] and defect_type[i] > 0:
                        true_defect = defect_type[i].item()
                        pred_defect = fine_preds[i].item()
                        # Adjust for the -1 padding in fine_logits
                        pred_defect_adjusted = pred_defect

                        if true_defect != pred_defect + 1:  # +1 because pred is in 0~(K-1)
                            tracker.add_fine_error(
                                path=sample_info.get('path', f"sample_{sample_idx}"),
                                true_label=true_defect,
                                pred_label=pred_defect + 1,
                                all_probs=fine_probs[i].cpu().tolist()
                            )

            total_loss += loss.item()
            num_batches += 1
            sample_idx_offset += batch_size

        # Save misclassification report
        summary = tracker.get_summary()

        # Compute anomaly score statistics for reference
        anomaly_stats = None
        if hasattr(self.model, 'anomaly'):
            dist_mean = self.model.anomaly.dist_mean.item()
            dist_std = self.model.anomaly.dist_std.item()
            anomaly_stats = {
                'dist_mean': dist_mean,
                'dist_std': dist_std,
                'threshold_used': 2.0  # z-score threshold
            }
            summary['anomaly_stats'] = anomaly_stats

        if save_errors and summary['gate_total_errors'] + summary['fine_total_errors'] > 0:
            report_path = os.path.join(self.output_dir, "misclassification_reports")
            tracker.save_report(report_path, prefix="val")
            print(f"\n[Misclassification Report]")
            print(f"  Gate errors: {summary['gate_total_errors']} "
                  f"(defect->nuisance: {summary['gate_errors_by_type']['defect_as_nuisance']}, "
                  f"nuisance->defect: {summary['gate_errors_by_type']['nuisance_as_defect']})")
            print(f"  Fine errors: {summary['fine_total_errors']}")
            if anomaly_stats:
                print(f"  Anomaly detection: dist_mean={anomaly_stats['dist_mean']:.4f}, "
                      f"dist_std={anomaly_stats['dist_std']:.4f}")
            print(f"  Report saved to: {report_path}/")

        result = {
            "val_loss": total_loss / num_batches,
            "gate_metrics": gate_metrics.compute(),
            "fine_metrics": fine_metrics.compute(),
            "misclassification_summary": summary if save_errors else None
        }

        return result

    def _get_sample_info(self, dataset, idx: int) -> dict:
        """Get sample information including file paths."""
        info = {'path': f'sample_{idx}'}

        # Try RealWaferDataset
        if hasattr(dataset, 'samples') and idx < len(dataset.samples):
            sample = dataset.samples[idx]
            if 'paths' in sample and sample['paths']:
                info['path'] = str(sample['paths'][0])  # First view path
            info['label'] = sample.get('label', -1)
            info['is_defect'] = sample.get('is_defect', -1)

        # Try SubsetDataset
        elif hasattr(dataset, 'indices') and hasattr(dataset, 'parent'):
            real_idx = dataset.indices[idx]
            return self._get_sample_info(dataset.parent, real_idx)

        return info

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)
