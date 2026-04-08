"""
Training engine for wafer defect classification.

Includes:
- WaferDefectTrainer: Base trainer with misclassification tracking
- ThreePhaseTrainer: Three-phase training (classification -> Dinomaly2 -> joint)
- generate_markdown_report: Markdown report generation

Three-Phase Training:
    Phase 1 (classification): Train Gate + Fine heads
    Phase 2 (dinomaly2): Train Dinomaly2 decoder
    Phase 3 (joint): Joint fine-tuning (optional)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from ..losses import CombinedLoss
from ..utils.metrics import GateMetrics, FineMetrics


# Training phases
PHASE_CLASSIFICATION = "classification"
PHASE_DINOMALITY2 = "dinomaly2"
PHASE_JOINT = "joint"


def _ascii_bar_chart(values: list, keys: list = None, width: int = 40,
                      fill_char: str = "█", null_char: str = "─") -> str:
    """Generate a horizontal ASCII bar chart."""
    if not values:
        return ""
    max_val = max(values)
    if max_val == 0:
        max_val = 1
    lines = []
    for i, v in enumerate(values):
        bar_len = int(round(v / max_val * width))
        bar = fill_char * bar_len
        label = keys[i] if keys and i < len(keys) else str(i)
        lines.append(f"  {label:<12} {bar} {v:.4f}")
    return "\n".join(lines)


def _ascii_confusion_matrix(cm: list, labels: list = None, null_char: str = "─") -> str:
    """Generate an ASCII heatmap-style confusion matrix."""
    if not cm:
        return "N/A"
    n = len(cm)
    header_labels = [f"{l[:8]:>8}" for l in (labels or [str(i) for i in range(n)])]
    sep = "+" + "+".join("-" * 9 for _ in header_labels) + "+"
    lines = [sep]
    lines.append("|" + "|".join([" Pred →"] + header_labels) + "|")
    lines.append(sep.replace("─", "="))
    for i, row in enumerate(cm):
        true_lbl = f"{labels[i][:8] if labels else str(i):>8}"
        vals = "".join(f"{v:>9}" for v in row)
        lines.append(f"| {true_lbl} |{vals}|")
        lines.append(sep)
    return "\n".join(lines)


def _color_text(text: str, color: str) -> str:
    """Return ANSI-colored text string."""
    colors = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
        "bold": "\033[1m", "reset": "\033[0m",
    }
    c = colors.get(color, "")
    r = colors["reset"]
    return f"{c}{text}{r}"


class MisclassificationTracker:
    """Track misclassified samples for debugging."""

    def __init__(self, class_names=None, num_classes=10):
        self.num_classes = num_classes

        # Gate errors (Nuisance vs Defect confusion)
        self.gate_errors = []

        # Fine errors (Defect type confusion)
        self.fine_errors = []

        # Confusion tracking
        self.gate_confusion = defaultdict(int)
        self.fine_confusion = defaultdict(int)

        # Per-class error counts
        self.gate_per_class_errors = defaultdict(int)
        self.fine_per_class_errors = defaultdict(int)

        if class_names is None:
            self.class_names = {}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = {i: str(name) for i, name in enumerate(class_names)}

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
            return "defect_as_nuisance"
        elif true_label != 0 and pred_label == 0:
            return "nuisance_as_defect"
        else:
            return "other"

    def get_summary(self) -> dict:
        """Get summary statistics including error lists."""
        return {
            'gate_total_errors': len(self.gate_errors),
            'fine_total_errors': len(self.fine_errors),
            'gate_errors_by_type': {
                'defect_as_nuisance': sum(1 for e in self.gate_errors if e['error_type'] == 'defect_as_nuisance'),
                'nuisance_as_defect': sum(1 for e in self.gate_errors if e['error_type'] == 'nuisance_as_defect')
            },
            'gate_errors': self.gate_errors,
            'fine_errors': self.fine_errors,
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

        summary = self.get_summary()
        with open(os.path.join(output_dir, f"{prefix}_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.gate_errors:
            gate_report = {
                'defect_as_nuisance': [],
                'nuisance_as_defect': [],
                'other': []
            }
            for e in self.gate_errors:
                gate_report[e['error_type']].append(e)

            with open(os.path.join(output_dir, f"{prefix}_gate_errors.json"), 'w', encoding='utf-8') as f:
                json.dump(gate_report, f, ensure_ascii=False, indent=2)

        if self.fine_errors:
            fine_by_true = defaultdict(list)
            for e in self.fine_errors:
                fine_by_true[e['true_name']].append(e)

            fine_report = {
                'by_true_class': {k: list(v) for k, v in fine_by_true.items()},
                'all_errors': self.fine_errors
            }
            with open(os.path.join(output_dir, f"{prefix}_fine_errors.json"), 'w', encoding='utf-8') as f:
                json.dump(fine_report, f, ensure_ascii=False, indent=2)

        # CSV exports
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


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    val_results: dict,
    history: list = None,
    class_names: dict = None,
    dataset_info: dict = None,
    output_dir: str = "output",
    prefix: str = "validation_report"
) -> str:
    """
    Generate a comprehensive markdown validation report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{prefix}.md")

    if class_names is None:
        class_names = {}
    elif not isinstance(class_names, dict):
        class_names = {i: str(n) for i, n in enumerate(class_names)}

    def name_of(label: int, fallback: str = "unknown") -> str:
        return class_names.get(label, fallback)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Wafer Defect Classification - Validation Report")
    md.append(f"")
    md.append(f"> Generated: `{now}`  - Engine: `WaferDefectTrainer`")
    md.append(f"")

    # Dataset Info
    if dataset_info:
        md.append(f"## 1. Dataset Overview")
        md.append(f"")
        md.append(f"| Item | Value |")
        md.append(f"|------|-------|")
        md.append(f"| Total samples | {dataset_info.get('total', 'N/A')} |")
        md.append(f"| Training samples | {dataset_info.get('train', 'N/A')} |")
        md.append(f"| Validation samples | {dataset_info.get('val', 'N/A')} |")
        md.append(f"| Number of classes | {dataset_info.get('num_classes', 'N/A')} |")
        md.append(f"| Defect classes | {dataset_info.get('defect_classes', 'N/A')} |")
        md.append(f"")

    # Gate Metrics
    gate = val_results.get('gate_metrics', {})
    md.append(f"## 2. Gate Head - Nuisance vs Defect")
    md.append(f"")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| **Accuracy** | `{gate.get('accuracy', 0):.4f}` |")
    md.append(f"| Nuisance Recall | `{gate.get('nuisance_recall', 0):.4f}` |")
    md.append(f"| **Defect Recall** | `{gate.get('defect_recall', 0):.4f}` |")
    md.append(f"| Nuisance->Defect Rate (FP) | `{gate.get('nuisance_as_defect_rate', 0):.4f}` |")
    md.append(f"| **Defect->Nuisance Rate (FN)** | `{gate.get('defect_as_nuisance_rate', 0):.4f}` |")
    if gate.get('auc') is not None:
        md.append(f"| AUC | `{gate.get('auc', 0):.4f}` |")
    md.append(f"")

    # Fine Metrics
    fine = val_results.get('fine_metrics', {})
    md.append(f"## 3. Fine Head - Defect Type Classification")
    md.append(f"")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| **Accuracy** | `{fine.get('accuracy', 0):.4f}` |")
    md.append(f"| Macro Precision | `{fine.get('macro_precision', 0):.4f}` |")
    md.append(f"| Macro Recall | `{fine.get('macro_recall', 0):.4f}` |")
    md.append(f"| **Macro F1** | `{fine.get('macro_f1', 0):.4f}` |")
    md.append(f"")

    # Per-class metrics
    pcp = fine.get('per_class_precision', [])
    pcr = fine.get('per_class_recall', [])
    pcf = fine.get('per_class_f1', [])
    if pcp:
        md.append(f"### Per-Class Metrics")
        md.append(f"")
        md.append(f"| Class | Precision | Recall | F1-Score |")
        md.append(f"|-------|-----------|--------|----------|")
        for i, (p, r, f) in enumerate(zip(pcp, pcr, pcf)):
            md.append(f"| *{name_of(i+1, f'class_{i+1}')}* | {p:.4f} | {r:.4f} | {f:.4f} |")
        md.append(f"")

    # Misclassification Analysis
    summary = val_results.get('misclassification_summary')
    if summary:
        md.append(f"## 4. Misclassification Analysis")
        md.append(f"")
        ge = summary['gate_errors_by_type']
        fe_count = summary['fine_total_errors']
        md.append(f"**Gate Errors:** {summary['gate_total_errors']} total")
        md.append(f"  - FN (Defect->Nuisance): {ge['defect_as_nuisance']}")
        md.append(f"  - FP (Nuisance->Defect): {ge['nuisance_as_defect']}")
        md.append(f"")
        md.append(f"**Fine Errors:** {fe_count} total")
        md.append(f"")

    # Training History
    if history:
        md.append(f"## 5. Training History")
        md.append(f"")
        md.append(f"| Epoch | Train Loss | Val Loss | Gate Acc | Fine F1 |")
        md.append(f"|-------|------------|---------|---------|---------|")
        for rec in history:
            md.append(f"| {rec['epoch']} | {rec['train_loss']:.4f} | {rec['val_loss']:.4f} | "
                      f"{rec.get('gate_accuracy', 0):.4f} | {rec.get('fine_macro_f1', 0):.4f} |")
        md.append(f"")

    md.append(f"---")
    md.append(f"*Report generated by WaferDefectTrainer - {now}*")

    content = "\n".join(md)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nReport saved -> {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
#  Base Trainer
# ─────────────────────────────────────────────────────────────────────────────

class WaferDefectTrainer:
    """
    Base trainer for wafer defect classification model.

    Supports:
    - Classification training (Gate + Fine heads)
    - Misclassification tracking
    - Markdown report generation
    - Checkpoint saving/loading

    Args:
        model: The model to train
        optimizer: Optimizer instance
        device: Device to use
        gate_weight: Gate loss weight
        fine_weight: Fine loss weight
        metric_weight: Metric loss weight
        defect_weight: Weight for defect class in gate loss
        output_dir: Output directory for checkpoints and reports
        class_names: Class name mapping
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

            outputs = self.model(images, return_features=True)

            losses = self.criterion(
                gate_logits=outputs["gate_logits"],
                fine_logits=outputs["fine_logits"],
                features=outputs["feat"],
                is_defect_target=is_defect,
                defect_target=defect_type_adjusted
            )

            loss = losses["total"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            gate_metrics.update(outputs["gate_logits"], is_defect)

            defect_mask = is_defect == 1
            if defect_mask.sum() > 0:
                fine_metrics.update(
                    outputs["fine_logits"][defect_mask],
                    defect_type_adjusted[defect_mask]
                )

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Update anomaly centers at end of epoch (only for models that support it)
        with torch.no_grad():
            if hasattr(self.model, 'update_anomaly_centers'):
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

        mask = all_labels > 0
        if mask.sum() > 0:
            self.model.update_anomaly_centers(all_feats[mask], all_labels[mask] - 1)

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

        tracker = MisclassificationTracker(
            class_names=self.class_names,
            num_classes=self.model.num_defect_classes
        )

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

            if save_errors and parent_dataset is not None:
                gate_probs = torch.softmax(outputs["gate_logits"], dim=1)
                fine_probs = torch.softmax(outputs["fine_logits"], dim=1)

                gate_preds = outputs["is_defect"]
                fine_preds = outputs["defect_type"]

                for i in range(batch_size):
                    sample_idx = sample_idx_offset + i
                    sample_info = self._get_sample_info(parent_dataset, sample_idx)

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

                    if defect_mask[i] and defect_type[i] > 0:
                        true_defect = defect_type[i].item()
                        pred_defect = fine_preds[i].item()

                        if true_defect != pred_defect + 1:
                            tracker.add_fine_error(
                                path=sample_info.get('path', f"sample_{sample_idx}"),
                                true_label=true_defect,
                                pred_label=pred_defect + 1,
                                all_probs=fine_probs[i].cpu().tolist()
                            )

            total_loss += loss.item()
            num_batches += 1
            sample_idx_offset += batch_size

        summary = tracker.get_summary()

        anomaly_stats = None
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            if hasattr(anomaly, 'dist_mean') and hasattr(anomaly, 'dist_std'):
                dist_mean = anomaly.dist_mean.item()
                dist_std = anomaly.dist_std.item()
            elif hasattr(anomaly, '_score_mean') and hasattr(anomaly, '_score_std'):
                dist_mean = anomaly._score_mean.item() if isinstance(anomaly._score_mean, torch.Tensor) else float(anomaly._score_mean)
                dist_std = anomaly._score_std.item() if isinstance(anomaly._score_std, torch.Tensor) else float(anomaly._score_std)
            else:
                dist_mean, dist_std = 0.0, 1.0
            threshold = getattr(anomaly, 'anomaly_threshold', 2.0)
            anomaly_stats = {
                'dist_mean': dist_mean,
                'dist_std': dist_std,
                'threshold_used': threshold
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

        return {
            "val_loss": total_loss / num_batches,
            "gate_metrics": gate_metrics.compute(),
            "fine_metrics": fine_metrics.compute(),
            "misclassification_summary": summary if save_errors else None
        }

    def _get_sample_info(self, dataset, idx: int) -> dict:
        """Get sample information including file paths."""
        info = {'path': f'sample_{idx}'}

        if hasattr(dataset, 'samples') and idx < len(dataset.samples):
            sample = dataset.samples[idx]
            # Handle both dict samples (real data) and WaferDefectSample objects (synthetic)
            if isinstance(sample, dict):
                if 'paths' in sample and sample['paths']:
                    info['path'] = str(sample['paths'][0])
                info['label'] = sample.get('label', -1)
                info['is_defect'] = sample.get('is_defect', -1)
            else:
                # WaferDefectSample object
                info['label'] = getattr(sample, 'label', -1)
                info['is_defect'] = getattr(sample, 'is_defect', -1)

        elif hasattr(dataset, 'indices') and hasattr(dataset, 'parent'):
            real_idx = dataset.indices[idx]
            return self._get_sample_info(dataset.parent, real_idx)

        return info

    def save_checkpoint(self, path: str, epoch: int, extra: dict = None):
        """Save model checkpoint with metadata."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_names": self.class_names,
        }
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            if hasattr(anomaly, 'dist_mean'):
                v = anomaly.dist_mean
                ckpt["anomaly_score_mean"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, 'dist_std'):
                v = anomaly.dist_std
                ckpt["anomaly_score_std"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, '_score_mean'):
                v = anomaly._score_mean
                ckpt["rad_score_mean"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, '_score_std'):
                v = anomaly._score_std
                ckpt["rad_score_std"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, 'anomaly_threshold'):
                ckpt["anomaly_threshold"] = float(anomaly.anomaly_threshold)
            ckpt["use_rad_anomaly"] = getattr(self.model, 'use_rad_anomaly', False)
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint and restore anomaly stats."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.class_names = checkpoint.get("class_names", {})
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            if "anomaly_score_mean" in checkpoint and hasattr(anomaly, 'dist_mean'):
                anomaly.dist_mean = torch.tensor(checkpoint["anomaly_score_mean"])
            if "anomaly_score_std" in checkpoint and hasattr(anomaly, 'dist_std'):
                anomaly.dist_std = torch.tensor(checkpoint["anomaly_score_std"])
            if "rad_score_mean" in checkpoint and hasattr(anomaly, '_score_mean'):
                anomaly._score_mean = torch.tensor(checkpoint["rad_score_mean"])
            if "rad_score_std" in checkpoint and hasattr(anomaly, '_score_std'):
                anomaly._score_std = torch.tensor(checkpoint["rad_score_std"])
            if "anomaly_threshold" in checkpoint and hasattr(anomaly, 'anomaly_threshold'):
                anomaly.anomaly_threshold = checkpoint["anomaly_threshold"]
        return checkpoint.get("epoch", 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Three-Phase Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ThreePhaseTrainer:
    """
    Three-phase trainer for wafer defect classification with Dinomaly2.

    Phase 1 (classification): Train Gate + Fine heads
    Phase 2 (dinomaly2): Train Dinomaly2 decoder
    Phase 3 (joint): Joint fine-tuning (optional)

    Args:
        model: The WaferDefectModel to train
        optimizer: Optimizer instance
        device: Device to use
        output_dir: Output directory for checkpoints
        class_names: Class name mapping
        phase1_config: Configuration for Phase 1
        phase2_config: Configuration for Phase 2
        phase3_config: Configuration for Phase 3
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        output_dir: str = "output",
        class_names: dict = None,
        phase1_config: dict = None,
        phase2_config: dict = None,
        phase3_config: dict = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.class_names = class_names or {}

        # Default configurations
        self.phase1_config = phase1_config or {
            'gate_weight': 1.0,
            'fine_weight': 0.5,
            'metric_weight': 0.1,
            'defect_weight': 3.0,
        }

        self.phase2_config = phase2_config or {
            'lr': 2e-3,
            'iters': 40000,
            'log_interval': 500,
        }

        self.phase3_config = phase3_config or {
            'lr': 1e-5,
            'epochs': 5,
        }

        # Current phase
        self.current_phase = None
        self.phase_history = {}

        # Create base trainer for Phase 1 and 3
        self.base_trainer = WaferDefectTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            gate_weight=self.phase1_config.get('gate_weight', 1.0),
            fine_weight=self.phase1_config.get('fine_weight', 0.5),
            metric_weight=self.phase1_config.get('metric_weight', 0.1),
            defect_weight=self.phase1_config.get('defect_weight', 3.0),
            output_dir=output_dir,
            class_names=class_names,
        )

        os.makedirs(output_dir, exist_ok=True)

    def set_phase(self, phase: str):
        """
        Set the current training phase.

        Args:
            phase: One of "classification", "dinomaly2", "joint"
        """
        valid_phases = [PHASE_CLASSIFICATION, PHASE_DINOMALITY2, PHASE_JOINT]
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {valid_phases}")

        if self.current_phase != phase:
            print(f"\n{'=' * 60}")
            print(f"Switching to Phase: {phase.upper()}")
            print(f"{'=' * 60}")
            self.current_phase = phase

            # Configure optimizer for the phase
            self._configure_optimizer(phase)

    def _configure_optimizer(self, phase: str):
        """Configure optimizer for the given phase."""
        if phase == PHASE_CLASSIFICATION:
            # Classification: use main learning rate
            for param_group in self.optimizer.param_groups:
                if 'lr' not in param_group:
                    param_group['lr'] = self.phase1_config.get('lr', 1e-4)

        elif phase == PHASE_DINOMALITY2:
            # Dinomaly2: higher learning rate for decoder
            dinomaly_lr = self.phase2_config.get('lr', 2e-3)
            # Note: This would need access to decoder parameters specifically
            print(f"[Phase 2] Dinomaly2 decoder LR: {dinomaly_lr}")

        elif phase == PHASE_JOINT:
            # Joint: lower learning rate for fine-tuning
            joint_lr = self.phase3_config.get('lr', 1e-5)
            print(f"[Phase 3] Joint fine-tuning LR: {joint_lr}")

    def train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_checkpoints: bool = True,
    ) -> List[dict]:
        """
        Phase 1: Train classification heads (Gate + Fine).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_checkpoints: Whether to save checkpoints

        Returns:
            List of epoch records
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Classification Training (Gate + Fine Heads)")
        print("=" * 60)

        self.set_phase(PHASE_CLASSIFICATION)
        self.phase_history[PHASE_CLASSIFICATION] = []

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            train_results = self.base_trainer.train_epoch(train_loader, epoch)
            val_results = self.base_trainer.validate(val_loader)

            print(f"Train Loss: {train_results['train_loss']:.4f}")
            print(f"Gate - Acc: {train_results['gate_metrics']['accuracy']:.4f}, "
                  f"Defect Recall: {train_results['gate_metrics']['defect_recall']:.4f}")
            print(f"Fine - Acc: {val_results['fine_metrics']['accuracy']:.4f}, "
                  f"Macro F1: {val_results['fine_metrics']['macro_f1']:.4f}")

            summary = val_results.get('misclassification_summary')
            record = {
                'epoch': epoch,
                'train_loss': train_results['train_loss'],
                'val_loss': val_results['val_loss'],
                'gate_accuracy': train_results['gate_metrics']['accuracy'],
                'gate_defect_recall': train_results['gate_metrics']['defect_recall'],
                'fine_accuracy': val_results['fine_metrics']['accuracy'],
                'fine_macro_f1': val_results['fine_metrics']['macro_f1'],
                'gate_errors': summary['gate_total_errors'] if summary else 0,
                'fine_errors': summary['fine_total_errors'] if summary else 0,
            }
            self.phase_history[PHASE_CLASSIFICATION].append(record)

            if save_checkpoints and val_results['val_loss'] < best_val_loss:
                best_val_loss = val_results['val_loss']
                best_epoch = epoch
                ckpt_path = os.path.join(self.output_dir, "phase1_best.pt")
                self.save_checkpoint(ckpt_path, epoch, phase=PHASE_CLASSIFICATION)
                print(f"New best! Loss: {best_val_loss:.4f}")

        if save_checkpoints:
            last_path = os.path.join(self.output_dir, "phase1_last.pt")
            self.save_checkpoint(last_path, epochs, phase=PHASE_CLASSIFICATION)

        print(f"\nPhase 1 complete. Best epoch: {best_epoch} (loss={best_val_loss:.4f})")
        return self.phase_history[PHASE_CLASSIFICATION]

    def train_phase2(
        self,
        defect_loader: DataLoader,
        save_decoder: bool = True,
    ) -> dict:
        """
        Phase 2: Train Dinomaly2 decoder.

        Args:
            defect_loader: DataLoader with defect samples only
            save_decoder: Whether to save the trained decoder

        Returns:
            Training info dict
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Dinomaly2 Decoder Training")
        print("=" * 60)

        self.set_phase(PHASE_DINOMALITY2)

        if not self.model.use_dinomaly2:
            print("[Warning] Dinomaly2 is not enabled. Skipping Phase 2.")
            return {'status': 'skipped'}

        # Train the decoder
        save_path = os.path.join(self.output_dir, "dinomaly2_decoder.pth") if save_decoder else None
        self.model.build_dinomaly2(
            defect_loader=defect_loader,
            device=self.device,
            save_path=save_path,
            log_interval=self.phase2_config.get('log_interval', 500),
        )

        print(f"Dinomaly2 decoder training complete.")
        if save_decoder:
            print(f"Decoder saved to: {save_path}")

        self.phase_history[PHASE_DINOMALITY2] = {
            'status': 'completed',
            'decoder_path': save_path,
        }
        return self.phase_history[PHASE_DINOMALITY2]

    def train_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        save_checkpoints: bool = True,
    ) -> List[dict]:
        """
        Phase 3: Joint fine-tuning (optional).

        Fine-tunes the entire model with a lower learning rate.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config if None)
            save_checkpoints: Whether to save checkpoints

        Returns:
            List of epoch records
        """
        print("\n" + "=" * 60)
        print("PHASE 3: Joint Fine-Tuning")
        print("=" * 60)

        self.set_phase(PHASE_JOINT)

        # Reduce learning rate for joint fine-tuning
        joint_lr = self.phase3_config.get('lr', 1e-5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = joint_lr
        print(f"Learning rate reduced to: {joint_lr}")

        epochs = epochs or self.phase3_config.get('epochs', 5)
        self.phase_history[PHASE_JOINT] = []

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            train_results = self.base_trainer.train_epoch(train_loader, epoch)
            val_results = self.base_trainer.validate(val_loader)

            print(f"Train Loss: {train_results['train_loss']:.4f}")
            print(f"Gate - Acc: {train_results['gate_metrics']['accuracy']:.4f}, "
                  f"Defect Recall: {train_results['gate_metrics']['defect_recall']:.4f}")
            print(f"Fine - Acc: {val_results['fine_metrics']['accuracy']:.4f}, "
                  f"Macro F1: {val_results['fine_metrics']['macro_f1']:.4f}")

            summary = val_results.get('misclassification_summary')
            record = {
                'epoch': epoch,
                'train_loss': train_results['train_loss'],
                'val_loss': val_results['val_loss'],
                'gate_accuracy': train_results['gate_metrics']['accuracy'],
                'gate_defect_recall': train_results['gate_metrics']['defect_recall'],
                'fine_accuracy': val_results['fine_metrics']['accuracy'],
                'fine_macro_f1': val_results['fine_metrics']['macro_f1'],
                'gate_errors': summary['gate_total_errors'] if summary else 0,
                'fine_errors': summary['fine_total_errors'] if summary else 0,
            }
            self.phase_history[PHASE_JOINT].append(record)

            if save_checkpoints and val_results['val_loss'] < best_val_loss:
                best_val_loss = val_results['val_loss']
                best_epoch = epoch
                ckpt_path = os.path.join(self.output_dir, "phase3_best.pt")
                self.save_checkpoint(ckpt_path, epoch, phase=PHASE_JOINT)
                print(f"New best! Loss: {best_val_loss:.4f}")

        if save_checkpoints:
            last_path = os.path.join(self.output_dir, "phase3_last.pt")
            self.save_checkpoint(last_path, epochs, phase=PHASE_JOINT)

        print(f"\nPhase 3 complete. Best epoch: {best_epoch} (loss={best_val_loss:.4f})")
        return self.phase_history[PHASE_JOINT]

    def train_all_phases(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        defect_loader: DataLoader = None,
        phase1_epochs: int = 10,
        phase3_epochs: Optional[int] = None,
        skip_phase2: bool = False,
        skip_phase3: bool = False,
    ) -> dict:
        """
        Run the full three-phase training pipeline.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            defect_loader: Defect-only loader for Dinomaly2 (optional)
            phase1_epochs: Epochs for Phase 1
            phase3_epochs: Epochs for Phase 3 (optional)
            skip_phase2: Skip Dinomaly2 training
            skip_phase3: Skip joint fine-tuning

        Returns:
            Full training history
        """
        history = {}

        # Phase 1: Classification
        history['phase1'] = self.train_phase1(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=phase1_epochs,
        )

        # Load best model from Phase 1 for Phase 2
        best_p1_path = os.path.join(self.output_dir, "phase1_best.pt")
        if os.path.exists(best_p1_path):
            self.load_checkpoint(best_p1_path)
            print("Loaded Phase 1 best model for Phase 2...")

        # Phase 2: Dinomaly2 (if enabled and data provided)
        if not skip_phase2 and self.model.use_dinomaly2:
            if defect_loader is not None:
                history['phase2'] = self.train_phase2(defect_loader=defect_loader)
            else:
                print("[Warning] No defect_loader provided for Phase 2. Skipping.")
                history['phase2'] = {'status': 'skipped', 'reason': 'no_defect_loader'}
        else:
            history['phase2'] = {'status': 'skipped'}

        # Phase 3: Joint fine-tuning
        if not skip_phase3:
            history['phase3'] = self.train_phase3(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=phase3_epochs,
            )
        else:
            history['phase3'] = {'status': 'skipped'}

        # Save final combined model
        final_path = os.path.join(self.output_dir, "final_model.pt")
        self.save_checkpoint(final_path, epoch=-1, phase='final')

        self.phase_history = history
        return history

    def save_checkpoint(self, path: str, epoch: int, phase: str = None, extra: dict = None):
        """Save model checkpoint with phase information."""
        ckpt = {
            "epoch": epoch,
            "phase": phase or self.current_phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_names": self.class_names,
            "phase_history": self.phase_history,
        }
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            if hasattr(anomaly, 'dist_mean'):
                v = anomaly.dist_mean
                ckpt["anomaly_score_mean"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, 'dist_std'):
                v = anomaly.dist_std
                ckpt["anomaly_score_std"] = v.item() if isinstance(v, torch.Tensor) else float(v)
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint and restore state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.class_names = checkpoint.get("class_names", {})
        self.phase_history = checkpoint.get("phase_history", {})
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            if "anomaly_score_mean" in checkpoint and hasattr(anomaly, 'dist_mean'):
                anomaly.dist_mean = torch.tensor(checkpoint["anomaly_score_mean"])
            if "anomaly_score_std" in checkpoint and hasattr(anomaly, 'dist_std'):
                anomaly.dist_std = torch.tensor(checkpoint["anomaly_score_std"])
        return checkpoint.get("epoch", 0)

    def validate(self, val_loader: DataLoader, save_errors: bool = True) -> dict:
        """Run validation using the base trainer."""
        return self.base_trainer.validate(val_loader, save_errors=save_errors)

    def get_history(self) -> dict:
        """Get the full training history across all phases."""
        return self.phase_history
