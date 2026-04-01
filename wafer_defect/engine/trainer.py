"""
Training engine for wafer defect classification.
Includes misclassification tracking and markdown report generation.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from ..losses import CombinedLoss
from ..utils.metrics import GateMetrics, FineMetrics


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
        self.gate_errors = []  # {'path': str, 'true': int, 'pred': int, 'prob': float}

        # Fine errors (Defect type confusion)
        self.fine_errors = []  # {'path': str, 'true': int, 'pred': int, 'probs': list}

        # Confusion tracking
        self.gate_confusion = defaultdict(int)  # (true, pred) -> count
        self.fine_confusion = defaultdict(int)  # (true, pred) -> count

        # Per-class error counts
        self.gate_per_class_errors = defaultdict(int)
        self.fine_per_class_errors = defaultdict(int)

        # class_names may come as a list (from get_class_names) or dict
        # Normalize to dict: list index = label id
        if class_names is None:
            self.class_names = {}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            # Assume list/tuple, index = label id
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
            return "defect_as_nuisance"  # 把缺陷误认为正常（漏检）
        elif true_label != 0 and pred_label == 0:
            return "nuisance_as_defect"  # 把正常误认为缺陷（误报）
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
            'gate_errors': self.gate_errors,  # full error list for report
            'fine_errors': self.fine_errors,  # full error list for report
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
    Generate a comprehensive markdown validation report with ASCII charts
   , confusion matrices, and error analysis.

    Args:
        val_results: output from WaferDefectTrainer.validate()
        history: list of epoch records from training loop
        class_names: label -> class name mapping (dict or list)
        dataset_info: {'total': int, 'train': int, 'val': int, 'num_classes': int}
        output_dir: where to save the report
        prefix: filename prefix

    Returns:
        Path to the saved markdown file
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{prefix}.md")

    # Normalize class_names to dict
    if class_names is None:
        class_names = {}
    elif not isinstance(class_names, dict):
        class_names = {i: str(n) for i, n in enumerate(class_names)}

    def name_of(label: int, fallback: str = "unknown") -> str:
        return class_names.get(label, fallback)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Header ──────────────────────────────────────────────────────────────
    md = []
    md.append(f"# 🔬 Wafer Defect Classification — Validation Report")
    md.append(f"")
    md.append(f"> Generated: `{now}`  ·  Engine: `WaferDefectTrainer`")
    md.append(f"")

    # ── 2. Dataset Info ────────────────────────────────────────────────────────
    md.append(f"## 1. Dataset Overview")
    md.append(f"")
    if dataset_info:
        md.append(f"| Item | Value |")
        md.append(f"|------|-------|")
        md.append(f"| Total samples | {dataset_info.get('total', 'N/A')} |")
        md.append(f"| Training samples | {dataset_info.get('train', 'N/A')} |")
        md.append(f"| Validation samples | {dataset_info.get('val', 'N/A')} |")
        md.append(f"| Number of classes | {dataset_info.get('num_classes', 'N/A')} |")
        md.append(f"| Defect classes | {dataset_info.get('defect_classes', 'N/A')} |")
        md.append(f"")
    md.append(f"**Class Label Mapping:**")
    for lid, lname in sorted(class_names.items()):
        md.append(f"  - `{lid}` → *{lname}*")
    md.append(f"")

    # ── 3. Gate Metrics (Nuisance vs Defect) ─────────────────────────────────
    gate = val_results.get('gate_metrics', {})
    md.append(f"## 2. Gate Head — Nuisance vs Defect")
    md.append(f"")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| **Accuracy** | `{gate.get('accuracy', 0):.4f}` |")
    md.append(f"| Nuisance Recall | `{gate.get('nuisance_recall', 0):.4f}` |")
    md.append(f"| **Defect Recall** | `{gate.get('defect_recall', 0):.4f}` |")
    md.append(f"| Nuisance→Defect Rate (误报) | `{gate.get('nuisance_as_defect_rate', 0):.4f}` |")
    md.append(f"| **Defect→Nuisance Rate (漏检)** | `{gate.get('defect_as_nuisance_rate', 0):.4f}` |")
    if gate.get('auc') is not None:
        md.append(f"| AUC | `{gate.get('auc', 0):.4f}` |")
    md.append(f"")

    cm_gate = gate.get('confusion_matrix', [])
    if cm_gate:
        gate_labels = [name_of(0, "Nuisance"), name_of(1, "Defect")]
        md.append(f"### Confusion Matrix")
        md.append(f"")
        md.append(f"```")
        md.append(f"Nuisance vs Defect Confusion Matrix (rows=true, cols=pred)")
        md.append(f"")
        md.append(_ascii_confusion_matrix(cm_gate, labels=gate_labels))
        md.append(f"```")
        md.append(f"")
        md.append(f"| | Pred Nuisance | Pred Defect |")
        md.append(f"|---|---|---|")
        md.append(f"| **True Nuisance** | {cm_gate[0][0]} (TN) | {cm_gate[0][1]} (FP) |")
        md.append(f"| **True Defect** | {cm_gate[1][0]} (FN) | {cm_gate[1][1]} (TP) |")
        md.append(f"")

    # Gate score bar
    gate_acc = gate.get('accuracy', 0)
    defect_recall = gate.get('defect_recall', 0)
    md.append(f"### Score Card")
    md.append(f"")
    md.append(f"```")
    md.append(f"Gate Accuracy          {''.join('█' if i < int(gate_acc*20) else '░' for i in range(20))} {gate_acc:.1%}")
    md.append(f"Defect Recall           {''.join('█' if i < int(defect_recall*20) else '░' for i in range(20))} {defect_recall:.1%}")
    md.append(f"```")
    md.append(f"")

    # ── 4. Fine Head (Defect Type) ────────────────────────────────────────────
    fine = val_results.get('fine_metrics', {})
    md.append(f"## 3. Fine Head — Defect Type Classification")
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

    # Classification report
    report_str = fine.get('classification_report', '')
    if report_str:
        md.append(f"### sklearn Classification Report")
        md.append(f"")
        md.append(f"```")
        md.append(report_str.rstrip())
        md.append(f"```")
        md.append(f"")

    # ── 5. Misclassification Analysis ─────────────────────────────────────────
    summary = val_results.get('misclassification_summary')
    if summary:
        md.append(f"## 4. Misclassification Analysis")
        md.append(f"")

        ge = summary['gate_errors_by_type']
        fe_count = summary['fine_total_errors']
        md.append(f"**Gate Errors:** {summary['gate_total_errors']} total  ")
        md.append(f"  - 漏检 (Defect→Nuisance): {ge['defect_as_nuisance']}  ")
        md.append(f"  - 误报 (Nuisance→Defect): {ge['nuisance_as_defect']}")
        md.append(f"")
        md.append(f"**Fine Errors:** {fe_count} total  ")
        md.append(f"")

        # Top confusion pairs
        top_gate = summary.get('top_gate_confusion', [])
        if top_gate:
            md.append(f"### Top Gate Confusion Pairs")
            md.append(f"")
            md.append(f"| True Label | Pred Label | Count |")
            md.append(f"|------------|------------|-------|")
            for item in top_gate[:5]:
                tn = name_of(item['true'], f"class_{item['true']}")
                pn = name_of(item['pred'], f"class_{item['pred']}")
                md.append(f"| {tn} | {pn} | {item['count']} |")
            md.append(f"")

        top_fine = summary.get('top_fine_confusion', [])
        if top_fine:
            md.append(f"### Top Fine Confusion Pairs")
            md.append(f"")
            md.append(f"| True Type | Pred Type | Count |")
            md.append(f"|-----------|-----------|-------|")
            for item in top_fine[:5]:
                tn = name_of(item['true'], f"class_{item['true']}")
                pn = name_of(item['pred'], f"class_{item['pred']}")
                md.append(f"| {tn} | {pn} | {item['count']} |")
            md.append(f"")

        # Gate error rate bar per class
        most_confused = summary.get('most_confused_classes', [])
        if most_confused:
            md.append(f"### Error Count by Class")
            md.append(f"")
            md.append(f"```")
            max_err = max((c['errors'] for c in most_confused), default=1)
            for c in most_confused:
                bar = '█' * int(c['errors'] / max_err * 30)
                cn = name_of(c['class'], f"class_{c['class']}")
                md.append(f"  {cn:<12} {bar} ({c['errors']})")
            md.append(f"```")
            md.append(f"")

        # Gate error details
        if summary.get('gate_errors'):
            md.append(f"### Gate Error Samples")
            md.append(f"")
            md.append(f"| # | File Path | True | Pred | Error Type | Defect Prob |")
            md.append(f"|---|-----------|------|------|------------|-------------|")
            for i, e in enumerate(summary['gate_errors'][:10]):
                md.append(f"| {i+1} | `{e['path']}` | {e['true_name']} | {e['pred_name']} | {e['error_type']} | {e.get('defect_prob', 0):.4f} |")
            md.append(f"")
            if len(summary['gate_errors']) > 10:
                md.append(f"*... and {len(summary['gate_errors']) - 10} more gate errors (see JSON report)*")
                md.append(f"")

        # Fine error details
        if summary.get('fine_errors'):
            md.append(f"### Fine Error Samples")
            md.append(f"")
            md.append(f"| # | File Path | True Type | Pred Type | Prob |")
            md.append(f"|---|-----------|-----------|-----------|------|")
            for i, e in enumerate(summary['fine_errors'][:10]):
                md.append(f"| {i+1} | `{e['path']}` | {e['true_name']} | {e['pred_name']} | {e.get('pred_prob', 0):.4f} |")
            md.append(f"")
            if len(summary['fine_errors']) > 10:
                md.append(f"*... and {len(summary['fine_errors']) - 10} more fine errors (see JSON report)*")
                md.append(f"")

        # Anomaly detection stats
        if summary.get('anomaly_stats'):
            as_data = summary['anomaly_stats']
            md.append(f"### Anomaly Detection Stats")
            md.append(f"")
            md.append(f"| Stat | Value |")
            md.append(f"|------|-------|")
            md.append(f"| Distance Mean | `{as_data.get('dist_mean', 0):.4f}` |")
            md.append(f"| Distance Std | `{as_data.get('dist_std', 0):.4f}` |")
            md.append(f"| Z-Score Threshold | `{as_data.get('threshold_used', 2.0)}` |")
            md.append(f"")
            md.append(f"> Samples with anomaly score > {as_data.get('threshold_used', 2.0)} σ are flagged as **unknown/novel defects**.")
            md.append(f"")

    # ── 6. Training History ──────────────────────────────────────────────────
    if history:
        md.append(f"## 5. Training History")
        md.append(f"")
        md.append(f"### Loss Curve")
        md.append(f"")
        md.append(f"| Epoch | Train Loss | Val Loss |")
        md.append(f"|-------|------------|---------|")
        for rec in history:
            md.append(f"| {rec['epoch']} | {rec['train_loss']:.4f} | {rec['val_loss']:.4f} |")
        md.append(f"")

        # ASCII loss chart
        train_losses = [r['train_loss'] for r in history]
        val_losses = [r['val_loss'] for r in history]
        max_loss = max(max(train_losses or [1]), max(val_losses or [1]))
        chart_lines = ["```", "Loss (lower is better)"]
        chart_lines.append(f"  Train ▐{'█' * 15} {train_losses[-1] if train_losses else 0:.4f} (final)")
        chart_lines.append(f"  Val   ▐{'▓' * 15} {val_losses[-1] if val_losses else 0:.4f} (final)")
        chart_lines.append(f"  Min   {'─' * 15} {min(val_losses or [0]):.4f} (best)")
        chart_lines.append("  Epoch " + "".join(f"{i%10}" for i in range(1, len(history)+1)))
        chart_lines.append("```")
        md.extend(chart_lines)
        md.append(f"")

        md.append(f"### Accuracy & F1 Per Epoch")
        md.append(f"")
        md.append(f"| Epoch | Gate Acc | Defect Recall | Fine Acc | Macro F1 |")
        md.append(f"|-------|-----------|----------------|-----------|----------|")
        for rec in history:
            md.append(f"| {rec['epoch']} | {rec['gate_accuracy']:.4f} | {rec['gate_defect_recall']:.4f} | "
                      f"{rec['fine_accuracy']:.4f} | {rec['fine_macro_f1']:.4f} |")
        md.append(f"")

        # Error count chart
        md.append(f"### Error Count Per Epoch")
        md.append(f"")
        md.append(f"| Epoch | Gate Errors | Fine Errors |")
        md.append(f"|-------|-------------|-------------|")
        for rec in history:
            md.append(f"| {rec['epoch']} | {rec['gate_errors']} | {rec['fine_errors']} |")
        md.append(f"")

        ge_vals = [r['gate_errors'] for r in history]
        fe_vals = [r['fine_errors'] for r in history]
        max_err = max(max(ge_vals or [1]), max(fe_vals or [1]))
        err_chart = ["```", "Errors (lower is better)"]
        err_chart.append(f"  Gate ▐{'█' * min(int(ge_vals[-1]/max_err*20), 20) if ge_vals else 0} {ge_vals[-1] if ge_vals else 0} (final)")
        err_chart.append(f"  Fine ▐{'▓' * min(int(fe_vals[-1]/max_err*20), 20) if fe_vals else 0} {fe_vals[-1] if fe_vals else 0} (final)")
        err_chart.append("```")
        md.extend(err_chart)
        md.append(f"")

    # ── 7. Recommendations ───────────────────────────────────────────────────
    md.append(f"## 6. Recommendations")
    md.append(f"")
    recommendations = []
    if summary:
        fn_rate = gate.get('defect_as_nuisance_rate', 1.0)
        fp_rate = gate.get('nuisance_as_defect_rate', 1.0)
        if fn_rate > 0.1:
            recommendations.append(f"- 🔴 **高漏检率 ({fn_rate:.1%})**: 增加 defect class 样本，或提高 `defect_weight`")
        if fp_rate > 0.1:
            recommendations.append(f"- 🟡 **高误报率 ({fp_rate:.1%})**: Nuisance 样本可能存在与缺陷相似的纹理，尝试数据增强")
        fine_f1 = fine.get('macro_f1', 0)
        if fine_f1 < 0.7:
            recommendations.append(f"- 🟡 **Fine F1 偏低 ({fine_f1:.1%})**: 缺陷类型之间特征区分度不足，考虑增加 margin-based loss")
        if fine.get('accuracy', 0) < 0.5:
            recommendations.append(f"- 🔴 **Fine Accuracy 过低**: 检查标签映射是否正确（已修复 class_names 类型问题）")
        gate_acc = gate.get('accuracy', 0)
        if gate_acc < 0.8:
            recommendations.append(f"- 🟠 **Gate Accuracy 偏低 ({gate_acc:.1%})**: 二分类边界模糊，增加训练 epochs 或调整学习率")
    if not recommendations:
        recommendations.append(f"- ✅ 模型表现良好，继续监控验证集指标")
    md.extend(recommendations)
    md.append(f"")
    md.append(f"---")
    md.append(f"*Report generated by WaferDefectTrainer · {now}*")

    # Write file
    content = "\n".join(md)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n📄 Markdown report saved → {report_path}")
    return report_path


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
            # defect_type from dataset is 1~K, but update_centers expects 0~(K-1)
            self.model.update_anomaly_centers(all_feats[mask], all_labels[mask] - 1)

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
            anomaly = self.model.anomaly
            # Handle both class-center (AnomalyHead) and RAD (RADAnomalyHead) models
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

    def save_checkpoint(self, path: str, epoch: int, extra: dict = None):
        """Save model checkpoint with metadata."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_names": self.class_names,
        }
        # Save anomaly/RAD stats if present (RAD bank itself is saved separately via rad_bank.pth)
        if hasattr(self.model, 'anomaly'):
            anomaly = self.model.anomaly
            # AnomalyHead: dist_mean/dist_std
            if hasattr(anomaly, 'dist_mean'):
                v = anomaly.dist_mean
                ckpt["anomaly_score_mean"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, 'dist_std'):
                v = anomaly.dist_std
                ckpt["anomaly_score_std"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            # RADAnomalyHead: _score_mean/_score_std
            if hasattr(anomaly, '_score_mean'):
                v = anomaly._score_mean
                ckpt["rad_score_mean"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, '_score_std'):
                v = anomaly._score_std
                ckpt["rad_score_std"] = v.item() if isinstance(v, torch.Tensor) else float(v)
            if hasattr(anomaly, 'anomaly_threshold'):
                ckpt["anomaly_threshold"] = float(anomaly.anomaly_threshold)
            # Flag for RAD mode
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
        # Restore anomaly/RAD stats
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
