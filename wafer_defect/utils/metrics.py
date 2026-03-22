"""
Evaluation metrics for wafer defect classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


class GateMetrics:
    """
    Metrics for Gate (Nuisance vs Defect) classification.
    Focus on the critical business constraint.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ):
        """Update with batch results."""
        probs = torch.softmax(logits, dim=-1)[:, 1]  # probability of defect
        preds = (probs > 0.5).long()

        self.predictions.extend(preds.cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
        self.probabilities.extend(probs.cpu().tolist())

    def compute(self) -> dict:
        """Compute metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        probs = np.array(self.probabilities)

        # Basic metrics
        accuracy = accuracy_score(targets, preds)

        # Precision, Recall, F1 for each class
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average=None, labels=[0, 1]
        )

        # Confusion matrix
        cm = confusion_matrix(targets, preds, labels=[0, 1])

        # Specific metrics
        nuisance_recall = recall[0]  # How well we detect actual Nuisance
        defect_recall = recall[1]    # How well we detect actual Defects

        # Misclassification rates
        nuisance_as_defect = cm[0, 1] / (cm[0].sum() + 1e-8)  # False positive
        defect_as_nuisance = cm[1, 0] / (cm[1].sum() + 1e-8)  # False negative (critical!)

        # AUC if possible
        try:
            auc = roc_auc_score(targets, probs)
        except:
            auc = None

        return {
            "accuracy": accuracy,
            "nuisance_recall": nuisance_recall,
            "defect_recall": defect_recall,
            "nuisance_as_defect_rate": nuisance_as_defect,
            "defect_as_nuisance_rate": defect_as_nuisance,
            "auc": auc,
            "confusion_matrix": cm.tolist()
        }


class FineMetrics:
    """
    Metrics for Fine (defect type) classification.
    Reports per-class and macro metrics.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update with batch results."""
        preds = logits.argmax(dim=-1)
        self.predictions.extend(preds.cpu().tolist())
        self.targets.extend(targets.cpu().tolist())

    def compute(self) -> dict:
        """Compute metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        accuracy = accuracy_score(targets, preds)

        # Macro metrics (average over classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                targets, preds, average=None, zero_division=0
            )
        )

        # Classification report as string
        report = classification_report(
            targets, preds, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "per_class_precision": precision_per_class.tolist(),
            "per_class_recall": recall_per_class.tolist(),
            "per_class_f1": f1_per_class.tolist(),
            "classification_report": report
        }


class AnomalyMetrics:
    """
    Metrics for anomaly (unknown defect) detection.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = []
        self.is_anomaly = []

    def update(
        self,
        scores: torch.Tensor,
        threshold: float
    ):
        """Update with batch results."""
        self.scores.extend(scores.cpu().tolist())
        self.is_anomaly.extend((scores > threshold).cpu().tolist())

    def compute(self, scores: np.ndarray = None) -> dict:
        """Compute metrics."""
        if scores is None:
            scores = np.array(self.scores)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max()
        }
