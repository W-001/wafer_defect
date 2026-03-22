"""
Training engine for wafer defect classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import CombinedLoss
from ..utils.metrics import GateMetrics, FineMetrics


class WaferDefectTrainer:
    """
    Trainer for wafer defect classification model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        gate_weight: float = 1.0,
        fine_weight: float = 0.5,
        metric_weight: float = 0.1,
        defect_weight: float = 3.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gate_weight = gate_weight
        self.fine_weight = fine_weight
        self.metric_weight = metric_weight
        self.defect_weight = defect_weight

        self.criterion = CombinedLoss(
            gate_weight=gate_weight,
            fine_weight=fine_weight,
            metric_weight=metric_weight,
            defect_weight=defect_weight
        )

        self.gate_metrics = GateMetrics()
        self.fine_metrics = FineMetrics(num_classes=model.num_defect_classes)

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
            # defect_type is -1 for nuisance, 1~K for defects
            # We want to classify only defects, so nuisance should be excluded from fine loss
            defect_type_adjusted = defect_type.clone()
            mask = defect_type > 0
            defect_type_adjusted[mask] = defect_type[mask] - 1  # 1~K -> 0~(K-1)

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
            self.model.update_anomaly_centers(
                all_feats[mask],
                all_labels[mask]
            )

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()

        gate_metrics = GateMetrics()
        fine_metrics = FineMetrics(num_classes=self.model.num_defect_classes)

        total_loss = 0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"].to(self.device)
            label = batch["label"].to(self.device)
            is_defect = batch["is_defect"].to(self.device)
            defect_type = batch["defect_type"].to(self.device)

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

            total_loss += loss.item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "gate_metrics": gate_metrics.compute(),
            "fine_metrics": fine_metrics.compute()
        }

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
