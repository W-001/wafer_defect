"""
Long-tail distribution samplers for imbalanced wafer defect datasets.

Implements several sampling strategies to handle class imbalance:
1. LongTailSampler: Samples inversely proportional to class frequency
2. BalancedBatchSampler: Ensures balanced classes within each batch
3. SquareRootSampler: Square-root frequency sampling (mild balance)

Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from typing import Iterator, List, Optional
from collections import Counter


class LongTailSampler(Sampler):
    """
    Long-tail sampler that samples inversely proportional to class frequency.

    For datasets with severe class imbalance (common in defect detection),
    this sampler helps balance the representation of minority classes.

    Sampling probability for class c:
        p(c) \propto (1 / n_c)^beta

    where n_c is the number of samples in class c, and beta controls
    the strength of rebalancing (0 < beta <= 1).

    Args:
        labels: List or array of class labels for all samples
        beta: Rebalancing exponent (default: 0.9999, close to inverse frequency)
              - beta=0: uniform sampling (no rebalancing)
              - beta=1: full inverse frequency sampling
        num_samples: Total number of samples to draw per epoch
        generator: Optional random generator for reproducibility
    """

    def __init__(
        self,
        labels: List[int],
        beta: float = 0.9999,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if beta < 0 or beta > 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.beta = beta
        self.num_samples = num_samples
        self.generator = generator

        # Compute class frequencies
        self.class_counts = Counter(labels)
        self.num_classes = len(self.class_counts)
        self.total_samples = len(labels)

        # If num_samples not specified, use full dataset
        if self.num_samples is None:
            self.num_samples = self.total_samples

        # Compute effective number of samples per class (from paper)
        # effective_num = (1 - beta^n) / (1 - beta)
        self.effective_nums = {}
        self.class_weights = {}
        self.sample_class_weights = []

        for c in range(max(labels) + 1 if isinstance(labels, list) else max(labels) + 1):
            n_c = self.class_counts.get(c, 0)
            if n_c > 0:
                effective_num = (1 - math.pow(self.beta, n_c)) / (1 - self.beta + 1e-8)
                self.effective_nums[c] = effective_num
                self.class_weights[c] = 1.0 / (effective_num + 1e-8)
            else:
                self.effective_nums[c] = 0
                self.class_weights[c] = 0

        # Normalize class weights
        total_weight = sum(self.class_weights.values())
        if total_weight > 0:
            for c in self.class_weights:
                self.class_weights[c] /= total_weight

        # Assign sample weights based on their class
        self.sample_weights = [self.class_weights.get(label, 0) for label in labels]
        self.labels = labels

    def __iter__(self) -> Iterator[int]:
        # Build weighted sampling indices
        indices = list(range(self.total_samples))
        weights = self.sample_weights

        # Weighted random sampling without replacement
        # Use torch for efficient weighted sampling
        if sum(weights) > 0:
            weights_tensor = torch.tensor(weights, dtype=torch.double)
            # Normalize to get probabilities
            probs = weights_tensor / weights_tensor.sum()

            # Sample indices according to weights
            sampled_indices = torch.multinomial(
                probs,
                num_samples=min(self.num_samples, self.total_samples),
                replacement=False,
                generator=self.generator
            ).tolist()
        else:
            # Fallback to uniform sampling
            indices_tensor = torch.randperm(self.total_samples, generator=self.generator)
            sampled_indices = indices_tensor[:min(self.num_samples, self.total_samples)].tolist()

        return iter(sampled_indices)

    def __len__(self) -> int:
        return self.num_samples

    def get_class_distribution(self) -> dict:
        """Return the target class distribution after rebalancing."""
        sampled_labels = [self.labels[i] for i in self]
        return dict(Counter(sampled_labels))


class BalancedBatchSampler(Sampler):
    """
    Batch-level balanced sampler.

    Ensures each batch has roughly equal representation from all classes,
    with optional quota per class.

    Args:
        labels: List of class labels
        batch_size: Size of each batch
        num_classes: Total number of classes
        samples_per_class: How many samples per class in each batch
            If None, automatically computed to fill batch_size
        drop_last: Whether to drop incomplete last batch
        generator: Random generator
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        num_classes: int,
        samples_per_class: Optional[int] = None,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.drop_last = drop_last
        self.generator = generator

        # Group indices by class
        self.class_indices = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(labels):
            if label in self.class_indices:
                self.class_indices[label].append(idx)

        # Sort indices within each class for reproducibility
        for c in self.class_indices:
            self.class_indices[c].sort()

        # Compute samples per class
        if samples_per_class is None:
            # Fill batch with samples from different classes
            self.samples_per_class = max(1, batch_size // num_classes)
        else:
            self.samples_per_class = samples_per_class

        # Compute number of batches
        self.num_batches = len(labels) // batch_size
        if not drop_last and len(labels) % batch_size != 0:
            self.num_batches += 1

        # Number of complete cycles per class per epoch
        self.cycles_per_class = {}
        for c in range(num_classes):
            class_len = len(self.class_indices[c])
            if class_len > 0 and self.samples_per_class > 0:
                self.cycles_per_class[c] = max(1, class_len // self.samples_per_class)
            else:
                self.cycles_per_class[c] = 1

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        rng = np.random.default_rng()
        if self.generator is not None and hasattr(self.generator, 'initial_seed'):
            rng = np.random.default_rng(self.generator.initial_seed())

        for c in self.class_indices:
            rng.shuffle(self.class_indices[c])

        # Build batch indices
        batch_indices = []
        current_batch = []

        # Round-robin through classes
        class_pointers = {c: 0 for c in range(self.num_classes)}

        while len(batch_indices) < self.num_batches * self.batch_size:
            for c in range(self.num_classes):
                indices = self.class_indices[c]
                if len(indices) == 0:
                    continue

                # Add samples from this class
                for _ in range(self.samples_per_class):
                    idx = class_pointers[c]
                    if idx < len(indices):
                        current_batch.append(indices[idx])
                        class_pointers[c] += 1

                        if len(current_batch) == self.batch_size:
                            batch_indices.extend(current_batch)
                            current_batch = []

                        # Check if we've collected enough batches
                        if len(batch_indices) >= self.num_batches * self.batch_size:
                            break

                if len(batch_indices) >= self.num_batches * self.batch_size:
                    break

        # Shuffle the order of batches
        rng.shuffle(batch_indices)

        # Yield batch boundaries
        for i in range(0, len(batch_indices), self.batch_size):
            if self.drop_last and i + self.batch_size > len(batch_indices):
                break
            yield batch_indices[i:i + self.batch_size]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


class SquareRootSampler(Sampler):
    """
    Square-root frequency sampler.

    Sampling probability:
        p(c) \propto sqrt(n_c)

    This is a milder form of rebalancing than LongTailSampler,
    useful when you don't want to over-sample minority classes too aggressively.

    Args:
        labels: List of class labels
        num_samples: Number of samples to draw per epoch
        generator: Random generator
    """

    def __init__(
        self,
        labels: List[int],
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = labels
        self.num_samples = num_samples if num_samples else len(labels)
        self.generator = generator

        # Compute class frequencies
        self.class_counts = Counter(labels)
        self.total_samples = len(labels)

        # Compute square-root weights
        self.class_weights = {}
        for c, n_c in self.class_counts.items():
            self.class_weights[c] = math.sqrt(n_c)

        # Normalize
        total_weight = sum(self.class_weights.values())
        if total_weight > 0:
            for c in self.class_weights:
                self.class_weights[c] /= total_weight

        # Sample weights
        self.sample_weights = [self.class_weights.get(label, 0) for label in labels]

    def __iter__(self) -> Iterator[int]:
        weights_tensor = torch.tensor(self.sample_weights, dtype=torch.double)
        probs = weights_tensor / weights_tensor.sum()

        sampled = torch.multinomial(
            probs,
            num_samples=min(self.num_samples, self.total_samples),
            replacement=False,
            generator=self.generator
        ).tolist()

        return iter(sampled)

    def __len__(self) -> int:
        return self.num_samples


def create_longtail_dataloader(
    dataset,
    labels: List[int],
    batch_size: int,
    beta: float = 0.9999,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with LongTailSampler for long-tail distribution.

    Args:
        dataset: PyTorch dataset
        labels: Class labels for each sample in the dataset
        batch_size: Batch size
        beta: Rebalancing exponent (0-1)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop incomplete last batch
        shuffle: Whether to shuffle (if False, uses weighted sampling order)

    Returns:
        DataLoader with LongTailSampler
    """
    sampler = LongTailSampler(
        labels=labels,
        beta=beta,
        num_samples=len(dataset),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler if shuffle else None,
        shuffle=shuffle if not shuffle else False,  # Only use shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def compute_class_weights(
    labels: List[int],
    method: str = "effective_number",
    beta: float = 0.9999,
) -> torch.Tensor:
    """
    Compute class weights for weighted loss functions.

    Args:
        labels: Class labels
        method: Weighting method ("effective_number", "sqrt", "log", "balanced")
        beta: Exponent for effective_number method

    Returns:
        Tensor of class weights [num_classes]
    """
    class_counts = Counter(labels)
    num_classes = max(labels) + 1
    total = len(labels)

    weights = torch.zeros(num_classes)

    for c in range(num_classes):
        n_c = class_counts.get(c, 0)

        if method == "effective_number":
            # From "Class-Balanced Loss" paper
            effective_num = (1 - math.pow(beta, n_c)) / (1 - beta + 1e-8)
            weights[c] = 1.0 / (effective_num + 1e-8)
        elif method == "sqrt":
            # Square root weighting
            weights[c] = math.sqrt(total / (n_c + 1e-8))
        elif method == "log":
            # Logarithmic weighting
            weights[c] = math.log(total / (n_c + 1e-8) + 1)
        elif method == "balanced":
            # Balanced weighting: n_samples / (n_classes * n_c)
            weights[c] = total / (num_classes * (n_c + 1e-8))
        else:
            raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights
