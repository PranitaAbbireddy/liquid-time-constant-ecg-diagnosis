import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def labels_from_dataset(dataset: Dataset) -> np.ndarray:
    return dataset.labels.numpy()


def build_weighted_sampler(labels: np.ndarray, num_classes: int = 5) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_weights = 1.0 / counts[labels.astype(np.int64)]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True,
    )


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampling: str,
) -> DataLoader:
    if sampling == "weighted":
        labels = labels_from_dataset(dataset)
        sampler = build_weighted_sampler(labels, num_classes=int(labels.max()) + 1)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
