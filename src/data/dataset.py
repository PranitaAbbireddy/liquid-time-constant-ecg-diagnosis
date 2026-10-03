from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import resolve_path


class ECGDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        signals: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ):
        if signals is not None and labels is not None:
            self.signals = torch.tensor(signals, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            path = resolve_path(str(file_path))
            data = pd.read_csv(path, header=None)
            self.signals = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
            self.labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_csv_arrays(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_path(str(file_path))
    data = pd.read_csv(path, header=None)
    signals = data.iloc[:, :-1].values.astype(np.float32)
    labels = data.iloc[:, -1].values.astype(np.int64)
    return signals, labels
