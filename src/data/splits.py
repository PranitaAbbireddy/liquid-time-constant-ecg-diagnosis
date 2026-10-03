import numpy as np
from sklearn.model_selection import train_test_split

from src.data.dataset import ECGDataset


def stratified_train_val_split(
    train_csv: str,
    val_fraction: float,
    seed: int,
) -> tuple[ECGDataset, ECGDataset]:
    from src.data.dataset import load_csv_arrays

    signals, labels = load_csv_arrays(train_csv)
    train_x, val_x, train_y, val_y = train_test_split(
        signals,
        labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_ds = ECGDataset(train_csv, signals=train_x, labels=train_y)
    val_ds = ECGDataset(train_csv, signals=val_x, labels=val_y)
    return train_ds, val_ds
