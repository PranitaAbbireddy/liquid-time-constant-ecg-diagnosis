from pathlib import Path

import numpy as np
import pandas as pd


def validate_ecg_csv(
    file_path: Path,
    num_features: int = 187,
    num_classes: int = 5,
) -> list[str]:
    issues: list[str] = []
    df = pd.read_csv(file_path, header=None)

    expected_cols = num_features + 1
    if df.shape[1] != expected_cols:
        issues.append(f"{file_path.name}: expected {expected_cols} columns, got {df.shape[1]}")

    if df.isna().any().any():
        issues.append(f"{file_path.name}: contains NaN values")

    labels = df.iloc[:, -1]
    if not np.issubdtype(labels.dtype, np.number):
        issues.append(f"{file_path.name}: labels are not numeric")

    unique_labels = sorted(labels.unique())
    invalid = [int(x) for x in unique_labels if x < 0 or x >= num_classes]
    if invalid:
        issues.append(f"{file_path.name}: labels outside [0, {num_classes - 1}]: {invalid}")

    return issues
