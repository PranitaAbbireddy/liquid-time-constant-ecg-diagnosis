import numpy as np
import torch


class TrainStatisticsNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std if std > 1e-8 else 1.0

    @classmethod
    def fit(cls, signals: np.ndarray) -> "TrainStatisticsNormalizer":
        mean = float(np.mean(signals))
        std = float(np.std(signals))
        return cls(mean=mean, std=std)

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "TrainStatisticsNormalizer":
        return cls(mean=data["mean"], std=data["std"])
