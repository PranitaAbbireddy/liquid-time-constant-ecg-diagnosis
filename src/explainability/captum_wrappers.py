import torch
import torch.nn as nn

from src.preprocessing.normalize import TrainStatisticsNormalizer


class ECGCaptumWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        normalizer: TrainStatisticsNormalizer | None = None,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            x = self.normalizer.transform_tensor(x)
        return self.model(x)
