import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_labels(
    labels: np.ndarray,
    num_classes: int,
    scheme: str = "inverse",
) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if scheme == "sqrt_inverse":
        weights = np.sqrt(counts.sum() / (num_classes * counts))
    else:
        weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    loss_type: str = "weighted_ce",
    weight_scheme: str = "inverse",
    focal_gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "focal":
        weight = None
        if weight_scheme != "none":
            weight = class_weights_from_labels(labels, num_classes, weight_scheme).to(device)
        return FocalLoss(gamma=focal_gamma, weight=weight)
    weight = class_weights_from_labels(labels, num_classes, weight_scheme).to(device)
    return nn.CrossEntropyLoss(weight=weight)
