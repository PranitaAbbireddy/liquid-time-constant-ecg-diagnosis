from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.preprocessing.normalize import TrainStatisticsNormalizer


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    normalizer: TrainStatisticsNormalizer | None,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    max_batches: int | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (signals, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        signals = signals.to(device)
        labels = labels.to(device)
        if normalizer is not None:
            signals = normalizer.transform_tensor(signals)

        with torch.set_grad_enabled(is_train):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(signals)
                    loss = criterion(logits, labels)
            else:
                logits = model(signals)
                loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    normalizer: TrainStatisticsNormalizer | None,
    use_amp: bool,
    evaluate_fn,
) -> TrainHistory:
    history = TrainHistory()
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    for _ in range(epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            normalizer,
            use_amp,
            scaler,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            normalizer,
            use_amp,
            None,
        )
        metrics = evaluate_fn(model, val_loader, device, normalizer)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(metrics["macro_f1"])

    return history
