from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.evaluation.thresholds import predict_from_logits
from src.preprocessing.normalize import TrainStatisticsNormalizer


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: TrainStatisticsNormalizer | None = None,
    logit_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            if normalizer is not None:
                signals = normalizer.transform_tensor(signals)
            logits = model(signals).cpu().numpy()
            if logit_bias is not None:
                logits = logits + logit_bias.reshape(1, -1)
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            preds = logits.argmax(axis=1)
            all_logits.append(logits)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
        np.vstack(all_logits),
    )


def _pr_auc_scores(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> tuple[float | None, list[float | None]]:
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    per_class: list[float | None] = []
    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            per_class.append(None)
        else:
            per_class.append(float(average_precision_score(y_bin[:, i], y_prob[:, i])))
    valid = [v for v in per_class if v is not None]
    macro_pr = float(np.mean(valid)) if valid else None
    return macro_pr, per_class


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        ),
    }

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        metrics["macro_roc_auc"] = float(
            roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        )
        metrics["per_class_roc_auc"] = roc_auc_score(
            y_bin, y_prob, average=None, multi_class="ovr"
        ).tolist()
    except ValueError:
        metrics["macro_roc_auc"] = None
        metrics["per_class_roc_auc"] = None

    macro_pr, per_class_pr = _pr_auc_scores(y_true, y_prob, num_classes)
    metrics["macro_pr_auc"] = macro_pr
    metrics["per_class_pr_auc"] = per_class_pr

    cm = np.array(metrics["confusion_matrix"], dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(cm)
    np.divide(cm, row_sums, out=normalized, where=row_sums > 0)
    metrics["confusion_matrix_normalized"] = normalized.tolist()

    return metrics


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    normalizer: TrainStatisticsNormalizer | None = None,
    logit_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true, y_pred, y_prob, _ = collect_predictions(
        model, loader, device, normalizer, logit_bias
    )
    return compute_metrics(y_true, y_pred, y_prob, num_classes)


def metrics_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
    logit_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    if logit_bias is not None:
        logits = logits + logit_bias.reshape(1, -1)
    y_pred = predict_from_logits(logits)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return compute_metrics(y_true, y_pred, probs, num_classes)
