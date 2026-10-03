from copy import deepcopy
from typing import Any

import numpy as np
import torch

from src.data.dataset import ECGDataset, build_dataloader
from src.data.samplers import build_train_dataloader
from src.data.splits import stratified_train_val_split
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.thresholds import tune_logit_biases
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.training.losses import build_criterion
from src.training.trainer import _run_epoch


def _fit_normalizer(config: dict, train_signals: np.ndarray) -> TrainStatisticsNormalizer | None:
    if not config["data"].get("normalize", False):
        return None
    return TrainStatisticsNormalizer.fit(train_signals)


def train_one_experiment(
    config: dict,
    experiment: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
    evaluate_test_diagnostic: bool = False,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["model"].update(experiment.get("model", {}))
    cfg["data"].update(experiment.get("data", {}))
    cfg["training"].update(experiment.get("training", {}))

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    seed = cfg["project"]["seed"]

    train_ds, val_ds = stratified_train_val_split(
        paths["train_csv"],
        data_cfg["val_fraction"],
        seed,
    )
    train_labels = train_ds.labels.numpy()
    train_signals = train_ds.signals.numpy()
    normalizer = _fit_normalizer(cfg, train_signals)

    train_loader = build_train_dataloader(
        train_ds,
        train_cfg["batch_size"],
        train_cfg["num_workers"],
        train_cfg.get("sampling", "none"),
    )
    val_loader = build_dataloader(
        val_ds, train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )
    model = build_model(cfg).to(device)
    criterion = build_criterion(
        train_labels,
        data_cfg["num_classes"],
        device,
        loss_type=train_cfg.get("loss_type", "weighted_ce"),
        weight_scheme=train_cfg.get("weight_scheme", "inverse"),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler() if train_cfg.get("use_amp", True) and device.type == "cuda" else None

    best_state = None
    best_val_f1 = -1.0
    epochs = train_cfg["epochs"]

    for epoch in range(epochs):
        _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            normalizer,
            train_cfg.get("use_amp", True),
            scaler,
            max_batches,
        )
        val_metrics = evaluate_model(model, val_loader, device, data_cfg["num_classes"], normalizer)
        print(
            f"[{experiment['name']}] epoch {epoch + 1}/{epochs} val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val, _, _, val_logits = collect_predictions(
        model, val_loader, device, normalizer, logit_bias=None
    )
    logit_bias = None
    if experiment.get("tune_logit_bias", True):
        logit_bias = tune_logit_biases(val_logits, y_val, data_cfg["num_classes"], seed=seed)

    y_val_sel, y_pred_val, y_prob_val, _ = collect_predictions(
        model, val_loader, device, normalizer, logit_bias=logit_bias
    )
    val_metrics = compute_metrics(y_val_sel, y_pred_val, y_prob_val, data_cfg["num_classes"])

    test_metrics_diagnostic = None
    if evaluate_test_diagnostic:
        test_ds = ECGDataset(paths["test_csv"])
        test_loader = build_dataloader(
            test_ds, train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
        )
        y_test, y_pred, y_prob, _ = collect_predictions(
            model, test_loader, device, normalizer, logit_bias=logit_bias
        )
        test_metrics_diagnostic = compute_metrics(y_test, y_pred, y_prob, data_cfg["num_classes"])

    return {
        "name": experiment["name"],
        "config": {
            "model": cfg["model"],
            "data": cfg["data"],
            "training": {
                k: train_cfg[k]
                for k in (
                    "loss_type",
                    "weight_scheme",
                    "sampling",
                    "focal_gamma",
                    "batch_size",
                    "epochs",
                    "learning_rate",
                )
                if k in train_cfg
            },
            "tune_logit_bias": experiment.get("tune_logit_bias", True),
        },
        "best_val_macro_f1_checkpoint_argmax": best_val_f1,
        "val_metrics": val_metrics,
        "logit_bias": logit_bias.tolist() if logit_bias is not None else None,
        "logit_bias_fit_split": "validation",
        "test_metrics_diagnostic": test_metrics_diagnostic,
        "model_state_dict": best_state,
        "normalizer": normalizer.to_dict() if normalizer else None,
        "full_config": cfg,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    normalizer: TrainStatisticsNormalizer | None,
) -> dict:
    from src.evaluation.metrics import evaluate_loader

    return evaluate_loader(model, loader, device, num_classes, normalizer)
