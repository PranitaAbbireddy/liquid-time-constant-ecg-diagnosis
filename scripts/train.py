"""Train Liquid ECG classifier and export artifacts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset, build_dataloader, load_csv_arrays
from src.data.samplers import build_train_dataloader
from src.data.splits import stratified_train_val_split
from src.data.validation import validate_ecg_csv
from src.evaluation.metrics import collect_predictions, compute_metrics, evaluate_loader
from src.evaluation.reports import save_metrics_json, save_per_class_table
from src.evaluation.thresholds import tune_logit_biases
from src.evaluation.visualization import (
    export_evaluation_figures,
    plot_class_distribution,
    plot_sample_beats,
    plot_training_curves,
)
from src.explainability.attributions import (
    plot_integrated_gradients_per_class,
    plot_saliency_comparison,
)
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.training.losses import build_criterion
from src.training.trainer import TrainHistory, _run_epoch
from src.utils.config import ensure_output_dirs, load_config
from src.utils.paths import resolve_path
from src.utils.seed import set_seed


def _class_name_map(config: dict) -> dict[int, str]:
    return {int(k): v for k, v in config["classes"].items()}


def _fit_normalizer(config: dict, train_signals: np.ndarray) -> TrainStatisticsNormalizer | None:
    if not config["data"].get("normalize", False):
        return None
    return TrainStatisticsNormalizer.fit(train_signals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LNN ECG classifier")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for smoke tests",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_output_dirs(config)
    set_seed(config["project"]["seed"])

    paths = config["paths"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    max_batches = args.max_batches
    class_names = _class_name_map(config)

    for rel in (paths["train_csv"], paths["test_csv"]):
        issues = validate_ecg_csv(resolve_path(rel), data_cfg["num_features"], data_cfg["num_classes"])
        if issues:
            raise ValueError("Data validation failed:\n" + "\n".join(issues))

    train_ds, val_ds = stratified_train_val_split(
        paths["train_csv"],
        data_cfg["val_fraction"],
        config["project"]["seed"],
    )
    test_ds = ECGDataset(paths["test_csv"])

    train_signals, train_labels = load_csv_arrays(paths["train_csv"])
    normalizer = _fit_normalizer(config, train_signals)

    train_loader = build_train_dataloader(
        train_ds,
        train_cfg["batch_size"],
        train_cfg["num_workers"],
        train_cfg.get("sampling", "none"),
    )
    val_loader = build_dataloader(
        val_ds, train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )
    test_loader = build_dataloader(
        test_ds, train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"]
    )

    plot_class_distribution(
        train_labels,
        class_names,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )
    plot_sample_beats(
        train_signals,
        train_labels,
        class_names,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    loss_type = train_cfg.get("loss_type", "weighted_ce" if train_cfg.get("use_class_weights", True) else "ce")
    criterion = build_criterion(
        train_ds.labels.numpy(),
        data_cfg["num_classes"],
        device,
        loss_type=loss_type,
        weight_scheme=train_cfg.get("weight_scheme", "inverse"),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    def eval_fn(m, loader, dev, norm):
        return evaluate_loader(m, loader, dev, data_cfg["num_classes"], norm)

    history = TrainHistory()
    best_f1 = -1.0
    checkpoint_path = resolve_path(f"{paths['checkpoints_dir']}/{train_cfg['checkpoint_name']}")
    scaler = torch.cuda.amp.GradScaler() if train_cfg["use_amp"] and device.type == "cuda" else None

    for epoch in range(train_cfg["epochs"]):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            normalizer,
            train_cfg["use_amp"],
            scaler,
            max_batches,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            normalizer,
            train_cfg["use_amp"],
            None,
            max_batches,
        )
        val_metrics = eval_fn(model, val_loader, device, normalizer)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(val_metrics["macro_f1"])
        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalizer": normalizer.to_dict() if normalizer else None,
                    "val_macro_f1": best_f1,
                    "logit_bias": None,
                },
                checkpoint_path,
            )

    plot_training_curves(history, paths["graphs_dir"], config["evaluation"]["figure_dpi"])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("normalizer"):
        normalizer = TrainStatisticsNormalizer.from_dict(checkpoint["normalizer"])

    y_val, _, _, val_logits = collect_predictions(model, val_loader, device, normalizer)
    logit_bias = None
    if train_cfg.get("tune_logit_bias", True):
        logit_bias = tune_logit_biases(
            val_logits, y_val, data_cfg["num_classes"], seed=config["project"]["seed"]
        )

    y_val, y_pred_val, y_prob_val, _ = collect_predictions(
        model, val_loader, device, normalizer, logit_bias
    )
    val_metrics_final = compute_metrics(
        y_val, y_pred_val, y_prob_val, data_cfg["num_classes"]
    )

    y_true, y_pred, y_prob, _ = collect_predictions(
        model, test_loader, device, normalizer, logit_bias
    )
    test_metrics = compute_metrics(y_true, y_pred, y_prob, data_cfg["num_classes"])

    checkpoint["logit_bias"] = logit_bias.tolist() if logit_bias is not None else None
    checkpoint["logit_bias_fit_split"] = "validation"
    checkpoint["logit_bias_tuned_on_test"] = False
    checkpoint["val_metrics_with_bias"] = val_metrics_final
    torch.save(checkpoint, checkpoint_path)

    save_metrics_json(test_metrics, paths["reports_dir"], "test_metrics.json")
    save_per_class_table(test_metrics, class_names, paths["reports_dir"], "per_class_metrics.csv")
    export_evaluation_figures(
        test_metrics,
        y_true,
        y_prob,
        class_names,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )

    exp_cfg = config["explainability"]
    plot_integrated_gradients_per_class(
        model,
        test_loader,
        class_names,
        device,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
        exp_cfg["ig_steps"],
        exp_cfg["samples_per_class"],
        exp_cfg["attribution_sample_seed"],
        normalizer,
    )
    plot_saliency_comparison(
        model,
        test_loader,
        class_names,
        device,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
        exp_cfg["ig_steps"],
        exp_cfg["attribution_sample_seed"],
        normalizer,
    )

    summary = {
        "parameters": model.count_parameters(),
        "model_selection": {
            "checkpoint_criterion": "validation macro-F1 (argmax, per epoch)",
            "best_val_macro_f1_checkpoint_argmax": best_f1,
            "validation_metrics_after_frozen_bias": val_metrics_final,
        },
        "final_test_report": {
            "note": "Single held-out evaluation; not used for checkpoint or bias selection.",
            "metrics": test_metrics,
        },
        "checkpoint": str(checkpoint_path),
    }
    summary_path = resolve_path(f"{paths['reports_dir']}/summaries/run_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
