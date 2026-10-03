"""Evaluate a saved checkpoint on the held-out test set."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset, build_dataloader
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.reports import save_metrics_json, save_per_class_table
from src.evaluation.visualization import export_evaluation_figures
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> None:
    config = load_config()
    paths = config["paths"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    class_names = {int(k): v for k, v in config["classes"].items()}

    checkpoint_path = resolve_path(f"{paths['checkpoints_dir']}/{train_cfg['checkpoint_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", config)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    normalizer = None
    if checkpoint.get("normalizer"):
        normalizer = TrainStatisticsNormalizer.from_dict(checkpoint["normalizer"])

    bias = checkpoint.get("logit_bias")
    logit_bias = np.array(bias, dtype=np.float32) if bias is not None else None

    test_loader = build_dataloader(
        ECGDataset(paths["test_csv"]),
        train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    y_true, y_pred, y_prob, _ = collect_predictions(model, test_loader, device, normalizer, logit_bias)
    metrics = compute_metrics(y_true, y_pred, y_prob, data_cfg["num_classes"])

    save_metrics_json(metrics, paths["reports_dir"], "test_metrics.json")
    save_per_class_table(metrics, class_names, paths["reports_dir"], "per_class_metrics.csv")
    export_evaluation_figures(
        metrics,
        y_true,
        y_prob,
        class_names,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )

    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall")}, indent=2))


if __name__ == "__main__":
    main()
