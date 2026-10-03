"""Tune validation logit biases on an existing checkpoint without retraining."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset, build_dataloader
from src.data.splits import stratified_train_val_split
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.thresholds import tune_logit_biases
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["project"]["seed"])
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_path(f"{paths['checkpoints_dir']}/{config['training']['checkpoint_name']}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", config)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = None
    if checkpoint.get("normalizer"):
        normalizer = TrainStatisticsNormalizer.from_dict(checkpoint["normalizer"])

    _, val_ds = stratified_train_val_split(
        paths["train_csv"], config["data"]["val_fraction"], config["project"]["seed"]
    )
    val_loader = build_dataloader(val_ds, config["training"]["batch_size"], shuffle=False)
    test_loader = build_dataloader(ECGDataset(paths["test_csv"]), config["training"]["batch_size"], shuffle=False)

    y_val, _, _, val_logits = collect_predictions(model, val_loader, device, normalizer)
    bias = tune_logit_biases(val_logits, y_val, config["data"]["num_classes"], seed=config["project"]["seed"])

    y_test, y_pred, y_prob, _ = collect_predictions(model, test_loader, device, normalizer, bias)
    metrics = compute_metrics(y_test, y_pred, y_prob, config["data"]["num_classes"])
    print(json.dumps({"logit_bias": bias.tolist(), "test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
