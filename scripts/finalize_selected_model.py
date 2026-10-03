"""Run a single held-out test evaluation for the validation-selected experiment."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset, build_dataloader
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.reports import save_metrics_json, save_per_class_table, write_logit_bias_provenance
from src.evaluation.visualization import export_evaluation_figures
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> None:
    config = load_config()
    paths = config["paths"]
    class_names = {int(k): v for k, v in config["classes"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection_path = resolve_path("reports/summaries/model_selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    experiment_name = selection.get("selected_experiment")
    if not experiment_name:
        raise ValueError("No selected_experiment in model_selection.json; run build_experiment_table.py first.")

    ckpt_path = resolve_path(f"{paths['checkpoints_dir']}/experiments/{experiment_name}.pt")
    if not ckpt_path.exists():
        ckpt_path = resolve_path(f"{paths['checkpoints_dir']}/{config['training']['checkpoint_name']}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", config)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    normalizer = None
    if checkpoint.get("normalizer"):
        normalizer = TrainStatisticsNormalizer.from_dict(checkpoint["normalizer"])

    bias = checkpoint.get("logit_bias")
    import numpy as np

    logit_bias = np.array(bias, dtype=np.float32) if bias is not None else None

    test_loader = build_dataloader(
        ECGDataset(paths["test_csv"]),
        cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    y_test, y_pred, y_prob, _ = collect_predictions(model, test_loader, device, normalizer, logit_bias)
    test_metrics = compute_metrics(y_test, y_pred, y_prob, config["data"]["num_classes"])

    save_metrics_json(test_metrics, paths["reports_dir"], "test_metrics.json")
    save_per_class_table(test_metrics, class_names, paths["reports_dir"], "per_class_metrics.csv")
    export_evaluation_figures(
        test_metrics,
        y_test,
        y_prob,
        class_names,
        paths["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )

    final_report = {
        "note": "Single unbiased held-out test evaluation for the validation-selected model.",
        "selected_experiment": experiment_name,
        "validation_metrics_used_for_selection": selection.get("validation_metrics"),
        "test_metrics": test_metrics,
        "logit_bias": bias,
        "logit_bias_fit_split": checkpoint.get("logit_bias_fit_split", "validation"),
    }
    final_path = resolve_path(f"{paths['reports_dir']}/summaries/final_test_report.json")
    final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    write_logit_bias_provenance(selection, experiment_name, bias, checkpoint.get("logit_bias_fit_split", "validation"))

    run_summary = {
        "model_selection": selection,
        "final_test_report_path": str(final_path),
        "checkpoint": str(ckpt_path),
    }
    resolve_path(f"{paths['reports_dir']}/summaries/run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
