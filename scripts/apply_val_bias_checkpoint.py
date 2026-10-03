"""Apply validation-tuned logit bias; report val metrics; optional single test report."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset, build_dataloader
from src.data.splits import stratified_train_val_split
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.reports import save_metrics_json, save_per_class_table
from src.evaluation.thresholds import tune_logit_biases
from src.evaluation.visualization import export_evaluation_figures
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-test-report",
        action="store_true",
        help="Run one held-out test evaluation after freezing validation-derived bias.",
    )
    args = parser.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    paths = config["paths"]
    class_names = {int(k): v for k, v in config["classes"].items()}
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

    y_val, _, _, val_logits = collect_predictions(model, val_loader, device, normalizer, logit_bias=None)
    bias = tune_logit_biases(
        val_logits, y_val, config["data"]["num_classes"], seed=config["project"]["seed"]
    )

    y_val, y_pred_val, y_prob_val, _ = collect_predictions(model, val_loader, device, normalizer, bias)
    val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val, config["data"]["num_classes"])

    checkpoint["logit_bias"] = bias.tolist()
    checkpoint["logit_bias_fit_split"] = "validation"
    checkpoint["logit_bias_tuned_on_test"] = False
    checkpoint["val_metrics_with_bias"] = val_metrics
    torch.save(checkpoint, ckpt_path)

    experiment_record = {
        "name": "checkpoint_val_logit_bias_posthoc",
        "selectable": True,
        "config": {"tune_logit_bias": True, "training": cfg.get("training", {})},
        "val_metrics": val_metrics,
        "logit_bias": bias.tolist(),
        "logit_bias_fit_split": "validation",
        "test_metrics_diagnostic": None,
    }
    exp_path = resolve_path("reports/experiments/checkpoint_val_logit_bias_posthoc.json")
    exp_path.write_text(json.dumps(experiment_record, indent=2), encoding="utf-8")

    # Post-hoc bias on checkpoints/best_model.pt is not the validation-selected
    # experiment. Official provenance is written by finalize_selected_model.py.

    if args.write_test_report:
        test_loader = build_dataloader(
            ECGDataset(paths["test_csv"]), config["training"]["batch_size"], shuffle=False
        )
        y_test, y_pred, y_prob, _ = collect_predictions(model, test_loader, device, normalizer, bias)
        test_metrics = compute_metrics(y_test, y_pred, y_prob, config["data"]["num_classes"])
        experiment_record["test_metrics_diagnostic"] = test_metrics
        exp_path.write_text(json.dumps(experiment_record, indent=2), encoding="utf-8")

        save_metrics_json(test_metrics, paths["reports_dir"], "test_metrics.json")
        save_per_class_table(test_metrics, class_names, paths["reports_dir"], "per_class_metrics.csv")
        export_evaluation_figures(
            test_metrics, y_test, y_prob, class_names, paths["graphs_dir"], config["evaluation"]["figure_dpi"]
        )
        final_report = {
            "note": "Single held-out test report; not used to tune bias or select experiments.",
            "validation_metrics_at_freeze": val_metrics,
            "test_metrics": test_metrics,
            "logit_bias": bias.tolist(),
        }
        resolve_path(f"{paths['reports_dir']}/summaries/final_test_report.json").write_text(
            json.dumps(final_report, indent=2),
            encoding="utf-8",
        )

    summary = {
        "model_selection": {
            "validation_metrics": val_metrics,
            "logit_bias_fit_split": "validation",
        },
        "final_test_report_written": args.write_test_report,
        "checkpoint": str(ckpt_path),
    }
    resolve_path(f"{paths['reports_dir']}/summaries/run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
