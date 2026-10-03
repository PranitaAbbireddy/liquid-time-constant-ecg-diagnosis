import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import resolve_path


def write_logit_bias_provenance(
    selection: dict[str, Any],
    experiment_name: str,
    logit_bias: Any = None,
    fit_split: str = "validation",
) -> Path:
    """Record frozen-bias validation metrics for the validation-selected experiment."""
    exp_path = resolve_path(f"reports/experiments/{experiment_name}.json")
    val_metrics = selection.get("validation_metrics")
    if exp_path.exists():
        experiment = json.loads(exp_path.read_text(encoding="utf-8"))
        val_metrics = experiment.get("val_metrics") or val_metrics
        if logit_bias is None:
            logit_bias = experiment.get("logit_bias")
        fit_split = experiment.get("logit_bias_fit_split", fit_split)

    note = {
        "selected_experiment": experiment_name,
        "logit_bias_source": "validation split only",
        "logit_bias_optimization_labels": "y_val from stratified train/val split of mitbih_train.csv",
        "test_labels_used_for_bias_tuning": False,
        "logit_bias": logit_bias,
        "logit_bias_fit_split": fit_split,
        "validation_metrics_with_frozen_bias": val_metrics,
    }
    path = resolve_path("reports/summaries/logit_bias_provenance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(note, indent=2), encoding="utf-8")
    return path


def save_metrics_json(metrics: dict[str, Any], reports_dir: str, filename: str) -> Path:
    out = resolve_path(f"{reports_dir}/metrics/{filename}")
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: v
        for k, v in metrics.items()
        if k != "classification_report"
    }
    serializable["classification_report"] = metrics.get("classification_report", "")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return out


def save_per_class_table(
    metrics: dict[str, Any],
    class_names: dict[int, str],
    reports_dir: str,
    filename: str,
) -> Path:
    rows = []
    for idx in sorted(class_names.keys()):
        rows.append(
            {
                "class_id": idx,
                "class_name": class_names[idx],
                "precision": metrics["per_class_precision"][idx],
                "recall": metrics["per_class_recall"][idx],
                "f1": metrics["per_class_f1"][idx],
            }
        )
    df = pd.DataFrame(rows)
    out = resolve_path(f"{reports_dir}/tables/{filename}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
