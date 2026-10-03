"""Build experiment comparison and validation-only model selection artifacts."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reports import write_logit_bias_provenance
from src.evaluation.selection import select_best_experiment
from src.utils.paths import resolve_path

SELECTION_NOTE = (
    "Model/experiment chosen using validation metrics only. "
    "Test metrics in this table are diagnostic/reference when present and are not used for selection."
)


def _strategy_from_config(cfg: dict) -> str:
    training = cfg.get("training", {})
    return (
        f"loss={training.get('loss_type')}, weights={training.get('weight_scheme')}, "
        f"sampling={training.get('sampling')}, val_bias={cfg.get('tune_logit_bias')}"
    )


def _metrics_columns(prefix: str, metrics: dict) -> dict:
    cols = {
        f"{prefix}_accuracy": metrics.get("accuracy"),
        f"{prefix}_macro_precision": metrics.get("macro_precision"),
        f"{prefix}_macro_recall": metrics.get("macro_recall"),
        f"{prefix}_macro_f1": metrics.get("macro_f1"),
        f"{prefix}_weighted_f1": metrics.get("weighted_f1"),
        f"{prefix}_macro_roc_auc": metrics.get("macro_roc_auc"),
        f"{prefix}_macro_pr_auc": metrics.get("macro_pr_auc"),
    }
    for i, f1 in enumerate(metrics.get("per_class_f1") or []):
        cols[f"{prefix}_f1_class_{i}"] = f1
    return cols


def _row_from_experiment(data: dict) -> dict:
    cfg = data.get("config", {})
    row = {
        "experiment": data.get("name"),
        "strategy": _strategy_from_config(cfg),
        "selectable": data.get("selectable", True),
    }
    if data.get("val_metrics"):
        row.update(_metrics_columns("val", data["val_metrics"]))
    test_diag = data.get("test_metrics_diagnostic") or data.get("test_metrics")
    if test_diag:
        row.update(_metrics_columns("test_diagnostic", test_diag))
    return row


def load_experiment_records() -> list[dict]:
    records: list[dict] = []
    exp_dir = resolve_path("reports/experiments")
    for path in sorted(exp_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "baseline_snapshot.json":
            data["selectable"] = False
            data["name"] = data.get("name", "baseline_snapshot")
            data["test_metrics_diagnostic"] = data.get("test_metrics")
            data["val_metrics"] = data.get("val_metrics")
        records.append(data)
    return records


def main() -> None:
    records = load_experiment_records()
    rows = [_row_from_experiment(r) for r in records if r.get("name")]
    df = pd.DataFrame(rows)

    if "val_macro_f1" in df.columns:
        df = df.sort_values(
            ["selectable", "val_macro_f1", "val_f1_class_1", "val_f1_class_3", "val_macro_roc_auc"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )

    out = resolve_path("reports/tables/experiment_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    selection_payload = {
        "protocol": SELECTION_NOTE,
        "primary_metric": "validation macro_f1",
        "secondary_metrics": ["validation per-class F1 (minimum)", "validation macro ROC-AUC"],
        "selected_experiment": None,
        "validation_metrics": None,
    }

    try:
        best = select_best_experiment(records)
        selection_payload["selected_experiment"] = best["name"]
        selection_payload["validation_metrics"] = best["val_metrics"]
        selection_payload["selection_sort_key"] = {
            "macro_f1": best["val_metrics"]["macro_f1"],
            "min_per_class_f1": min(best["val_metrics"]["per_class_f1"]),
            "macro_roc_auc": best["val_metrics"].get("macro_roc_auc"),
        }
        write_logit_bias_provenance(selection_payload, best["name"], best.get("logit_bias"))
    except ValueError as exc:
        selection_payload["selection_error"] = str(exc)

    selection_path = resolve_path("reports/summaries/model_selection.json")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    print(SELECTION_NOTE)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")
    print(f"Wrote {selection_path}")


if __name__ == "__main__":
    main()
