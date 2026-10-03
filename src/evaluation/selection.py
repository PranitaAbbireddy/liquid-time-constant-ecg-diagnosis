from typing import Any


def selection_sort_key(val_metrics: dict[str, Any]) -> tuple:
    # Primary: validation macro-F1; secondary: min per-class F1; tertiary: macro ROC-AUC
    per_class_f1 = val_metrics.get("per_class_f1") or []
    min_class_f1 = min(per_class_f1) if per_class_f1 else 0.0
    roc = val_metrics.get("macro_roc_auc")
    roc_key = roc if roc is not None else 0.0
    return (
        float(val_metrics.get("macro_f1", 0.0)),
        float(min_class_f1),
        float(roc_key),
    )


def select_best_experiment(experiment_records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in experiment_records if r.get("val_metrics") and r.get("selectable", True)]
    if not eligible:
        raise ValueError("No experiments with validation metrics available for selection.")
    return max(eligible, key=lambda r: selection_sort_key(r["val_metrics"]))
