"""Verify logit bias metadata and that tuning code paths use validation data only."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> None:
    config = load_config()
    ckpt_path = resolve_path(
        f"{config['paths']['checkpoints_dir']}/{config['training']['checkpoint_name']}"
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    report = {
        "checkpoint": str(ckpt_path),
        "logit_bias_present": checkpoint.get("logit_bias") is not None,
        "logit_bias_fit_split": checkpoint.get("logit_bias_fit_split"),
        "logit_bias_tuned_on_test": checkpoint.get("logit_bias_tuned_on_test"),
        "expected_fit_split": "validation",
        "expected_tuned_on_test": False,
    }
    report["protocol_ok"] = (
        report["logit_bias_fit_split"] == "validation"
        and checkpoint.get("logit_bias_tuned_on_test") is False
    )

    prov_path = resolve_path("reports/summaries/logit_bias_provenance.json")
    selection_path = resolve_path("reports/summaries/model_selection.json")
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        report["provenance_file"] = provenance
        frozen = (provenance.get("validation_metrics_with_frozen_bias") or {}).get("macro_f1")
        report["provenance_macro_f1"] = frozen
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_f1 = (selection.get("validation_metrics") or {}).get("macro_f1")
            report["selected_experiment"] = selection.get("selected_experiment")
            report["selected_validation_macro_f1"] = selected_f1
            report["provenance_matches_selection"] = (
                provenance.get("selected_experiment") == selection.get("selected_experiment")
                and frozen is not None
                and selected_f1 is not None
                and abs(float(frozen) - float(selected_f1)) < 1e-12
            )
        else:
            report["provenance_matches_selection"] = True

    print(json.dumps(report, indent=2))
    if not report["protocol_ok"] or not report.get("provenance_matches_selection", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
