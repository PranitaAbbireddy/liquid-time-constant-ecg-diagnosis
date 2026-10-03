"""Systematic experiments with validation-only selection (test optional, diagnostic only)."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_csv_arrays
from src.evaluation.selection import select_best_experiment
from src.evaluation.visualization import plot_class_distribution
from src.training.pipeline import train_one_experiment
from src.utils.config import ensure_output_dirs, load_config
from src.utils.paths import resolve_path
from src.utils.seed import set_seed

import importlib.util


def _run_build_comparison_table() -> None:
    path = ROOT / "scripts" / "build_experiment_table.py"
    spec = importlib.util.spec_from_file_location("build_experiment_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()

EXPERIMENTS = [
    {
        "name": "baseline_weighted_ce_argmax",
        "training": {
            "loss_type": "weighted_ce",
            "weight_scheme": "inverse",
            "sampling": "none",
            "epochs": 10,
        },
        "tune_logit_bias": False,
    },
    {
        "name": "baseline_weighted_ce_val_bias",
        "training": {
            "loss_type": "weighted_ce",
            "weight_scheme": "inverse",
            "sampling": "none",
            "epochs": 10,
        },
        "tune_logit_bias": True,
    },
    {
        "name": "focal_loss_gamma2",
        "training": {
            "loss_type": "focal",
            "weight_scheme": "none",
            "sampling": "none",
            "focal_gamma": 2.0,
            "epochs": 10,
        },
        "tune_logit_bias": True,
    },
    {
        "name": "weighted_sampler_plain_ce",
        "training": {
            "loss_type": "ce",
            "weight_scheme": "none",
            "sampling": "weighted",
            "epochs": 10,
        },
        "tune_logit_bias": True,
    },
    {
        "name": "sqrt_inverse_class_weights",
        "training": {
            "loss_type": "weighted_ce",
            "weight_scheme": "sqrt_inverse",
            "sampling": "none",
            "epochs": 10,
        },
        "tune_logit_bias": True,
    },
    {
        "name": "stride2_hidden128_weighted_ce",
        "data": {"temporal_stride": 2},
        "model": {"hidden_size": 128},
        "training": {
            "loss_type": "weighted_ce",
            "weight_scheme": "sqrt_inverse",
            "sampling": "none",
            "epochs": 12,
        },
        "tune_logit_bias": True,
    },
]


def _save_experiment_checkpoint(result: dict, config: dict) -> Path:
    paths = config["paths"]
    ckpt_dir = resolve_path(f"{paths['checkpoints_dir']}/experiments")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{result['name']}.pt"
    torch.save(
        {
            "model_state_dict": result["model_state_dict"],
            "config": result["full_config"],
            "normalizer": result.get("normalizer"),
            "logit_bias": result.get("logit_bias"),
            "logit_bias_fit_split": result.get("logit_bias_fit_split", "validation"),
            "experiment_name": result["name"],
            "val_metrics": result.get("val_metrics"),
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--diagnostic-test",
        action="store_true",
        help="Compute test metrics for reference only (not used for selection).",
    )
    parser.add_argument(
        "--finalize-test",
        action="store_true",
        help="After validation-based selection, run one held-out test evaluation.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_output_dirs(config)
    set_seed(config["project"]["seed"])

    train_signals, train_labels = load_csv_arrays(config["paths"]["train_csv"])
    class_names = {int(k): v for k, v in config["classes"].items()}
    plot_class_distribution(
        train_labels,
        class_names,
        config["paths"]["graphs_dir"],
        config["evaluation"]["figure_dpi"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.only]
        if not experiments:
            raise ValueError(f"Unknown experiment: {args.only}")

    results = []
    for spec in experiments:
        print(f"\n=== Running experiment: {spec['name']} ===", flush=True)
        result = train_one_experiment(
            config,
            spec,
            device,
            max_batches=args.max_batches,
            evaluate_test_diagnostic=args.diagnostic_test,
        )
        _save_experiment_checkpoint(result, config)

        serializable = deepcopy(result)
        serializable.pop("model_state_dict")
        out_path = resolve_path(f"reports/experiments/{spec['name']}.json")
        out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        results.append(result)

    _run_build_comparison_table()

    best = select_best_experiment(results)
    print(
        f"\nValidation-selected experiment: {best['name']} "
        f"(val macro-F1={best['val_metrics']['macro_f1']:.4f})"
    )

    if args.finalize_test:
        finalize_path = ROOT / "scripts" / "finalize_selected_model.py"
        spec = importlib.util.spec_from_file_location("finalize_selected_model", finalize_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.main()


if __name__ == "__main__":
    main()
