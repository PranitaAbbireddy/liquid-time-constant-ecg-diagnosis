from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root, resolve_path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = project_root() / "src" / "config" / "default.yaml"
    else:
        config_path = resolve_path(str(config_path))

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    root = project_root()
    for key in ("graphs_dir", "reports_dir", "checkpoints_dir"):
        rel = config["paths"][key]
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
    for sub in (
        "dataset",
        "training",
        "evaluation",
        "explainability",
    ):
        (root / config["paths"]["graphs_dir"] / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "tables", "summaries"):
        (root / config["paths"]["reports_dir"] / sub).mkdir(parents=True, exist_ok=True)
