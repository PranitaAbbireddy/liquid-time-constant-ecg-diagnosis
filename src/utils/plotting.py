from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import resolve_path


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    subdir: str,
    graphs_dir: str = "graphs",
    dpi: int = 300,
) -> Path:
    out_dir = resolve_path(f"{graphs_dir}/{subdir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
