from pathlib import Path


def project_root() -> Path:
    # Resolve from src/utils/paths.py -> project root
    return Path(__file__).resolve().parents[2]


def resolve_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return project_root() / path
