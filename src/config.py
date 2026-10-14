from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str = "config.yaml"):
    """Load YAML config"""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str):
    """Turn relative path entry under cgf['paths'] into absolute path."""
    return PROJECT_ROOT / cfg["paths"][key]