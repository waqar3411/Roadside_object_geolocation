from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Return a torch device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON."""
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load either a state_dict checkpoint or a full PyTorch model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        loaded_model = checkpoint.to(device)
        loaded_model.eval()
        return loaded_model

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model state_dict checkpoint."""
    ensure_dir(path.parent)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
