"""
Model checkpoint saving and loading utilities.
"""

from __future__ import annotations
import os
import torch
from typing import Optional, Dict, Any
from ..model.transformer import Transformer


def save_checkpoint(
    filepath: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    val_loss: float = float("inf"),
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save training state to a checkpoint file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "config": config if config is not None else model.get_config(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state"] = scheduler.state_dict()

    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[Transformer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint
