"""
Learning rate schedulers for Transformer training.
Implements the Noam learning rate schedule with warmup.
"""

from __future__ import annotations
import torch
from torch.optim import Optimizer


class NoamScheduler:
    """
    Noam learning rate schedule with linear warmup followed by inverse square root decay:
    lr = factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 256,
        warmup_steps: int = 400,
        factor: float = 1.0,
        min_lr: float = 1e-6
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.min_lr = min_lr
        self.step_num = 0

    def step(self) -> float:
        """Update step and adjust optimizer learning rates."""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Compute the learning rate for the current step."""
        step = max(1, self.step_num)
        lr = self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        return max(lr, self.min_lr)

    def state_dict(self) -> dict:
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "min_lr": self.min_lr
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_num = state_dict["step_num"]
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.factor = state_dict["factor"]
        self.min_lr = state_dict.get("min_lr", 1e-6)
