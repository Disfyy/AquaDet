"""Exponential Moving Average for model parameters."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Model Exponential Moving Average with decay ramp-up.

    Early in training the model changes rapidly; a high fixed decay
    causes the EMA to lag badly.  The ramp-up schedule starts with a
    low effective decay and asymptotically approaches the target decay:

        decay_t = min(target_decay, (1 + step) / (10 + step))

    Args:
        model: The model whose parameters are averaged.
        decay: Target EMA decay (default 0.9999).
        device: Device to store the shadow parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device | None = None,
    ):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.target_decay = decay
        self.device = device
        self.num_updates = 0
        if device is not None:
            self.module.to(device=device)

    @property
    def decay(self) -> float:
        """Compute the current decay based on step count."""
        return min(
            self.target_decay,
            (1 + self.num_updates) / (10 + self.num_updates),
        )

    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(),
                model.state_dict().values(),
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current decay."""
        self.num_updates += 1
        d = self.decay
        self._update(model, update_fn=lambda e, m: d * e + (1.0 - d) * m)

    def set(self, model: nn.Module) -> None:
        """Hard-copy model parameters to EMA (no averaging)."""
        self._update(model, update_fn=lambda e, m: m)
