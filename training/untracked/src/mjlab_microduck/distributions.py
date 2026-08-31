"""Training-only policy distributions for Microduck tasks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules.distribution import Distribution, GaussianDistribution


class BoundedGaussianDistribution(GaussianDistribution):
    """State-independent Gaussian with a smoothly bounded exploration std.

    The bounds affect stochastic PPO sampling only. Deterministic policy export
    remains the MLP mean, so ONNX inference and the 61D deployment contract are
    unchanged.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.6,
        min_std: float = 0.05,
        max_std: float = 1.5,
    ) -> None:
        Distribution.__init__(self, output_dim)
        if not 0.0 < min_std < init_std < max_std:
            raise ValueError(
                "expected 0 < min_std < init_std < max_std, got "
                f"{min_std}, {init_std}, {max_std}"
            )
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        fraction = (init_std - min_std) / (max_std - min_std)
        raw_init = math.log(fraction / (1.0 - fraction))
        self.raw_std_param = nn.Parameter(torch.full((output_dim,), raw_init))
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _bounded_std(self) -> torch.Tensor:
        return self.min_std + (self.max_std - self.min_std) * torch.sigmoid(
            self.raw_std_param
        )

    def update(self, mlp_output: torch.Tensor) -> None:
        std = self._bounded_std().expand_as(mlp_output)
        self._distribution = Normal(mlp_output, std)
