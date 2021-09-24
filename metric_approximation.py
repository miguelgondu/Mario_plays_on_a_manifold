"""
Basing myself on Alison's code, I
consider the output of the hierarchical VAE as
a categorical, and take it from there.
"""
from typing import Union

import torch as t
from torch.distributions import Categorical, kl_divergence
import numpy as np

from vae_zelda_hierarchical import VAEZeldaHierarchical
from vae_mario_hierarchical import VAEMarioHierarchical
from vae_mario import VAEMario

Models = Union[VAEMario, VAEMarioHierarchical]


class MetricApproximation:
    def __init__(
        self,
        model: Models,
        z_dim: int,
        eps: float = 0.1,
    ) -> None:
        super(MetricApproximation, self).__init__()
        self.model = model
        self.model = model
        self.latent_dim = z_dim
        assert z_dim == 2
        self.num_unknows = int(
            self.latent_dim + self.latent_dim * (self.latent_dim - 1) / 2
        )
        self.eps = eps

    def __call__(self, z: t.Tensor) -> t.Tensor:
        # Alison's

        KL_e0 = (
            kl_divergence(
                self.model.reweight(z),
                self.model.reweight(z + self.eps * t.Tensor([1.0, 0.0])),
            )
            .sum()
            .item()
        )
        KL_e1 = (
            kl_divergence(
                self.model.reweight(z),
                self.model.reweight(z + self.eps * t.Tensor([0.0, 1.0])),
            )
            .sum()
            .item()
        )
        KL_e0_plus_e1 = (
            kl_divergence(
                self.model.reweight(z),
                self.model.reweight(z + self.eps * t.Tensor([1.0, 1.0])),
            )
            .sum()
            .item()
        )

        metric = (
            t.Tensor(
                [
                    [2 * KL_e0, KL_e0_plus_e1 - KL_e0 - KL_e1],
                    [KL_e0_plus_e1 - KL_e0 - KL_e1, 2 * KL_e1],
                ]
            )
            * (1 / self.eps ** 2)
        )
        # metric[metric == float("Inf")] = 1e5
        # metric[metric == -float("Inf")] = -1e5

        return metric
