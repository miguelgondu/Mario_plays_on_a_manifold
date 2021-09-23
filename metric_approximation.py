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

        dzs = t.Tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        KLs = {}
        for dz in dzs:
            dzi, dzj = dz
            dist_z = self.model.decode(z)
            dist_z_dz = self.model.decode(z + (dz * self.eps))
            KLs[(dzi.item(), dzj.item())] = kl_divergence(dist_z, dist_z_dz).mean()

        # print(KLs)
        symmetric_term = KLs[1.0, 1.0] - KLs[1.0, 0.0] - KLs[0.0, 1.0]
        metric = (
            t.Tensor(
                [
                    [2 * KLs[1.0, 0.0], symmetric_term],
                    [symmetric_term, 2 * KLs[0.0, 1.0]],
                ]
            )
            * (1 / self.eps ** 2)
        )
        metric[metric == float("Inf")] = 1e5
        metric[metric == -float("Inf")] = -1e5

        return metric
