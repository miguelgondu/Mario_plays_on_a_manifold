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


class MetricApproximation:
    def __init__(
        self,
        model: Union[VAEZeldaHierarchical, VAEMarioHierarchical],
        obs_dim: int,
        z_dim: int,
        eps: float = 0.05,
    ) -> None:
        super(MetricApproximation, self).__init__()
        self.model = model
        self.obs_dim = obs_dim
        self.model = model
        self.latent_dim = z_dim
        self.num_unknows = int(
            self.latent_dim + self.latent_dim * (self.latent_dim - 1) / 2
        )
        self.eps = eps

    def __call__(self, latent_z) -> t.Tensor:
        # Alison's

        latent_dzs = self.create_deltas()
        batch_z_dzs = (latent_dzs + latent_z[:, None]).t()
        kl_vector = t.empty(self.obs_dim, self.num_unknows)
        for i, z_dzs in enumerate(batch_z_dzs):
            dist_z = self.model.decode(latent_z)
            dist_z_dz = self.model.decode(z_dzs)
            kl_vector[:, i] = kl_divergence(dist_z, dist_z_dz)
        if self.obs_dim == 1:
            metric = self.construct_metric(t.squeeze(kl_vector))
        else:
            metric = self.get_metric(kl_vector)
        return metric

    def create_deltas(self) -> t.Tensor:
        """
        Create the pertubation dz.
        For a 3x3 metric, we should have 6 unknowns.
        The diag. terms are obtained with: dzs = [1,0,0], [0,1,0], [0,0,1]
        The non-diag. terms with: dzs = [1,1,0], [1,0,1], [0,1,1]

        This code is Alison's.
        """
        vecs = []
        for i in range(1, self.latent_dim):
            identity = t.eye(self.latent_dim - i)
            res = t.vstack((t.ones(self.latent_dim - i), identity))
            res = t.vstack((t.zeros((i - 1, self.latent_dim - i)), res))
            vecs.append(res)
        nondiagonal = t.hstack(vecs)
        latent_dzs = t.hstack((t.eye(self.latent_dim), nondiagonal))
        return self.eps * latent_dzs

    def construct_metric(self, kl_vector) -> t.Tensor:
        """
        We construct the metric using the KL divergence of pertubations.
        If we have kl_vector = [x1, x2, x3, s4, s5, s6] for a 3x3 metric, then
        M = [[x1, x4, x5], [x4, x2, x6], [x5, x6, x3]].
        We first obatin the diagonal, and compute the non-diagonal terms, s.t.:
        s4 = x1 + x2 + 2 * x4.
        """
        assert kl_vector.size(0) == self.num_unknows
        diagonal = 2 * kl_vector[: self.latent_dim] / (self.eps ** 2)  # metric elements
        metric = t.diag_embed(diagonal)
        jmax = self.latent_dim
        for i in range(0, self.latent_dim - 1):
            jmax -= 1
            for j in range(i + 1, i + 1 + jmax):
                metric[i, j] = (
                    kl_vector[self.latent_dim + j + i - 1] - kl_vector[i] - kl_vector[j]
                ) / (self.eps ** 2)
        metric = metric + metric.t() - t.diag_embed(diagonal)
        # metric / (self.eps**2)
        return metric

    def get_metric(self, kl_vector) -> t.Tensor:
        """
        Here, the 'final' metric is the product of the individual metric
        obtained for each component of the observational variable.
        If x = [x1, x2, x3], then M = M1 @ M2 @ M3.
        """
        M = self.construct_metric(kl_vector[0, :])
        for i in range(1, self.obs_dim):
            Mi = self.construct_metric(kl_vector[i, :])
            M = M + Mi
        return M
