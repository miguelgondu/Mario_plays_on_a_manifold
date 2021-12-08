import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_geometry_base import VAEGeometryBase
from .base_diffusion import BaseDiffusion


class NormalDifussion(BaseDiffusion):
    def __init__(self, n_steps: int, scale: float = 0.5) -> None:
        super().__init__(n_steps=n_steps)
        self.scale = scale

    def run(self, initial_points: t.Tensor) -> t.Tensor:
        """Returns the random walk as a Tensor of shape [n_steps, z_dim=2]"""
        z_dim = initial_points.shape[1]

        # Random starting point (or the one provided)
        idx = np.random.randint(len(initial_points))
        z_n = initial_points[idx, :]

        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_steps):
            d = MultivariateNormal(z_n, covariance_matrix=self.scale * t.eye(z_dim))
            z_n = d.rsample()
            zs.append(z_n)

        return t.vstack(zs)
