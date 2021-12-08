import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_geometry_base import VAEGeometryBase
from .base_diffusion import BaseDiffusion


def get_random_point(encodings: t.Tensor) -> t.Tensor:
    idx = np.random.randint(len(encodings))
    return encodings[idx, :]


class BaselineDiffusion(BaseDiffusion):
    def __init__(self, n_steps: int, step_size: float = 1.0) -> None:
        super().__init__(n_steps=n_steps)
        self.step_size = step_size

    def run(self, initial_points: t.Tensor) -> t.Tensor:
        """
        Returns the random walk as a Tensor of shape [n_points, z_dim=2].

        It randomly samples an encoding and takes a step in that direction.
        """

        # Random starting point
        idx = np.random.randint(len(initial_points))
        z_n = initial_points[idx, :]

        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_steps):
            target = get_random_point(initial_points)
            direction = target - z_n
            z_n = z_n + direction * (self.step_size / direction.norm())
            zs.append(z_n)

        return t.vstack(zs)
