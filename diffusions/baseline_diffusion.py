import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_geometry_base import VAEGeometryBase


def get_random_point(encodings: t.Tensor) -> t.Tensor:
    idx = np.random.randint(len(encodings))
    return encodings[idx, :]


class BaselineDiffusion:
    def __init__(self, n_points: int, step_size: float = 1.0) -> None:
        self.n_points = n_points
        self.step_size = step_size

    def run(self, vae: VAEGeometryBase, z_0: t.Tensor = None) -> t.Tensor:
        """
        Returns the random walk as a Tensor of shape [n_points, z_dim=2].

        It randomly samples an encoding and takes a step in that direction.
        """

        # Random starting point (or the one provided)
        if z_0 is None:
            idx = np.random.randint(len(vae.encodings))
            z_n = vae.encodings[idx, :]
        else:
            z_n = z_0

        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_points):
            target = get_random_point(vae.encodings)
            direction = target - z_n
            z_n = z_n + direction * (self.step_size / direction.norm())
            zs.append(z_n)

        return t.vstack(zs)
