import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_geometry_base import VAEGeometryBase


class NormalDifussion:
    def __init__(self, n_steps: int, scale: float = 0.5) -> None:
        self.n_steps = n_steps
        self.scale = scale

    def run(self, vae: VAEGeometryBase, z_0: t.Tensor = None) -> t.Tensor:
        """Returns the random walk as a Tensor of shape [n_steps, z_dim=2]"""

        # Random starting point (or the one provided)
        if z_0 is None:
            idx = np.random.randint(len(vae.encodings))
            z_n = vae.encodings[idx, :]
        else:
            z_n = z_0

        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_steps):
            d = MultivariateNormal(z_n, covariance_matrix=self.scale * t.eye(2))
            z_n = d.rsample()
            zs.append(z_n)

        return t.vstack(zs)
