import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_geometry_base import VAEGeometryBase

# TODO: implement this diffusion as a random walk on a graph.
class DiscreteDiffusion:
    def __init__(self, n_points: int, scale: float = 1.0) -> None:
        self.n_points = n_points
        self.scale = scale

    def run(self, vae: VAEGeometryBase, z_0: t.Tensor = None) -> t.Tensor:
        """Returns the random walk as a Tensor of shape [n_points, z_dim=2]"""

        # Random starting point (or the one provided)
        if z_0 is None:
            idx = np.random.randint(len(vae.encodings))
            z_n = vae.encodings[idx, :]
        else:
            z_n = z_0

        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_points):
            Mz = vae.metric(z_n)

            d = MultivariateNormal(z_n, covariance_matrix=self.scale * Mz.inverse())
            z_n = d.rsample()
            zs.append(z_n)

        return t.vstack(zs)
