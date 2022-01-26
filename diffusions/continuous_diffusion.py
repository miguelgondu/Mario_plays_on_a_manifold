from pathlib import Path
from typing import Dict, Tuple

import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from vae_mario_obstacles import VAEWithObstacles

from .base_diffusion import BaseDiffusion


class ContinuousDiffusion(BaseDiffusion):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_steps: int = 50
    ) -> None:
        super().__init__(vae_path, p_map, n_steps)
        # Storing it to do the obstacle updating only once.
        self.vae = self._load_vae()
        self.scale = 5.0

    def run(self, z_0: t.Tensor = None) -> Tuple[t.Tensor]:
        # Random starting point (or the one provided)
        z_n = z_0
        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_points):
            Mz = self.vae.metric(z_n)

            d = MultivariateNormal(z_n, covariance_matrix=self.scale * Mz.inverse())
            z_n = d.rsample()
            zs.append(z_n)

        return t.vstack(zs)

    def _load_vae(self) -> VAEWithObstacles:
        vae = VAEWithObstacles()
        vae.load_state_dict(t.load(self.vae_path, map_location=vae.device))

        unplayable_levels = self.zs[self.p != 1.0]
        vae.update_obstacles(t.from_numpy(unplayable_levels).type(t.float))
        return vae


# class GeometricDifussion:
#     def __init__(self, n_points: int, scale: float = 1.0) -> None:
#         self.n_points = n_points
#         self.scale = scale

#     def run(self, vae: VAEGeometryBase, z_0: t.Tensor = None) -> t.Tensor:
#         """Returns the random walk as a Tensor of shape [n_points, z_dim=2]"""

#         # Random starting point (or the one provided)
#         if z_0 is None:
#             idx = np.random.randint(len(vae.encodings))
#             z_n = vae.encodings[idx, :]
#         else:
#             z_n = z_0

#         zs = [z_n]

#         # Taking it from there.
#         for _ in range(self.n_points):
#             Mz = vae.metric(z_n)

#             d = MultivariateNormal(z_n, covariance_matrix=self.scale * Mz.inverse())
#             z_n = d.rsample()
#             zs.append(z_n)

#         return t.vstack(zs)
