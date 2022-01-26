from pathlib import Path
from typing import Dict, Tuple

from geoml.curve import CubicSpline
from geoml.discretized_manifold import DiscretizedManifold
import torch as t

from vae_mario_obstacles import VAEWithObstacles

from .base_interpolation import BaseInterpolation


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_points_in_line: int = 10
    ):
        super().__init__(vae_path, p_map, n_points_in_line)

        # Define the discretized manifold according to
        # the VAEWithObstacles (?).
        vae = self._load_vae()
        grid = [t.linspace(-5, 5, 50), t.linspace(-5, 5, 50)]
        Mx, My = t.meshgrid(grid[0], grid[1])
        grid2 = t.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)

        self.manifold = DiscretizedManifold(self, grid2, use_diagonals=True)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        # Use the connecting geom
        c, _ = self.manifold.connecting_geodesic(z, z_prime)
        time = t.linspace(0, 1, self.n_points_in_line)
        interpolation = c(time)

        # TODO: decode the levels
        levels = None

        return interpolation, levels

    def interpolate_and_return_geodesic(
        self, z: t.Tensor, z_prime: t.Tensor
    ) -> CubicSpline:
        c, _ = self.manifold.connecting_geodesic(z, z_prime)
        return c

    def _load_vae(self) -> VAEWithObstacles:
        # TODO: this one should be loading up the geometric one.
        vae = VAEWithObstacles()
        vae.load_state_dict(t.load(self.vae_path, map_location=vae.device))

        unplayable_levels = self.zs[self.p != 1.0]
        vae.update_obstacles(t.from_numpy(unplayable_levels).type(t.float))
        return vae
