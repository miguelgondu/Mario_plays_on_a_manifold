from pathlib import Path
from typing import Dict, Tuple

from geoml.curve import CubicSpline
from geoml.manifold import Manifold
import torch as t

from vae_mario_hierarchical import VAEMarioHierarchical

from .base_interpolation import BaseInterpolation


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_points_in_line: int = 10
    ):
        super().__init__(vae_path, p_map, n_points_in_line)

        # Define the discretized manifold according to
        # the VAEWithObstacles (?).

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

    def _load_vae(self) -> VAEMarioHierarchical:
        # TODO: this one should be loading up the geometric one.
        return super()._load_vae()
