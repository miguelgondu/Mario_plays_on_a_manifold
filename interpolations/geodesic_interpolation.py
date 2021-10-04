from typing import Tuple

from geoml.curve import CubicSpline
from geoml.discretized_manifold import DiscretizedManifold
import torch

from .base_interpolation import BaseInterpolation

Tensor = torch.Tensor


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self,
        dm: DiscretizedManifold,
        n_points_in_line: int = 10,
    ):
        super().__init__(n_points_in_line=n_points_in_line)
        self.dm = dm

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        c = self.dm.connecting_geodesic(z, z_prime)
        t = torch.linspace(0, 1, self.n_points_in_line)
        interpolation = c(t)
        return interpolation

    def interpolate_and_return_geodesic(
        self, z: Tensor, z_prime: Tensor
    ) -> CubicSpline:
        c = self.dm.connecting_geodesic(z, z_prime)
        return c
