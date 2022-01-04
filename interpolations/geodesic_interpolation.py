from typing import Tuple

from geoml.curve import CubicSpline
from geoml.manifold import Manifold
import torch

from .base_interpolation import BaseInterpolation

Tensor = torch.Tensor


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self,
        manifold: Manifold,
        n_points_in_line: int = 10,
    ):
        super().__init__(n_points_in_line=n_points_in_line)
        self.manifold = manifold

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        c, _ = self.manifold.connecting_geodesic(z, z_prime)
        t = torch.linspace(0, 1, self.n_points_in_line)
        interpolation = c(t)
        return interpolation

    def interpolate_and_return_geodesic(
        self, z: Tensor, z_prime: Tensor
    ) -> CubicSpline:
        c, _ = self.manifold.connecting_geodesic(z, z_prime)
        return c
