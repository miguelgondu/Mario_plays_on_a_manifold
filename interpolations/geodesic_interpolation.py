from geoml.discretized_manifold import DiscretizedManifold
import torch

from .base_interpolation import BaseInterpolation

Tensor = torch.Tensor


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self,
        dm: DiscretizedManifold,
        model_name: str,
        n_points_in_line: int = 10,
    ):
        super().__init__(n_points_in_line=n_points_in_line)
        self.model_name = model_name
        self.dm = dm

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        c = self.dm.connecting_geodesic(z, z_prime)
        t = torch.linspace(0, 1, self.n_points_in_line)
        interpolation = c(t)
        return interpolation
