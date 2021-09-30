"""
Implements a linear interpolation baseline
"""
import torch

from .base_interpolation import BaseInterpolation

Tensor = torch.Tensor


class LinearInterpolation(BaseInterpolation):
    def __init__(self, n_points_in_line: int = 10):
        super().__init__(n_points_in_line=n_points_in_line)

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        zs = [
            (1 - t) * z + (t * z_prime)
            for t in torch.linspace(0, 1, self.n_points_in_line)
        ]
        zs = torch.vstack(zs)
        return zs
