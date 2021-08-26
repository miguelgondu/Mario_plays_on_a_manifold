"""
Implements a linear interpolation baseline
"""
from typing import List
import torch
import matplotlib.pyplot as plt
from torch.tensor import Tensor

from vae_geometry import VAEGeometry
from base_interpolation import BaseInterpolation


class LinearInterpolation(BaseInterpolation):
    def __init__(self, n_points: int = 10):
        super().__init__(n_points=n_points)

    def interpolate(self, z: Tensor, z_prime: Tensor) -> List[Tensor]:
        zs = [(1 - t) * z + (t * z_prime) for t in torch.linspace(0, 1, self.n_points)]
        return zs
