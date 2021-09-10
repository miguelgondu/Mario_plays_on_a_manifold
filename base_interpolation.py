"""
Implements a base interpolation that all
the interpolation experiments inherit from.
"""
from typing import List
import torch

# Tensor type.
Tensor = torch.Tensor


class BaseInterpolation:
    def __init__(self, n_points_in_line: int = 10):
        # self.zs = zs
        # self.zs_prime = zs_prime
        self.n_points_in_line = n_points_in_line

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        """
        This function returns a tensor of zs in latent space that
        interpolate between z and z_prime.
        """
        raise NotImplementedError
