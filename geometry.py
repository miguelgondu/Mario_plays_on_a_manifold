"""
Defines a base geometry interface for the experiments
with common utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch as t

from interpolations.base_interpolation import BaseInterpolation
from diffusions.base_diffusion import BaseDiffusion


class Geometry:
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        pass
        self.playability_map = playability_map
        self.exp_name = exp_name
        self.grid = self._load_into_grid(playability_map)
        self.vae_path = vae_path

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def save_arrays(self, path: Path):
        """
        Loads up the VAE at said path and runs
        - 100 random interpolations of 10 points each.
        - 10 random walks (or diffusions) of 100 steps each.
        """
        self._save_arrays_for_interpolation(path)
        self._save_arrays_for_diffusion(path)

    def _save_arrays_for_interpolation(self, path: Path):
        # TODO: implement
        return

    def _save_arrays_for_diffusion(self, path: Path):
        # TODO: implement
        return

    # Should be a static method.
    def _load_into_grid(self, playability_map: Dict[tuple, int]) -> np.ndarray:
        # TODO: implement
        return


class BaselineGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return super().interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return super().diffuse(z_0)


class NormalGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return super().interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return super().diffuse(z_0)


class DiscreteGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return super().interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return super().diffuse(z_0)


class ContinuousGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return super().interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return super().diffuse(z_0)
