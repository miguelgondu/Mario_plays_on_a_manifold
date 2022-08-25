"""
Defines a base geometry interface for the experiments
with common utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch as t

from interpolations.linear_interpolation import LinearInterpolation

from diffusions.baseline_diffusion import BaselineDiffusion

from geometries.geometry import Geometry


class BaselineGeometry(Geometry):
    def __init__(
        self,
        playability_map: Dict[tuple, int],
        exp_name: str,
        vae_path: Path,
        exp_folder: str = "ten_vaes",
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path, exp_folder=exp_folder)
        self.interpolation = LinearInterpolation(vae_path, playability_map)
        self.diffusion = BaselineDiffusion(vae_path, playability_map)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)
