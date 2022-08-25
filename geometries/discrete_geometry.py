from pathlib import Path
from typing import Dict, Tuple

import torch as t

from interpolations.discrete_interpolation import DiscreteInterpolation
from diffusions.discrete_diffusion import DiscreteDiffusion

from .geometry import Geometry


class DiscreteGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)
        self.interpolation = DiscreteInterpolation(vae_path, playability_map)
        self.diffusion = DiscreteDiffusion(vae_path, playability_map)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)
