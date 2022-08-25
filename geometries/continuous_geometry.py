from pathlib import Path
from typing import Dict, Tuple

import torch as t

from interpolations.geodesic_interpolation import GeodesicInterpolation
from diffusions.continuous_diffusion import ContinuousDiffusion

from geoml.discretized_manifold import DiscretizedManifold

from .geometry import Geometry


class ContinuousGeometry(Geometry):
    def __init__(
        self,
        playability_map: Dict[tuple, int],
        exp_name: str,
        vae_path: Path,
        manifold: DiscretizedManifold = None,
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)
        self.manifold = manifold
        self.interpolation = GeodesicInterpolation(
            vae_path, playability_map, manifold=manifold
        )
        self.diffusion = ContinuousDiffusion(vae_path, playability_map)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)
