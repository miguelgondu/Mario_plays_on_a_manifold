"""
Implements a linear interpolation baseline
"""
from pathlib import Path
from typing import Dict, Tuple
import torch as t

from .base_interpolation import BaseInterpolation


class LinearInterpolation(BaseInterpolation):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_points_in_line: int = 10
    ):
        super().__init__(vae_path, p_map, n_points_in_line)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        zs = [
            (1 - t) * z + (t * z_prime) for t in t.linspace(0, 1, self.n_points_in_line)
        ]
        zs = t.vstack(zs)

        vae = self._load_vae()
        levels = vae.decode(zs).probs.argmax(dim=-1)
        return zs, levels
