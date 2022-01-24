from pathlib import Path
from typing import Dict, Tuple
import torch as t

import numpy as np
from .base_diffusion import BaseDiffusion


def get_random_point(encodings: t.Tensor) -> t.Tensor:
    idx = np.random.randint(len(encodings))
    return encodings[idx, :]


class BaselineDiffusion(BaseDiffusion):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_steps: int = 100
    ) -> None:
        super().__init__(vae_path, p_map, n_steps)
        self.step_size = 1.0

    def run(self, z_0: t.Tensor = None) -> Tuple[t.Tensor]:
        zs = [z_0]

        # Taking it from there.
        z_n = z_0
        for _ in range(self.n_steps):
            target = self._get_random_playable_point()
            direction = target - z_n
            z_n = z_n + direction * (self.step_size / direction.norm())
            zs.append(z_n)

        zs_in_rw = t.vstack(zs)
        vae = self._load_vae()
        levels = vae.decode(zs_in_rw).probs.argmax(dim=-1)

        return zs_in_rw, levels

    def _get_random_playable_point(self) -> t.Tensor:
        idx = np.random.randint(len(self.playable_points))
        return t.from_numpy(self.playable_points[idx, :]).type(t.float)
