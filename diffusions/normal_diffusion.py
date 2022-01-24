from pathlib import Path
from typing import Dict, Tuple

import torch as t
import numpy as np
from torch.distributions import MultivariateNormal

from .base_diffusion import BaseDiffusion


class NormalDiffusion(BaseDiffusion):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_steps: int = 100
    ) -> None:
        super().__init__(vae_path, p_map, n_steps)
        self.scale = 1.0

    def run(self, z_0: t.Tensor = None) -> Tuple[t.Tensor]:
        """Returns the random walk as a Tensor of shape [n_steps, z_dim=2]"""
        # return super().run(z_0)
        z_dim = len(z_0)
        zs = [z_0]

        # Taking it from there.
        z_n = z_0
        for _ in range(self.n_steps):
            d = MultivariateNormal(z_n, covariance_matrix=self.scale * t.eye(z_dim))
            z_n = d.rsample()
            zs.append(z_n)

        zs_in_rw = t.vstack(zs)
        vae = self._load_vae()
        levels = vae.decode(zs_in_rw).probs.argmax(dim=-1)

        return zs_in_rw, levels
