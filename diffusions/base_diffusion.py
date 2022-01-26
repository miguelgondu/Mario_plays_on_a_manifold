from pathlib import Path
from typing import Dict, Tuple

import torch as t
import numpy as np

from vae_mario_hierarchical import VAEMarioHierarchical


class BaseDiffusion:
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_steps: int = 50
    ) -> None:
        self.vae_path = vae_path
        self.p_map = p_map
        self.n_steps = n_steps

        # Some more arguments that are useful
        self.zs = np.array([k for k in p_map.keys()])
        self.p = np.array([p for p in p_map.values()])
        self.playable_points = self.zs[self.p == 1.0]

    def run(self, z_0: t.Tensor = None) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def _load_vae(self) -> VAEMarioHierarchical:
        vae = VAEMarioHierarchical()
        device = vae.device
        vae.load_state_dict(t.load(self.vae_path, map_location=device))

        return vae
