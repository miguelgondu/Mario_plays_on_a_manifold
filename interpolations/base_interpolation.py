"""
Implements a base interpolation that all
the interpolation experiments inherit from.
"""
from pathlib import Path
from typing import Dict, Tuple
import torch as t

from vae_mario_hierarchical import VAEMarioHierarchical


class BaseInterpolation:
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_points_in_line: int = 10
    ):
        self.vae_path = vae_path
        self.p_map = p_map
        self.n_points_in_line = n_points_in_line

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        """
        This function returns a tensor of zs
        in latent space that interpolate between z and z_prime,
        alongside with the decoded levels.
        """
        raise NotImplementedError

    def _load_vae(self) -> VAEMarioHierarchical:
        vae = VAEMarioHierarchical()
        device = vae.device
        vae.load_state_dict(t.load(self.vae_path, map_location=device))

        return vae
