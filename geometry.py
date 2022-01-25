"""
Defines a base geometry interface for the experiments
with common utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch as t
from diffusions.discrete_diffusion import DiscreteDiffusion
from experiment_utils import grid_from_map, positions_from_map, get_random_pairs

from interpolations.discrete_interpolation import DiscreteInterpolation
from interpolations.linear_interpolation import LinearInterpolation

from diffusions.baseline_diffusion import BaselineDiffusion
from diffusions.normal_diffusion import NormalDiffusion


class Geometry:
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        pass
        self.playability_map = playability_map
        self.exp_name = exp_name
        self.grid, self.positions = self._load_into_grid(playability_map)
        self.vae_path = vae_path

        self.zs = np.array([z for z in playability_map.keys()])
        self.p = np.array([p for p in playability_map.values()])
        self.playable_points = t.from_numpy(self.zs[self.p == 1]).type(t.float)

        self.interpolation_path = Path(
            f"./data/arrays/ten_vaes/interpolations/{exp_name}"
        )
        self.diffusion_path = Path(f"./data/arrays/ten_vaes/diffusions/{exp_name}")
        self.diffusion_path.mkdir(exist_ok=True, parents=True)
        self.model_name = vae_path.stem

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def save_arrays(self):
        """
        Loads up the VAE at said path and runs
        - 100 random interpolations of 10 points each.
        - 10 random walks (or diffusions) of 100 steps each.
        """
        self._save_arrays_for_interpolation()
        self._save_arrays_for_diffusion()

    def _save_arrays_for_interpolation(self):
        n_interpolations = 100
        z1s, z2s = get_random_pairs(self.playable_points, n_interpolations)
        for i, (z1, z2) in enumerate(zip(z1s, z2s)):
            path = self.interpolation_path / f"{self.model_name}_interp_{i:02d}.npz"
            print(f"Saving {path}")
            zs, levels = self.interpolate(z1, z2)
            np.savez(path, zs=zs.detach().numpy(), levels=levels.detach().numpy())

    def _save_arrays_for_diffusion(self, path: Path):
        n_diffusions = 10
        random_idxs = np.random.permutation(len(self.playable_points))[:n_diffusions]
        initial_points = self.playable_points[:random_idxs]
        for d, z_0 in enumerate(initial_points):
            path = self.diffusion_path / f"{self.model_name}_diff_{d:02d}.npz"
            print(f"Saving {path}")
            zs, levels = self.diffuse(z_0)
            np.savez(path, zs=zs.detach().numpy(), levels=levels.detach().numpy())

    # Should be a static method.
    def _load_into_grid(
        self, playability_map: Dict[tuple, int]
    ) -> Tuple[np.ndarray, dict]:
        p_img = grid_from_map(playability_map)
        positions = positions_from_map(playability_map)

        return p_img, positions


class BaselineGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)
        self.interpolation = LinearInterpolation(vae_path, playability_map)
        self.diffusion = BaselineDiffusion(vae_path, playability_map)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)


class NormalGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)
        self.interpolation = LinearInterpolation(vae_path, playability_map)
        self.diffusion = NormalDiffusion(vae_path, playability_map)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)


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


class ContinuousGeometry(Geometry):
    def __init__(
        self, playability_map: Dict[tuple, int], exp_name: str, vae_path: Path
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return super().interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return super().diffuse(z_0)
