"""
Defines a base geometry interface for the experiments
with common utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch as t
import matplotlib.pyplot as plt
from utils.experiment import (
    grid_from_map,
    load_arrays_as_map,
    load_csv_as_map,
    positions_from_map,
    get_random_pairs,
)

from interpolations.discrete_interpolation import DiscreteInterpolation
from interpolations.geodesic_interpolation import GeodesicInterpolation
from interpolations.linear_interpolation import LinearInterpolation
from diffusions.continuous_diffusion import ContinuousDiffusion
from diffusions.discrete_diffusion import DiscreteDiffusion

from diffusions.baseline_diffusion import BaselineDiffusion
from diffusions.normal_diffusion import NormalDiffusion

from geoml.discretized_manifold import DiscretizedManifold

from vae_models.vae_mario_obstacles import VAEWithObstacles
from vae_models.vae_zelda_hierachical import VAEZeldaHierarchical

from utils.zelda.grammar import grammar_check
from vae_models.vae_zelda_obstacles import VAEZeldaWithObstacles


class Geometry:
    def __init__(
        self,
        p_map: Dict[tuple, int],
        exp_name: str,
        vae_path: Path,
        exp_folder: str = "ten_vaes",
    ) -> None:
        pass
        self.playability_map = p_map
        self.exp_name = exp_name
        self.grid, self.positions = self._load_into_grid(p_map)
        self.inverse_positions = {v: k for k, v in self.positions.items()}
        self.vae_path = vae_path

        self.zs = np.array([z for z in p_map.keys()])
        self.p = np.array([p for p in p_map.values()])
        self.playable_points = t.from_numpy(self.zs[self.p == 1]).type(t.float)

        self.interpolation_path = Path(
            f"./data/arrays/{exp_folder}/interpolations/{exp_name}"
        )
        self.interpolation_path.mkdir(exist_ok=True, parents=True)
        self.diffusion_path = Path(f"./data/arrays/{exp_folder}/diffusions/{exp_name}")
        self.diffusion_path.mkdir(exist_ok=True, parents=True)
        self.model_name = vae_path.stem

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        raise NotImplementedError

    def save_arrays(self, force=False):
        """
        Loads up the VAE at said path and runs
        - 20 random interpolations of 10 points each.
        - 10 random walks (or diffusions) of 20 steps each.
        """
        try:
            self._save_arrays_for_interpolation(force)
        except Exception as e:
            print(f"Couldn't save interpolations for {self.exp_name} ({self.vae_path})")
            print(f"Exception: {e}")

        try:
            self._save_arrays_for_diffusion(force)
        except Exception as e:
            print(f"Couldn't save diffusions for {self.exp_name} ({self.vae_path})")
            print(f"Exception: {e}")
            raise e

    def _get_arrays_for_interpolation(self) -> Dict[int, tuple]:
        n_interpolations = 20
        z1s, z2s = get_random_pairs(self.playable_points, n_interpolations)
        all_interpolations = {}
        for i, (z1, z2) in enumerate(zip(z1s, z2s)):
            assert (z1 != z2).any()
            zs, levels = self.interpolate(z1, z2)
            all_interpolations[i] = (zs, levels)

        return all_interpolations

    def _save_arrays_for_interpolation(self, force=False):
        n_interpolations = 20
        z1s, z2s = get_random_pairs(self.playable_points, n_interpolations)
        for i, (z1, z2) in enumerate(zip(z1s, z2s)):
            assert (z1 != z2).any()
            path = self.interpolation_path / f"{self.model_name}_interp_{i:02d}.npz"
            if path.exists() and not force:
                print(f"There's already an array at {path}. Skipping.")
                continue

            print(f"Saving {path}")
            zs, levels = self.interpolate(z1, z2)
            np.savez(path, zs=zs.detach().numpy(), levels=levels.detach().numpy())

    def _get_arrays_for_diffusion(self) -> Dict[int, tuple]:
        n_diffusions = 10
        random_idxs = np.random.permutation(len(self.playable_points))[:n_diffusions]
        initial_points = self.playable_points[random_idxs]
        all_diffusions = {}
        for d, z_0 in enumerate(initial_points):
            zs, levels = self.diffuse(z_0)
            all_diffusions[d] = (zs, levels)

        return all_diffusions

    def _save_arrays_for_diffusion(self, force=False):
        n_diffusions = 10
        random_idxs = np.random.permutation(len(self.playable_points))[:n_diffusions]
        initial_points = self.playable_points[random_idxs]
        for d, z_0 in enumerate(initial_points):
            path = self.diffusion_path / f"{self.model_name}_diff_{d:02d}.npz"
            if path.exists() and not force:
                print(f"There's already an array at {path}. Skipping.")
                continue

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
