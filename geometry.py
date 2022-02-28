"""
Defines a base geometry interface for the experiments
with common utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch as t
import matplotlib.pyplot as plt
from experiment_utils import (
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

from vae_mario_obstacles import VAEWithObstacles
from vae_zelda_hierachical import VAEZeldaHierarchical

from grammar_zelda import grammar_check
from vae_zelda_obstacles import VAEZeldaWithObstacles


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


class NormalGeometry(Geometry):
    def __init__(
        self,
        playability_map: Dict[tuple, int],
        exp_name: str,
        vae_path: Path,
        exp_folder: str = "ten_vaes",
    ) -> None:
        super().__init__(playability_map, exp_name, vae_path, exp_folder=exp_folder)
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


class DiscretizedGeometry(Geometry):
    def __init__(
        self,
        p_map: Dict[tuple, int],
        exp_name: str,
        vae_path: Path,
        exp_folder: str = "ten_vaes",
        beta: float = -5.5,
        n_grid: int = 100,
        inner_steps_diff: int = 25,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        force: bool = False,
    ) -> None:
        metric_vol_path = Path(f"./data/processed/metric_volumes/{vae_path.stem}.npz")

        if metric_vol_path.exists() and not force:
            array = np.load(metric_vol_path)
            zs = array["zs"]
            metric_volumes = array["metric_volumes"]
        else:
            # Load the VAE
            if "zelda" in exp_name:
                model = VAEZeldaWithObstacles
            else:
                model = VAEWithObstacles

            # Set p_map == 0.0 as obstacles (given some beta)
            vae = model()
            vae.load_state_dict(t.load(vae_path, map_location=vae.device))
            vae.update_obstacles(
                t.Tensor([z for z, p in p_map.items() if p == 0.0]), beta=beta
            )

            # Consider a grid of arbitrary fineness (given some m)
            z1 = t.linspace(*x_lims, n_grid)
            z2 = t.linspace(*y_lims, n_grid)

            zs = t.Tensor([[x, y] for x in z1 for y in z2])
            metric_volumes = []
            metrics = vae.metric(zs)
            for Mz in metrics:
                detMz = t.det(Mz).item()
                if detMz < 0:
                    metric_volumes.append(np.inf)
                else:
                    metric_volumes.append(np.log(detMz))

            zs = zs.detach().numpy()
            metric_volumes = np.array(metric_volumes)

            np.savez(metric_vol_path, zs=zs, metric_volumes=metric_volumes)

        # build interpolation and diffusion with that new p_map
        p = (metric_volumes < metric_volumes.mean()).astype(int)

        new_p_map = load_arrays_as_map(zs, p)
        super().__init__(new_p_map, exp_name, vae_path, exp_folder=exp_folder)

        self.interpolation = DiscreteInterpolation(self.vae_path, self.playability_map)
        self.diffusion = DiscreteDiffusion(
            self.vae_path, self.playability_map, inner_steps=inner_steps_diff
        )

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        return self.interpolation.interpolate(z, z_prime)

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        return self.diffusion.run(z_0)


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


if __name__ == "__main__":
    # vae_path = Path("models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    # model_name = vae_path.stem
    # path_to_gt = (
    #     Path("./data/array_simulation_results/ten_vaes/ground_truth")
    #     / f"{model_name}.csv"
    # )
    # mean_p_map = load_csv_as_map(path_to_gt)
    # strict_p_map = {z: 1.0 if p == 1.0 else 0.0 for z, p in mean_p_map.items()}
    # ddg = DiscretizedGeometry(
    #     strict_p_map,
    #     "discretized_strict_gt",
    #     vae_path,
    #     beta=-5.5,
    #     n_grid=100,
    #     inner_steps_diff=30,
    # )

    # _, ax = plt.subplots(1, 1)
    # ax.imshow(ddg.grid, cmap="Blues", extent=[-5, 5, -5, 5])

    # interp = ddg.interpolation._full_interpolation(
    #     t.Tensor([-3.0, -4.5]), t.Tensor([3.0, 3.0])
    # )
    # diff, _ = ddg.diffuse(t.Tensor([-3.0, -4.5]))

    # ax.plot(interp[:, 0], interp[:, 1])
    # ax.scatter(diff[:, 0], diff[:, 1])

    # plt.show()
    vae_path = Path("./models/zelda/zelda_hierarchical_final_0.pt")
    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path))
    x_lims = (-4, 4)
    y_lims = (-4, 4)
    n_rows = n_cols = 100
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }
    zs_in_positions = t.Tensor([z for z in positions.keys()]).type(t.float)
    levels = vae.decode(zs_in_positions).probs.argmax(dim=-1)
    p_map = {z: grammar_check(level) for z, level in zip(positions.keys(), levels)}

    ddg = DiscretizedGeometry(
        p_map,
        "zelda_discretized_grammar_gt",
        vae_path,
        beta=-5.5,
        n_grid=100,
        inner_steps_diff=30,
        x_lims=x_lims,
        y_lims=y_lims,
    )

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7 * 3, 7))
    ax1.imshow(ddg.grid, cmap="Blues", extent=[*x_lims, *y_lims])
    ax2.imshow(ddg.grid, cmap="Blues", extent=[*x_lims, *y_lims])
    ax3.imshow(ddg.grid, cmap="Blues", extent=[*x_lims, *y_lims])
    non_playable = np.array([z for z, p in p_map.items() if p == 0.0])
    if len(non_playable) > 0:
        ax1.scatter(non_playable[:, 0], non_playable[:, 1], marker="x", c="#8F2D56")
        ax2.scatter(non_playable[:, 0], non_playable[:, 1], marker="x", c="#8F2D56")
        ax3.scatter(non_playable[:, 0], non_playable[:, 1], marker="x", c="#8F2D56")

    encodings = vae.encode(vae.train_data).mean.detach().numpy()
    ax3.scatter(encodings[:, 0], encodings[:, 1], marker="x", c="k")

    all_interps = ddg._get_arrays_for_interpolation()
    all_diffs = ddg._get_arrays_for_diffusion()
    for _, (interp, _) in all_interps.items():
        ax1.plot(interp[:, 0], interp[:, 1])

    for _, (diff, _) in all_diffs.items():
        ax2.scatter(diff[:, 0], diff[:, 1])

    plt.tight_layout()
    plt.savefig("./data/plots/zelda/geometry.png")
    plt.show()
