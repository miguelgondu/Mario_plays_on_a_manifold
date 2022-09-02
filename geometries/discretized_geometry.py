from pathlib import Path
from typing import Dict, Tuple
from itertools import product

import torch as t
import numpy as np
import networkx as nx

from utils.experiment import load_arrays_as_map

from interpolations.discrete_interpolation import DiscreteInterpolation
from diffusions.discrete_diffusion import DiscreteDiffusion

from vae_models.vae_mario_obstacles import VAEWithObstacles

from vae_models.vae_zelda_obstacles import VAEZeldaWithObstacles

from .geometry import Geometry


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
        with_obstacles: bool = True,
    ) -> None:
        metric_vol_folder = Path(f"./data/processed/metric_volumes/{exp_name}/")
        metric_vol_folder.mkdir(exist_ok=True, parents=True)
        metric_vol_path = metric_vol_folder / f"{vae_path.stem}.npz"

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
            if with_obstacles:
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

        self.zs_of_metric_volumes = zs
        self.metric_volumes = metric_volumes

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

    def diffuse(self, z_0: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        return self.diffusion.run(z_0)

    def _get_edges_from_grid(self):
        """
        Constructs a list of all pairs ((i, j), (k, l)) s.t.
        a 1 is connected to a neighbouring 1.
        """
        n, m = self.grid.shape
        assert n == m

        pos_id = lambda i, j: i + (m * j)
        all_positions = list(product(range(n), range(m)))
        graph_edges = []
        for (i, j) in all_positions:
            if self.grid[i, j] != 1:
                continue

            neighbours = [
                (i + 1, j),
                (i - 1, j),
                (i, j + 1),
                (i, j - 1),
            ]
            for r, s in neighbours:
                if r >= n or r < 0:
                    continue
                if s >= m or s < 0:
                    continue

                if self.grid[r, s] == 1:
                    min_ = min(pos_id(i, j), pos_id(r, s))
                    max_ = max(pos_id(i, j), pos_id(r, s))
                    graph_edges.append((min_, max_))
                    print(f"added {(min_, max_)} to graph")

        # Removing duplicates
        graph_edges = list(set(graph_edges))

        return graph_edges

    def to_graph(self) -> nx.Graph:
        """
        Uses the internal grid to construct a graph of
        all the connected ones.
        """
        graph_edges = self._get_edges_from_grid()
        return nx.Graph(graph_edges)
