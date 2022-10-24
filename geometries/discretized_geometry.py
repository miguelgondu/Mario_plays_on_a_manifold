from pathlib import Path
from typing import DefaultDict, Dict, Tuple, Union, List
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
        self.graph = None
        self.graph_nodes = None

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

    def _position_to_graph_idx(self, i: int, j: int) -> int:
        _, m = self.grid.shape

        return i + (m * j)

    def _graph_idx_to_position(self, idx: int) -> Tuple[int, int]:
        _, m = self.grid.shape
        i = idx % m
        j = idx // m

        assert i == int(i)
        assert j == int(j)

        return int(i), int(j)

    def _get_adjacency_dict_from_grid(self):
        """
        Constructs a list of all pairs ((i, j), (k, l)) s.t.
        a 1 is connected to a neighbouring 1.
        """
        n, m = self.grid.shape
        assert n == m

        pos_id = lambda i, j: i + (m * j)
        all_positions = list(product(range(n), range(m)))
        adjacency_dict = {}
        for (i, j) in all_positions:
            if self.grid[i, j] != 1:
                continue

            adjacency_dict[(i, j)] = []

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
                    adjacency_dict[(i, j)].append((r, s))

        return adjacency_dict

    def from_latent_code_to_graph_node(self, latent_code: t.Tensor) -> t.Tensor:
        # TODO: implement this to wrap up the bayesian optimization.
        _, idxs = self.interpolation.kd_tree.query(latent_code)
        zs_in_the_grid = self.zs[idxs]
        graph_nodes = t.Tensor(
            [self.positions[(z1.item(), z2.item())] for (z1, z2) in zs_in_the_grid]
        )

        return graph_nodes

    def from_graph_node_to_latent_code(
        self, node: Union[Tuple[int, int], List[Tuple[int, int]]]
    ) -> t.Tensor:
        if isinstance(node, tuple) and isinstance(node[0], (int, float)):
            node_idx = self._position_to_graph_idx(*node)
            return self.from_graph_idx_to_latent_code(node_idx)
        elif isinstance(node, (list, t.Tensor, np.ndarray)):
            node_idxs = [self._position_to_graph_idx(n[0], n[1]) for n in node]
            return self.from_graph_idx_to_latent_code(node_idxs)
        ...

    def from_graph_idx_to_latent_code(
        self, graph_idx: Union[t.Tensor, np.ndarray, List, int]
    ) -> np.ndarray:
        if isinstance(graph_idx, int):
            return self.zs[graph_idx]
        elif isinstance(graph_idx, t.Tensor):
            # inverting the graph_idx to tuples of positions
            # positions = [
            #     self._graph_idx_to_position(idx.item()) for idx in graph_idx.flatten()
            # ]
            # TODO: Will this work?
            return self.zs[graph_idx.type(t.long).flatten()]
        elif isinstance(graph_idx, np.ndarray):
            return self.zs[graph_idx.astype(int).flatten()]
        elif isinstance(graph_idx, list):
            return self.zs[np.array(graph_idx).astype(int).flatten()]
        else:
            raise ValueError(...)

    def to_graph(self) -> nx.Graph:
        """
        Uses the internal grid to construct a graph of
        all the connected ones.
        """
        if self.graph is None:
            adjacency = self._get_adjacency_dict_from_grid()
            self.graph = nx.Graph(adjacency)
            self.graph_nodes = list(self.graph.nodes())

        return self.graph
