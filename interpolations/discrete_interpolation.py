"""
Replaces Djisktra's interpolation on GPC with A star,
making it generic at the same time. Now it only needs a p_map.
"""
from pathlib import Path
from typing import Dict, Tuple, List
from queue import PriorityQueue
import random

import torch as t
import numpy as np
from scipy.spatial import cKDTree
from utils.experiment import grid_from_map, positions_from_map

from vae_mario_hierarchical import VAEMarioHierarchical

from .base_interpolation import BaseInterpolation


class DiscreteInterpolation(BaseInterpolation):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_points_in_line: int = 10
    ):
        super().__init__(vae_path, p_map, n_points_in_line)

        zs = np.array([z for z in p_map.keys()])
        z1s = np.array(sorted(list(set([z[0] for z in zs]))))
        z2s = np.array(sorted(list(set([z[1] for z in zs]))))

        self.z1 = z1s
        self.z2 = z2s
        self.zs = zs
        self.positions = positions_from_map(p_map)
        self.inv_positions = {v: k for k, v in self.positions.items()}
        self.grid = grid_from_map(p_map)
        self.predictions = np.array(p_map.values())
        self.kd_tree = cKDTree(self.zs)

    def _query_tree(self, z: np.ndarray) -> np.ndarray:
        """
        Returns the nearest point to {z} in self.grid
        """
        idx = self.kd_tree.query(z)[1]
        return self.zs[idx, :]

    def _query_grid(self, z: np.ndarray) -> float:
        """
        Queries the grid at a certain point
        and returns the playability therein.
        """
        z_in_grid = self._query_tree(z)
        z1i, z2i = z_in_grid
        i, j = self.positions[z1i, z2i]
        return self.grid[i, j]

    def get_neighbors(self, position: Tuple[int, int]):
        """
        Given (i, j) in {position}, returns
        all 8 neighbors (i-1, j-1), ..., (i+1, j+1).
        """
        i, j = position
        width, height = self.grid.shape

        if i < 0 or i >= width:
            raise ValueError(f"Position is out of bounds in x: {position}")

        if j < 0 or j >= height:
            raise ValueError(f"Position is out of bounds in x: {position}")

        neighbors = []

        if i - 1 >= 0:
            if j - 1 >= 0:
                neighbors.append((i - 1, j - 1))

            if j + 1 < height:
                neighbors.append((i - 1, j + 1))

            neighbors.append((i - 1, j))

        if i + 1 < width:
            if j - 1 >= 0:
                neighbors.append((i + 1, j - 1))

            if j + 1 < height:
                neighbors.append((i + 1, j + 1))

            neighbors.append((i + 1, j))

        if j - 1 >= 0:
            neighbors.append((i, j - 1))

        if j + 1 < height:
            neighbors.append((i, j + 1))

        random.shuffle(neighbors)

        return neighbors

    def weight(self, node: Tuple[int, int]) -> float:
        """
        Weight of a node.
        """
        playability = self.grid[node]
        if playability == 1.0:
            return 1.0
        elif playability == 0.0:
            return np.inf
        else:
            raise ValueError(f"Unexpected value for playability: {playability}")

    def heuristic(self, node: Tuple[int, int], final_pos: Tuple[int, int]) -> float:
        """
        The heuristic for A*, which combines
        playability with Euclidean distance to goal.
        """
        z = np.array(self.inv_positions[node])
        z_prime = np.array(self.inv_positions[final_pos])
        dist2 = np.sum((z - z_prime) ** 2)
        w = self.weight(node)

        return w + dist2

    def a_star_path(self, z: np.ndarray, z_prime: np.ndarray) -> List[np.ndarray]:
        parents = {}
        first_position = self.positions[tuple(z)]
        final_position = self.positions[tuple(z_prime)]

        h_first = self.heuristic(first_position, final_position)
        pq = PriorityQueue()
        pq.put((h_first, first_position))
        visited_positions = set([first_position])
        parents[first_position] = None

        while not pq.empty():
            current_h, position = pq.get()
            neighbors = self.get_neighbors(position)
            for neighbor in neighbors:
                if neighbor in visited_positions:
                    continue

                visited_positions.add(neighbor)
                parents[neighbor] = position

                if neighbor == final_position:
                    # We are done
                    parent = position
                    son = neighbor
                    path = {son: parent}
                    path_positions = [son, parent]
                    while parent is not None:
                        son, parent = parent, parents[parent]
                        path[son] = parent
                        path_positions.append(parent)

                    return path_positions

                h_neighbor = current_h + self.heuristic(neighbor, final_position)
                pq.put((h_neighbor, neighbor))

        raise ValueError(f"z={z} and z_prime={z_prime} are not connected in the graph.")

    def _full_interpolation(self, z: t.Tensor, z_prime: t.Tensor) -> t.Tensor:
        # Find the closest point to z on the grid, and join
        # by a line.
        # same thing for z_prime.
        z_in_grid = self._query_tree(z.detach().numpy())
        z_prime_in_grid = self._query_tree(z_prime.detach().numpy())

        # Interpolate in the grid itself using A*
        path_positions = self.a_star_path(z_in_grid, z_prime_in_grid)

        path_positions.reverse()
        path_positions = path_positions[1:]
        zs_in_path = np.array([self.inv_positions[p] for p in path_positions])

        return t.from_numpy(zs_in_path).type(t.float)

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> Tuple[t.Tensor]:
        # Get the full interpolation
        zs_in_path = self._full_interpolation(z, z_prime)

        idxs = np.round(
            np.linspace(0, len(zs_in_path) - 1, self.n_points_in_line)
        ).astype(int)
        # assert t.isclose(zs_in_path[0], z, atol=1e-1)
        # assert (zs_in_path[0] == z).all()
        # assert (zs_in_path[-1] == z_prime).all()

        zs_in_path = zs_in_path[idxs]
        vae = self._load_vae()
        levels = vae.decode(zs_in_path).probs.argmax(dim=-1)

        return zs_in_path, levels
