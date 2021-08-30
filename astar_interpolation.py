"""
Implements A* for interpolating between two points
in latent space.
"""
from typing import Dict, List, Tuple
from queue import LifoQueue, PriorityQueue

import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from base_interpolation import BaseInterpolation

Tensor = torch.Tensor


def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


class AStarInterpolation(BaseInterpolation):
    def __init__(self, n_points: int, model_name: str):
        super().__init__(n_points=n_points)

        # Computing the grid for A*
        df = pd.read_csv(
            f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
            index_col=0,
        )
        playability = df.groupby(["z1", "z2"]).mean()["marioStatus"]
        zs = np.array([z for z in playability.index])
        z1 = sorted(list(set(zs[:, 0])))
        z2 = sorted(list(set(zs[:, 1])), reverse=True)

        n_x = len(z1)
        n_y = len(z2)

        grid = np.zeros((n_y, n_x))
        positions = {}
        for (z1i, z2i), m in playability.iteritems():
            i, j = z2.index(z2i), z1.index(z1i)
            positions[(z1i, z2i)] = (i, j)
            grid[i, j] = m

        self.z1 = z1
        self.z2 = z2
        self.zs = np.array([[z1i, z2i] for z1i in z1 for z2i in z2])
        self.positions = positions
        self.inv_positions = {v: k for k, v in positions.items()}
        self.grid = grid

        # Construct a KDTree with the grid.
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

        if i - 1 > 0:
            if j - 1 > 0:
                neighbors.append((i - 1, j - 1))

            if j + 1 < height:
                neighbors.append((i - 1, j + 1))

            neighbors.append((i - 1, j))

        if i + 1 < width:
            if j - 1 > 0:
                neighbors.append((i + 1, j - 1))

            if j + 1 < height:
                neighbors.append((i + 1, j + 1))

            neighbors.append((i + 1, j))

        if j - 1 > 0:
            neighbors.append((i, j - 1))

        if j + 1 < height:
            neighbors.append((i, j + 1))

        random.shuffle(neighbors)

        return neighbors

    def heuristic(self, node: Tuple[int, int], final_position: Tuple[int, int]):
        """
        The heuristic for A*, which combines
        playability with Euclidean distance to goal.
        """
        z = np.array(self.inv_positions[node])
        z_prime = np.array(self.inv_positions[final_position])

        dist2 = np.sum((z - z_prime) ** 2)
        playability = self.grid[node]

        # Really high for non-playable levels
        return dist2 + 1 / (playability + 1e-6)

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
                    path_positions = [parent]
                    while parent is not None:
                        son, parent = parent, parents[parent]
                        path[son] = parent
                        path_positions.append(parent)

                    return path_positions

                h_neighbor = current_h + self.heuristic(neighbor, final_position)
                pq.put((h_neighbor, neighbor))

        raise ValueError(f"z={z} and z_prime={z_prime} are not connected in the graph.")

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
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

        return zs_in_path


if __name__ == "__main__":
    as_interpolation = AStarInterpolation(10, "mariovae_z_dim_2_overfitting_epoch_480")

    # print(as_interpolation.grid)
    # print(as_interpolation.positions)

    # Test that the grid makes sense.
    # Grid seems to be making sense
    # print((-6.0, -6.0), as_interpolation.positions[(-6.0, -6.0)])
    # print((-6.0, 6.0), as_interpolation.positions[(-6.0, 6.0)])
    # print((6.0, -6.0), as_interpolation.positions[(6.0, -6.0)])
    # print((6.0, 6.0), as_interpolation.positions[(6.0, 6.0)])

    z = torch.Tensor([5.0, -2.0])
    z_prime = torch.Tensor([5.0, 4.0])

    # z_in_grid = as_interpolation._query_tree(z.detach().numpy())
    # z_prime_in_grid = as_interpolation._query_tree(z_prime.detach().numpy())

    # print(z_in_grid, as_interpolation._query_grid(z_in_grid))
    # print(z_prime_in_grid, as_interpolation._query_grid(z_in_grid))

    # print(as_interpolation.a_star_path(z_in_grid, z_prime_in_grid))

    path = as_interpolation.interpolate(z, z_prime)
    _, ax = plt.subplots(1, 1)
    ax.imshow(as_interpolation.grid, extent=[-6, 6, -6, 6], cmap="Blues")
    ax.scatter(path[:, 0], path[:, 1], marker="x", c="r")
    plt.show()
