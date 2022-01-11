"""
This interpolation takes a trained GPC and does the
interpolation according to its predictions.
"""
"""
Implements A* for interpolating between two points
in latent space.
"""
from typing import List, Tuple
from queue import PriorityQueue
from itertools import product

import torch as t
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from .base_interpolation import BaseInterpolation


class AStarGPCInterpolation(BaseInterpolation):
    def __init__(self, n_points_in_line: int, gpc: GaussianProcessClassifier):
        super().__init__(n_points_in_line=n_points_in_line)

        z1s = np.linspace(-5, 5, 50)
        z2s = np.linspace(-5, 5, 50)

        zs = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])

        res = gpc.predict_proba(zs)
        decision_boundary = 0.8

        predictions = [0 if p[1] < decision_boundary else 1.0 for p in res]

        positions = {
            (x, y): (i, j)
            for j, x in enumerate(z1s)
            for i, y in enumerate(reversed(z2s))
        }
        pred_dict = {(z[0], z[1]): pred for z, pred in zip(zs, predictions)}

        grid = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            grid[i, j] = pred_dict[z]

        self.z1 = z1s
        self.z2 = z2s
        self.zs = zs
        self.positions = positions
        self.inv_positions = {v: k for k, v in positions.items()}
        self.grid = grid
        self.predictions = np.array(predictions)

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

        # TODO: sometimes the code is raising this. It shouldn't happen
        # All elements are connected, even if it's through expensive paths.
        # Weird. Maybe the get_neighbors logic is faulty. Or maybe the parent
        # keeping is faulty.

        # neighbor getting is working properly.

        # It is exploring essentially all the graph. Why?!
        raise ValueError(f"z={z} and z_prime={z_prime} are not connected in the graph.")

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> t.Tensor:
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

        idxs = np.round(
            np.linspace(0, len(zs_in_path) - 1, self.n_points_in_line)
        ).astype(int)

        return t.from_numpy(zs_in_path[idxs]).type(t.float)


if __name__ == "__main__":
    # Loading a given trace
    a = np.load("./data/evolution_traces/bigger_trace.npz")
    z = a["zs"]
    p = a["playabilities"]

    gpc = GaussianProcessClassifier()
    gpc.fit(z, p)

    astar = AStarGPCInterpolation(10, gpc)
    interpolation = (
        astar.interpolate(
            t.Tensor([-4.0, -4.0]),
            t.Tensor([3.0, 3.0]),
        )
        .detach()
        .numpy()
    )

    plt.scatter(interpolation[:, 0], interpolation[:, 1])
    plt.show()
