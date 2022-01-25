from pathlib import Path
from typing import Dict, Tuple
import torch as t
from torch.distributions import Categorical
import numpy as np
from torch.distributions import MultivariateNormal

from interpolations.discrete_interpolation import DiscreteInterpolation
from .base_diffusion import BaseDiffusion

# TODO: implement this diffusion as a random walk on a graph.
class DiscreteDiffusion(BaseDiffusion):
    def __init__(
        self, vae_path: Path, p_map: Dict[tuple, float], n_steps: int = 50
    ) -> None:
        super().__init__(vae_path, p_map, n_steps)
        # Has nice methods for dealing with the graph.
        self.di = DiscreteInterpolation(vae_path, p_map)

    def run(self, z_0: t.Tensor) -> Tuple[t.Tensor]:
        # Random starting point (or the one provided)
        z_n = self.di._query_tree(z_0)
        rw = [z_n]
        for _ in range(self.n_steps):
            z_n = self.step(z_n)
            rw.append(z_n)

        zs_in_rw = t.Tensor(rw)
        vae = self._load_vae()
        levels = vae.decode(zs_in_rw).probs.argmax(dim=-1)

        return zs_in_rw, levels

    def step(self, z_n: np.ndarray) -> np.ndarray:
        """
        One step in the random walk. It does
        10 mini-steps and consolidates the result.
        """
        current_pos = self.di.positions[tuple(z_n)]
        for _ in range(10):
            all_neighbours = np.array(self.di.get_neighbors(current_pos))
            all_neighbour_weights = np.array(
                [self.di.weight(tuple(n)) for n in all_neighbours]
            )

            connected_mask = all_neighbour_weights < np.inf
            connected_neighbours = all_neighbours[connected_mask]
            connected_neighbour_w = all_neighbour_weights[connected_mask]

            # Compute next position and store it
            # Now that we're doing the discrete thing, this is
            # a little bit overkill. random.choice(connected_neightbours)
            # would do the trick just fine.
            _dist = Categorical(logits=t.from_numpy(connected_neighbour_w))
            next_idx = _dist.sample().item()
            current_pos = tuple(connected_neighbours[next_idx])

        return np.array(self.di.inv_positions[current_pos])
