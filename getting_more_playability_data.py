from typing import Tuple
from itertools import product

import torch
import numpy as np

from vae_mario_hierarchical import VAEMarioHierarchical


def create_more_levels() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads up one of the hierarchical networks and
    gets more levels by sampling on a fine grid in latent space.

    returns zs and levels
    """
    n_grid = 50
    n_samples = 3

    print("Loading the network")
    vaeh = VAEMarioHierarchical()
    vaeh.load_state_dict(torch.load("./models/hierarchical_for_log_final.pt"))
    vaeh.eval()

    print("Getting the zs")
    zs = torch.Tensor(
        [
            [z1i, z2i]
            for z1i, z2i in product(
                np.linspace(-5, 5, n_grid), np.linspace(-5, 5, n_grid)
            )
        ]
    )
    print(zs)

    print("Decoding")
    cat = vaeh.decode(zs)

    print("Sampling")
    levels = cat.sample((n_samples,)).detach().numpy()
    print(levels.shape)

    print("Flattening")
    levels = levels.reshape(n_grid * n_grid * n_samples, *levels.shape[2:])

    return zs.detach().numpy(), levels


if __name__ == "__main__":
    zs, levels = create_more_levels()
    print(levels)
    print(levels.shape)

    # Saving the array
    np.savez("./data/arrays/samples_for_playability.npz", zs=zs, levels=levels)
    print("Array saved!")
