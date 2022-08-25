"""
Samples a random walk for visualization and for playing.
"""
from pathlib import Path
import random

import torch

from geometries import DiscretizedGeometry
from utils.experiment import load_csv_as_map

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

from simulator import test_level_from_int_tensor

from utils.mario.plotting import save_level_from_array


def plot_levels_in_random_walk():
    """
    Saves several levels in a random walk,
    """
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )

    vae = VAEMarioHierarchical()
    vae.load_state_dict(torch.load(vae_path, map_location=vae.device))
    vae.eval()

    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )

    p_map = load_csv_as_map(path_to_gt)
    playable_zs = [k for k, v in p_map.items() if v == 1.0]
    dg = DiscretizedGeometry(p_map, "for_showing_random_walks", vae_path)

    _, levels = dg.diffuse(random.choice(playable_zs))

    # Saving levels in one random diffusion
    plots_path = Path("./data/plots/one_random_walk")
    plots_path.mkdir(exist_ok=True, parents=True)
    for i, level in enumerate(levels):
        save_level_from_array(
            f"./data/plots/one_random_walk/{i:05d}.png", level.detach().numpy()
        )


if __name__ == "__main__":
    plot_levels_in_random_walk()
