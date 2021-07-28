"""
This script contains the implementation
of randomly selecting two points in latent
space, creating the line between them,
and playing 20 levels in said line.
"""

from typing import List
import torch
import matplotlib.pyplot as plt

from vae_geometry import VAEGeometry


def random_interpolation_experiment(
    n_pairs: int,
    x_lims: List[float],
    y_lims: List[float],
    n_points_in_line: int = 20,
):
    """
    This function selects n_pairs of points
    between x_lims and y_lims, interpolates
    them with a line and returns a list of levels
    in between.
    """
    # If you load this here, torch locks in multiprocessing
    # afterward. SO WEIRD.
    # vae = VAEGeometry()
    # vae.load_state_dict(torch.load(f"models_experiment/{model_name}.pt"))
    # vae.eval()

    lower = torch.Tensor([x_lims[0], y_lims[0]])
    upper = torch.Tensor([x_lims[1], y_lims[1]])

    lines_of_zs = []

    for n_pair in range(n_pairs):
        z1 = lower + (upper - lower) * torch.rand((1, 2))
        z2 = lower + (upper - lower) * torch.rand((1, 2))

        # print(z1, z2)
        t = torch.linspace(0, 1, n_points_in_line).unsqueeze(-1)
        line = (1 - t) * z1 + t * z2
        # levels = vae.decode(line)
        for z in line:
            lines_of_zs.append({"n_line": n_pair, "z": z})

    return lines_of_zs


if __name__ == "__main__":
    random_interpolation_experiment(100, (-6, 6), (-6, 6))
