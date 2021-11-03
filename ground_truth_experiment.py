"""
This experiment illuminates the latent space
by trying the decoded level several times using
Baumgarten's A* agent.
"""

import os
import json
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat

import click
import torch
import numpy as np

from simulator import test_level_from_z
from vae_mario_hierarchical import VAEMarioHierarchical


def process(i, z):
    """
    This function simulates the level
    that the VAE generates from z.
    """
    vae = VAEMarioHierarchical(device="cpu")
    vae.load_state_dict(torch.load(f"./models/another_vae_final.pt"))
    vae.eval()
    print(f"Testing {z} (index {i})")

    # Create the folder for the data
    cwd = Path(".")
    path_to_exp_folder = cwd / "data" / "ground_truth" / "another_vae_final"
    path_to_exp_folder.mkdir(parents=True, exist_ok=True)

    # Test the level 5 times
    for iteration in range(5):
        res = test_level_from_z(z, vae)
        thing_to_save = {
            "model_name": "another_vae_final",
            "z": tuple([float(zi) for zi in z.detach().numpy()]),
            "iteration": iteration,
            **res,
        }

        path_to_exp = path_to_exp_folder / f"z_{i:05d}_{iteration}.json"
        with open(path_to_exp, "w") as fp:
            json.dump(thing_to_save, fp)

    print(f"Processed z {z}.")


@click.command()
@click.option("--processes", type=int, default=5)
def main(processes):
    x_lims = [-5, 5]
    y_lims = [-5, 5]

    n_rows = 50
    n_cols = 50
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])

    print(f"Starting the Pool.")
    with Pool(processes) as p:
        p.starmap(
            process,
            zip(range(len(zs)), zs),
        )


if __name__ == "__main__":
    main()
