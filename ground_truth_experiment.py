"""
This experiment illuminates the latent space
by trying the decoded level several times using
Baumgarten's A* agent.

It creates an array that is can then simulated by
simulate_array.py
"""

import json
from pathlib import Path
from multiprocessing import Pool

import click
import torch
import numpy as np

from simulator import test_level_from_z
from vae_mario_hierarchical import VAEMarioHierarchical


# def process(i, z):
#     """
#     This function simulates the level
#     that the VAE generates from z.
#     """
#     print(f"Testing {z} (index {i})")

#     # Create the folder for the data
#     cwd = Path(".")
#     path_to_exp_folder = cwd / "data" / "ground_truth" / "another_vae_final"
#     path_to_exp_folder.mkdir(parents=True, exist_ok=True)

#     # Test the level 5 times
#     for iteration in range(5):
#         res = test_level_from_z(z, vae)
#         thing_to_save = {
#             "model_name": "another_vae_final",
#             "z": tuple([float(zi) for zi in z.detach().numpy()]),
#             "iteration": iteration,
#             **res,
#         }

#         path_to_exp = path_to_exp_folder / f"z_{i:05d}_{iteration}.json"
#         with open(path_to_exp, "w") as fp:
#             json.dump(thing_to_save, fp)

#     print(f"Processed z {z}.")


@click.command()
@click.option("--model-name", type=str, default="another_vae_final")
@click.option("--n-samples", type=int, default=5)
def main(model_name, n_samples):
    vae = VAEMarioHierarchical(device="cpu")
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.eval()

    x_lims = [-5, 5]
    y_lims = [-5, 5]

    n_grid = 50
    z1 = np.linspace(*x_lims, n_grid)
    z2 = np.linspace(*y_lims, n_grid)

    zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])
    cat = vae.decode(zs)

    levels = cat.sample((n_samples,)).detach().numpy()
    levels = levels.reshape(n_grid * n_grid * n_samples, *levels.shape[2:])

    zs = zs.detach().numpy()
    zs = np.repeat(zs, n_samples, axis=0)

    print(f"zs: {zs.shape}")
    print(f"levels: {levels.shape}")
    assert zs.shape[0] == levels.shape[0]

    print(f"Array saved for {model_name}.")
    np.savez(
        f"./data/arrays/ground_truth_{model_name}.npz",
        zs=zs,
        levels=levels,
    )


if __name__ == "__main__":
    main()
