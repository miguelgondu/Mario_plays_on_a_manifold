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
import matplotlib.pyplot as plt

from storage_interface import upload_blob_from_file, upload_blob_from_dict
from simulator import test_level_from_z
from vae_mario import VAEMario
from train_vae import load_data

MODELS_PATH = "./models_experiment"


def process(i, z, z_dim, model_name):
    """
    This function simulates the level
    that the VAE generates from z.
    """
    print(f"Loading the model {model_name} (z dim {z_dim})")
    vae = VAEMario(14, 14, z_dim=z_dim)
    vae.load_state_dict(torch.load(f"{MODELS_PATH}/{model_name}.pt"))
    vae.eval()
    print(f"Testing {z} (index {i})")

    # Create the folder for the data
    cwd = Path(".")
    path_to_exp_folder = cwd / "data" / "playability_experiment" / model_name
    path_to_exp_folder.mkdir(parents=True, exist_ok=True)

    # Test the level 5 times
    for iteration in range(5):
        res = test_level_from_z(z, vae)
        thing_to_save = {
            "model_name": model_name,
            "z": tuple([float(zi) for zi in z.detach().numpy()]),
            "iteration": iteration,
            **res,
        }

        path_to_exp = path_to_exp_folder / f"z_{i:05d}_{iteration}.json"
        with open(path_to_exp, "w") as fp:
            json.dump(thing_to_save, fp)

    print(f"Processed z {z} for model {model_name}.")


@click.command()
@click.option("--z-dim", type=int, default=2)
@click.option(
    "--model-name", type=str, default="mariovae_z_dim_2_overfitting_epoch_480"
)
@click.option("--processors", type=int, default=5)
def main(z_dim, model_name, processors):
    # get_zs(model_name, z_dim)
    # with open(f"./grid_specs/{model_name}.json") as fp:
    #     grid_specs = json.load(fp)

    x_min, y_min = -5, -5
    x_max, y_max = 5, 5
    x_lims = [x_min - 1, x_max + 1]
    y_lims = [y_min - 1, y_max + 1]

    n_rows = 50
    n_cols = 50
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])

    print(f"Starting the Pool for {model_name}.")
    with Pool(processors) as p:
        p.starmap(
            process,
            zip(range(len(zs)), zs, repeat(z_dim), repeat(model_name)),
        )


if __name__ == "__main__":
    main()
