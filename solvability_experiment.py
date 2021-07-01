"""
In this script, we illuminate the latent
space by solvability using the A* agent
in MarioGAN.jar
"""
import json
from multiprocessing import Pool

import torch
import numpy as np
import matplotlib.pyplot as plt

from storage_interface import upload_blob_from_file, upload_blob_from_dict
from simulator import test_level_from_z
from vae_mario import VAEMario


def process(z, i):
    """
    This function simulates the level
    that the VAE generates from z.
    """
    z_dim = 2
    checkpoint = 100
    model_name = f"mariovae_zdim_{z_dim}_playesting_epoch_{checkpoint}"
    print(f"Loading model {model_name}")
    vae = VAEMario(16, 16, z_dim=z_dim)
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.eval()
    print(f"testing {z}")
    res = test_level_from_z(z, vae)
    thing_to_save = {
        "model_name": model_name,
        "z": tuple([float(zi) for zi in z.detach().numpy()]),
        **res,
    }
    path_to_exp = f"./data/simulator_experiment/initial_{i:05d}.json"
    with open(path_to_exp, "w") as fp:
        json.dump(thing_to_save, fp)
    print(f"saved {z}")

    upload_blob_from_file(
        "simulation_experiments", path_to_exp, f"{model_name}/{i:05d}.json"
    )
    print(f"uploaded {z} to gs.")


def main():
    x_lims = [-6, 6]
    y_lims = [-6, 6]

    n_rows = 50
    n_cols = 50
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])
    colors = []

    # print("starting the pool")
    with Pool(5) as p:
        p.starmap(process, zip(zs, range(len(zs))))
    # for i, z in enumerate(zs):
    #     process(z, i)


if __name__ == "__main__":
    main()
