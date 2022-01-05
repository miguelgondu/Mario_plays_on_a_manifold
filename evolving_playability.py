"""
This script evolves a playability manifold using
something similar to B.O.

Let's start in 2D.
"""
from itertools import product
import multiprocessing as mp
from typing import Tuple
import json

import torch as t
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from vae_mario_hierarchical import VAEMarioHierarchical
from simulator import test_level_from_decoded_tensor
from mario_utils.plotting import get_img_from_level
from simulate_array import _simulate_array

"""
The core idea: we sample random points to bootstrap a playability manifold.
"""


def load_vae() -> VAEMarioHierarchical:
    model_name = "16388917374131331_mariovae_zdim_2_normal_final"
    vae = VAEMarioHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))

    return vae


def simulate_level(level: np.ndarray, processes: int, repeats: int) -> int:
    levels = np.repeat(level.reshape(1, 14, 14), repeats, axis=0)
    with mp.Pool(processes) as p:
        results = p.map(test_level_from_decoded_tensor, levels)

    playabilities = [r["marioStatus"] for r in results]
    return int(np.any(playabilities))


def get_initial_data():
    vae = load_vae()

    # Sampling random levels, playing them, and plotting their playability.
    # zs = vae.p_z.sample((100,))
    zs = t.Tensor(
        [[z1, z2] for z1, z2 in product(t.linspace(-5, 5, 10), t.linspace(-5, 5, 10))]
    )
    levels = vae.decode(zs).probs.argmax(dim=-1)

    np.savez(
        "./data/arrays/one_hundred_levels_in_a_grid.npz",
        zs=zs.detach().numpy(),
        levels=levels.detach().numpy(),
    )

    _simulate_array("./data/arrays/one_hundred_levels_in_a_grid.npz", 5, 5)


def load_initial_data() -> Tuple[np.ndarray]:
    df = pd.read_csv(
        "./data/array_simulation_results/one_hundred_levels_in_a_grid.csv", index_col=0
    )
    by_z = df.groupby("z")["marioStatus"].mean()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    playabilities = by_z.values
    playabilities[playabilities > 0.0] = 1.0

    return zs, 1 - playabilities


if __name__ == "__main__":
    # get_initial_data()
    zs, playabilities = load_initial_data()

    gpc = GaussianProcessClassifier()
    gpc.fit(zs, playabilities)

    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
    res, var = gpc.predict_proba(bigger_grid, return_var=True)
    var_dict = {(z[0], z[1]): v for z, v in zip(bigger_grid, var)}

    positions = {(x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(z2s)}

    var_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        var_img[i, j] = var_dict[z]

    # _, ax = plt.subplots(1, 1)
    # ax.imshow(var_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
    # ax.scatter(zs[:, 0], zs[:, 1], c=playabilities)
    # plt.show()
    # plt.close()

    inv_var_dict = {v: k for k, v in var_dict.items()}
    v = np.max(var)

    vae = load_vae()
    fig1, axes1 = plt.subplots(8, 8, figsize=(7 * 8, 7 * 8))
    fig2, axes2 = plt.subplots(8, 8, figsize=(7 * 8, 7 * 8))
    for ax1, ax2 in zip(axes1.flatten(), axes2.flatten()):
        next_point = inv_var_dict[v]
        print(f"trying {next_point}")
        level = vae.decode(t.Tensor(next_point)).probs.argmax(dim=-1)
        p = simulate_level(level, 5, 5)
        zs = np.vstack((zs, np.array(next_point)))
        playabilities = np.concatenate((playabilities, np.array([1 - p])))

        gpc = GaussianProcessClassifier()
        gpc.fit(zs, playabilities)

        z1s = np.linspace(-5, 5, 50)
        z2s = np.linspace(-5, 5, 50)

        bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
        res, var = gpc.predict_proba(bigger_grid, return_var=True)
        p_dict = {(z[0], z[1]): r[0] for z, r in zip(bigger_grid, res)}
        var_dict = {(z[0], z[1]): v for z, v in zip(bigger_grid, var)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(z2s)
        }

        var_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            var_img[i, j] = var_dict[z]

        p_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            p_img[i, j] = p_dict[z]

        ax1.imshow(var_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
        ax1.scatter([next_point[0]], [next_point[1]], c="red", marker="x")
        ax1.axis("off")

        ax2.imshow(p_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
        ax2.scatter([next_point[0]], [next_point[1]], c="red", marker="x")
        ax2.axis("off")
        # plt.show()
        # plt.close()

        inv_var_dict = {v: k for k, v in var_dict.items()}
        v = np.max(var)

    fig1.tight_layout()
    fig1.savefig("./data/plots/evolving_playability/several_iterations_variance.png")
    fig2.tight_layout()
    fig2.savefig(
        "./data/plots/evolving_playability/several_iterations_unplayability.png"
    )
    plt.show()
    plt.close()
    # raise
