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
from analysis_scripts.utils import zs_and_playabilities

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


def simulate_level(level: t.Tensor, processes: int, repeats: int) -> int:
    levels = np.repeat(level.reshape(1, 14, 14), repeats, axis=0)
    with mp.Pool(processes) as p:
        results = p.map(test_level_from_decoded_tensor, levels)

    playabilities = [r["marioStatus"] for r in results]
    res = np.mean(playabilities)
    if res > 0.0:
        res = 1.0

    return int(res)


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

    return zs, playabilities


def get_ground_truth(plot=False) -> np.ndarray:
    zs, playabilities = zs_and_playabilities(
        "./data/array_simulation_results/ground_truth_argmax_True_16388917374131331_mariovae_zdim_2_normal_final.csv"
    )

    p_dict = {(z[0], z[1]): p for z, p in zip(zs, playabilities)}

    # z1s = np.linspace(-5, 5, 50)
    # z2s = np.linspace(-5, 5, 50)
    z1s = np.array(sorted(list(set([z[0] for z in zs]))))
    z2s = np.array(sorted(list(set([z[1] for z in zs]))))

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(reversed(z2s))
    }

    p_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        p_img[i, j] = p_dict[z]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(p_img, extent=[-5, 5, -5, 5], cmap="Blues")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            "./data/plots/ground_truths/16388917374131331_mariovae_zdim_2_normal_final.png"
        )
        plt.close(fig)

    return p_img


def MVP():
    # get_initial_data()
    zs, playabilities = load_initial_data()

    gpc = GaussianProcessClassifier()
    gpc.fit(zs, playabilities)

    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
    res, var = gpc.predict_proba(bigger_grid, return_var=True)
    var_dict = {(z[0], z[1]): v for z, v in zip(bigger_grid, var)}

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(reversed(z2s))
    }

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
    fig3, axes3 = plt.subplots(8, 8, figsize=(7 * 8, 7 * 8))
    for ax1, ax2, ax3 in zip(axes1.flatten(), axes2.flatten(), axes3.flatten()):
        next_point = inv_var_dict[v]
        level = vae.decode(t.Tensor(next_point)).probs.argmax(dim=-1)
        p = simulate_level(level, 5, 5)
        zs = np.vstack((zs, np.array(next_point)))
        playabilities = np.concatenate((playabilities, np.array([p])))
        print(f"trying {next_point}, got {p}")

        gpc = GaussianProcessClassifier()
        gpc.fit(zs, playabilities)

        z1s = np.linspace(-5, 5, 50)
        z2s = np.linspace(-5, 5, 50)

        bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
        res, var = gpc.predict_proba(bigger_grid, return_var=True)
        predictions = gpc.predict(bigger_grid)
        p_dict = {(z[0], z[1]): r[0] for z, r in zip(bigger_grid, res)}
        var_dict = {(z[0], z[1]): v for z, v in zip(bigger_grid, var)}
        pred_dict = {(z[0], z[1]): pred for z, pred in zip(bigger_grid, predictions)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(z2s)
        }

        var_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            var_img[i, j] = var_dict[z]

        p_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            p_img[i, j] = p_dict[z]

        pred_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            pred_img[i, j] = pred_dict[z]

        ax1.imshow(var_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
        ax1.scatter([next_point[0]], [next_point[1]], c="red", marker="x")
        ax1.axis("off")

        ax2.imshow(p_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
        ax2.scatter([next_point[0]], [next_point[1]], c="red", marker="x")
        ax2.axis("off")

        ax3.imshow(pred_img, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues")
        ax3.scatter([next_point[0]], [next_point[1]], c="red", marker="x")
        ax3.axis("off")
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
    fig3.tight_layout()
    fig3.savefig("./data/plots/evolving_playability/several_iterations_predictions.png")
    # plt.show()
    plt.close()

    # Saving playtrace
    np.savez(
        "./data/evolution_traces/trace_simple_RBF_kernel.npz",
        zs=zs,
        playabilities=playabilities,
    )
    # raise

    # get_ground_truth(plot=True)


def load_trace(path: str) -> Tuple[np.ndarray]:
    a = np.load(path)
    return a["zs"], a["playabilities"]


def query(gpc) -> np.ndarray:
    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
    _, var = gpc.predict_proba(bigger_grid, return_var=True)
    next_point = bigger_grid[np.argmax(var)]

    return next_point


def run(
    gpc_kwargs: dict,
    n_iterations: int,
    zs: np.ndarray = None,
    playabilities: np.ndarray = None,
    name="trace",
):
    vae = load_vae()
    gpc = GaussianProcessClassifier(**gpc_kwargs)
    if zs is None or playabilities is None:
        zs, playabilities = load_initial_data()

    gpc.fit(zs, playabilities)

    for _ in range(n_iterations):
        # Get next point to query
        (next_point,) = query(gpc)
        next_level = vae.decode(t.Tensor(next_point)).probs.argmax(dim=-1)
        p = simulate_level(next_level, 5, 5)
        if zs is None:
            zs = np.array([next_point])
        else:
            zs = np.vstack((zs, next_point))
        if playabilities is None:
            playabilities = np.array([p])
        else:
            playabilities = np.concatenate((playabilities, np.array([p])))

        print(f"Tested {next_point}")
        gpc = GaussianProcessClassifier(**gpc_kwargs)
        gpc.fit(zs, playabilities)

    np.savez(f"./data/evolution_traces/{name}.npz", zs=zs, playabilities=playabilities)


if __name__ == "__main__":
    zs, playabilities = load_trace("./data/evolution_traces/trace.npz")
    run({"kernel": None}, 300, zs=zs, playabilities=playabilities, name="bigger_trace")
