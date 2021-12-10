"""
This script saves the arrays for all baseline experiments,
for several dimensions.

See table [ref] in the research log.
"""
from typing import List

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_mario_hierarchical import VAEMarioHierarchical
from train_vae import load_data

from interpolations.linear_interpolation import LinearInterpolation

from diffusions.normal_diffusion import NormalDifussion
from diffusions.baseline_diffusion import BaselineDiffusion

models = {
    2: "16388917374131331_mariovae_zdim_2_normal_final",
    8: "1638894528256156_mariovae_zdim_8_normal_final",
    32: "16388927503019269_mariovae_zdim_32_normal_final",
    64: "16388929591033669_mariovae_zdim_64_normal_final",
}


def get_random_pairs(
    points: t.Tensor, n_pairs: int = 20, seed: int = None
) -> List[t.Tensor]:
    if seed is not None:
        np.random.seed(seed)
    idx1 = np.random.choice(len(points), size=n_pairs, replace=False)
    idx2 = np.random.choice(len(points), size=n_pairs, replace=False)
    while np.any(idx1 == idx2):
        idx2 = np.random.choice(len(points), size=n_pairs, replace=False)

    pairs_1 = points[idx1]
    pairs_2 = points[idx2]

    return pairs_1, pairs_2


def save_arrays_for_model(model_name: str, z_dim: int) -> None:
    vae = VAEMarioHierarchical(z_dim=z_dim)
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))

    train_, test_ = load_data(only_playable=True)
    levels = t.cat((train_, test_))
    encodings = vae.encode(levels).mean

    # Saving linear interpolations
    li = LinearInterpolation()
    zs_1, zs_2 = get_random_pairs(encodings, n_pairs=20)
    for line_i, (z1, z2) in enumerate(zip(zs_1, zs_2)):
        line = li.interpolate(z1, z2)
        levels_in_line = vae.decode(line).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/baselines/{model_name}_linear_interpolation_{line_i:03d}.npz",
            zs=line.detach().numpy(),
            levels=levels_in_line.detach().numpy(),
        )

    # Saving diffusions
    nd = NormalDifussion(10, scale=0.5)
    bd = BaselineDiffusion(10, step_size=0.5)

    for run_i in range(20):
        normal_diffusion = nd.run(encodings)
        levels_normal = vae.decode(normal_diffusion).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/baselines/{model_name}_normal_diffusion_{run_i:03d}.npz",
            zs=normal_diffusion.detach().numpy(),
            levels=levels_normal.detach().numpy(),
        )

        baseline_diffusion = bd.run(encodings)
        levels_baseline = vae.decode(baseline_diffusion).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/baselines/{model_name}_baseline_diffusion_{run_i:03d}.npz",
            zs=baseline_diffusion.detach().numpy(),
            levels=levels_baseline.detach().numpy(),
        )

    print(f"Arrays saved for model {model_name}")


def inspect_z_dim_2():
    # Visual inspection for 2 dimensions
    model_2 = models[2]
    _, ax = plt.subplots(1, 1)

    vae = VAEMarioHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{model_2}.pt"))

    train_, test_ = load_data(only_playable=True)
    levels = t.cat((train_, test_))
    encodings = vae.encode(levels).mean.detach().numpy()

    ax.scatter(encodings[:, 0], encodings[:, 1], alpha=0.2)

    # Loading up the arrays:
    for line_i in range(20):
        line = np.load(
            f"./data/arrays/baselines/{model_2}_linear_interpolation_{line_i:03d}.npz"
        )
        zs = line["zs"]
        ax.scatter(zs[:, 0], zs[:, 1])

    # New fig. for the diffusions.
    _, ax2 = plt.subplots(1, 1)
    _, ax3 = plt.subplots(1, 1)
    ax2.scatter(encodings[:, 0], encodings[:, 1], alpha=0.2)
    ax3.scatter(encodings[:, 0], encodings[:, 1], alpha=0.2)
    for run_i in range(20):
        normal_d = np.load(
            f"./data/arrays/baselines/{model_2}_normal_diffusion_{run_i:03d}.npz"
        )
        baseline_d = np.load(
            f"./data/arrays/baselines/{model_2}_baseline_diffusion_{run_i:03d}.npz"
        )
        zs_n = normal_d["zs"]
        zs_b = baseline_d["zs"]

        ax2.scatter(zs_n[:, 0], zs_n[:, 1], c="red", alpha=0.5)
        ax3.scatter(zs_b[:, 0], zs_b[:, 1], c="red", alpha=0.5)

    plt.show()
    plt.close()


if __name__ == "__main__":
    for z_dim, model_name in models.items():
        save_arrays_for_model(model_name, z_dim)

    inspect_z_dim_2()
