from itertools import product
from pathlib import Path
from typing import Tuple

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

# from evolving_playability import get_ground_truth
from analysis_scripts.other_utils import zs_and_playabilities


def load_trace(path) -> Tuple[np.ndarray]:
    a = np.load(path)
    zs = a["zs"]
    p = a["playabilities"]

    return zs, p


def get_ground_truth(path: Path, save_at: Path = None) -> np.ndarray:
    """
    Gets ground truth from array simulation in path.
    """
    zs, playabilities = zs_and_playabilities(path)

    p_dict = {(z[0], z[1]): p for z, p in zip(zs, playabilities)}

    z1s = np.array(sorted(list(set([z[0] for z in zs]))))
    z2s = np.array(sorted(list(set([z[1] for z in zs]))))

    positions = {
        (x, y): (i, j) for i, y in enumerate(reversed(z2s)) for j, x in enumerate(z1s)
    }

    p_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        p_img[i, j] = p_dict[z]

    if save_at is not None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(p_img, extent=[-5, 5, -5, 5], cmap="Blues")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_at)
        plt.close(fig)

    return p_img


def plot_all_ground_truths():
    """
    Plots the ground truths for the five VAEs.
    """
    paths = Path("./data/array_simulation_results/five_vaes/ground_truth").glob(
        "vae_*_id_*.csv"
    )
    plots_path = Path("./data/plots/five_vaes/ground_truth")
    plots_path.mkdir(exist_ok=True, parents=True)
    for path in paths:
        print(f"Processing {path}.")
        save_at = plots_path / path.name.replace(".csv", ".png")
        get_ground_truth(path, save_at=save_at)


def load_vae(path):
    """
    Loads the VAE into the proper device
    """
    vae = VAEMarioHierarchical()
    device = vae.device
    vae.load_state_dict(t.load(path, map_location=device))

    return vae


def plot_all_grids():
    """
    plots all grids for the five vaes
    """
    model_paths = Path("./models").glob("vae_*_id_*_final.pt")
    plots_path = Path("./data/plots/five_vaes/grids")
    plots_path.mkdir(exist_ok=True, parents=True)

    for path in model_paths:
        save_at = plots_path / path.name.replace(".pt", ".png")
        vae = load_vae(path)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        vae.plot_grid(ax=ax)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_at)


def verify_length_of_all_traces():
    traces = Path("./data/evolution_traces/five_vaes").glob("*.npz")
    for path in traces:
        zs, _ = load_trace(path)
        print(path, len(zs))


def save_video_for_all_traces():
    traces = Path("./data/evolution_traces/ten_vaes").glob("*.npz")
    videos_path = Path("./data/plots/ten_vaes/AL_for_videos")
    videos_path.mkdir(exist_ok=True, parents=True)

    for f in traces:
        if "id_1" in f.name:
            continue

        all_zs, all_playabilities = load_trace(f)
        m_iters = min(len(all_zs) - 100, 300)
        for m in range(m_iters):
            # print(str(f), len(zs))
            print(f"{f.name}, {m:03d}/{m_iters}", flush=True, end="\r")
            zs = all_zs[: 100 + m]
            playabilities = all_playabilities[: 100 + m]

            kernel = 1.0 * Matern(nu=3 / 2) + 1.0 * WhiteKernel()
            gpc = GaussianProcessClassifier(kernel=kernel)
            gpc.fit(zs, playabilities)

            z1s = np.linspace(-5, 5, 50)
            z2s = np.linspace(-5, 5, 50)

            bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
            res = gpc.predict_proba(bigger_grid)

            p_dict = {(z[0], z[1]): r[1] for z, r in zip(bigger_grid, res)}

            positions = {
                (x, y): (i, j)
                for j, x in enumerate(z1s)
                for i, y in enumerate(reversed(z2s))
            }

            p_img = np.zeros((len(z2s), len(z1s)))
            for z, (i, j) in positions.items():
                p_img[i, j] = p_dict[z]

            fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 7 * 1))

            plt.title(f"{f.name}, m={m:04d}")
            ax.imshow(p_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
            ax.scatter(
                zs[:, 0], zs[:, 1], c=playabilities, cmap="Wistia", vmin=0.0, vmax=1.0
            )
            fig.tight_layout()
            fig.savefig(videos_path / f"{m:04d}_{f.name.replace('.npz', '.png')}")
            plt.close("all")

            if m == m_iters - 1:
                print()


if __name__ == "__main__":
    # plot_all_ground_truths()
    # plot_all_grids()
    # verify_length_of_all_traces()
    save_video_for_all_traces()
