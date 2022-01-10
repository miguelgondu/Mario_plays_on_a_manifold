"""
In this script, we do A* interpolation after updating
the understanding of the latent space using GPCs and AL.
"""
from itertools import product

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from interpolations.astar_gpc_interpolation import AStarGPCInterpolation

from vae_mario_obstacles import VAEWithObstacles

from baseline_metrics_for_different_z_dims import models
from evolving_playability import get_ground_truth


def plot_w_decision_boundary(decision_boundary: float):
    a = np.load("./data/evolution_traces/bigger_trace.npz")
    zs = a["zs"]
    p = a["playabilities"]

    gpc = GaussianProcessClassifier()
    gpc.fit(zs, p)

    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])

    res = gpc.predict_proba(bigger_grid)
    decision_boundary = 0.8

    predictions = [0 if p[1] < decision_boundary else 1.0 for p in res]

    positions = {(x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(z2s)}
    pred_dict = {(z[0], z[1]): pred for z, pred in zip(bigger_grid, predictions)}

    pred_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        pred_img[i, j] = pred_dict[z]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 10))

    gt_img = get_ground_truth()

    ax1.imshow(gt_img, extent=[-5, 5, -5, 5], cmap="Blues")
    ax1.set_title("Ground truth (2500 levels)")
    ax2.imshow(pred_img, extent=[-5, 5, -5, 5], cmap="Blues")
    ax2.set_title("GPC prediction with only\n 250 samples from latent space")

    plt.tight_layout()
    # plt.savefig("./data/plots/evolving_playability/ground_truth_vs_GPC.png")
    plt.show()


def avoiding_nonplayable():
    """
    The dual of Nicki's trick.
    """
    vae = VAEWithObstacles()
    vae.load_state_dict(t.load(f"./models/{models[2]}.pt"))
    a = np.load("./data/evolution_traces/bigger_trace.npz")
    zs = a["zs"]
    p = a["playabilities"]

    obstacles = zs[p == 0.0]
    vae.update_obstacles(t.from_numpy(obstacles).type(t.float))

    _, ax = plt.subplots(1, 1)
    vae.plot_w_geodesics(ax=ax)
    plt.show()


def a_star_interpolations_with_gpc():
    """
    Using A Star to avoid obstacles
    """
    a = np.load("./data/evolution_traces/bigger_trace.npz")
    zs = a["zs"]
    p = a["playabilities"]
    gpc = GaussianProcessClassifier()
    gpc.fit(zs, p)

    _, ax = plt.subplots(1, 1)

    astar = AStarGPCInterpolation(10, gpc)
    ax.imshow(astar.grid, extent=[-5, 5, -5, 5], cmap="Blues")
    data = t.Tensor(
        [
            [-4.0, -4.0],
            [-4.0, 4.0],
            [3.0, -4.0],
            # [4.0, 4.0],
            [0.0, -4.0],
            [0.0, 0.0],
            [2.0, 4.0],
            [-2.0, 4.0],
            [3.5, 3.5],
        ]
    )
    N = data.shape[0]
    for _ in range(5):
        idx = t.randint(N, (2,))
        interpolation = astar.interpolate(data[idx[0]], data[idx[1]])
        ax.scatter(interpolation[:, 0], interpolation[:, 1], c="red")

    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    # avoiding_nonplayable()
    a_star_interpolations_with_gpc()
