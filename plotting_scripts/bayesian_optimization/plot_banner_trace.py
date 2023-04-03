"""
This script plots the banner R.B.O. trace.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.experiment.bayesian_optimization import load_geometry
from utils.mario.plotting import plot_level_from_array

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def plot_trace(max_iteration: int = None):
    # Loading the trace
    model_id = 1
    scale = 1.3
    scale_as_string = f"{int(scale * 100)}"
    iteration = 9

    trace_path = (
        ROOT_DIR
        / "data"
        / "bayesian_optimization"
        / "traces"
        / f"restricted_bo_{scale_as_string}_{model_id}_{iteration}.npz"
    )
    arr = np.load(trace_path)
    if max_iteration is None:
        max_iteration = 25

    latent_codes = arr["zs"][2 : 2 + max_iteration]
    playability = arr["playability"][2 : 2 + max_iteration]
    jumps = arr["jumps"][2 : 2 + max_iteration]
    levels = arr["levels"][2 : 2 + max_iteration]

    # Loading the geometry
    discrete_geometry = load_geometry(
        mean_scale=scale,
        model_id=model_id,
        name=f"bo_for_model_{model_id}_scale_{scale_as_string}",
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(discrete_geometry.grid, cmap="Blues", extent=[-5, 5, -5, 5], alpha=0.45)

    for i in range(max_iteration - 1):
        # Plot an arrow going from z_i to z_i+1
        ax.quiver(
            [latent_codes[i][0]],
            [latent_codes[i][1]],
            [latent_codes[i + 1][0] - latent_codes[i][0]],
            [latent_codes[i + 1][1] - latent_codes[i][1]],
            scale=1.0,
            scale_units="xy",
            angles="xy",
            alpha=0.7,
            color="black",
            edgecolor="black",
        )

    ax.scatter(
        [latent_codes[0, 0]],
        [latent_codes[0, 1]],
        marker="d",
        c="#CA054D",
        edgecolors="black",
        linewidths=1.5,
        s=75,
        label="Initial point",
    )
    ax.scatter(
        latent_codes[1:max_iteration, 0],
        latent_codes[1:max_iteration, 1],
        # marker="x",
        edgecolors="black",
        c="#E59500",
        label="Iterations",
        s=50,
    )
    if max_iteration == 25:
        ax.scatter(
            [latent_codes[24, 0]],
            [latent_codes[24, 1]],
            marker="d",
            c="#B0E298",
            edgecolors="black",
            linewidths=1.5,
            s=75,
            label="Optima",
        )

    # ax.legend()
    ax.axis("off")

    ax.set_xlim([-3.5, 5.0])
    ax.set_ylim([-3.5, 3])

    PLOTS_DIR = ROOT_DIR / "data" / "plots" / "modl_presentation"
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(PLOTS_DIR / f"video_{max_iteration}.png", dpi=120, bbox_inches="tight")

    # Saving the levels
    # fig_initial, ax_initial = plt.subplots(1, 1, figsize=(6, 6))
    # plot_level_from_array(ax_initial, levels[0][0])
    # fig_initial.savefig(PLOTS_DIR / "initial.png", dpi=120, bbox_inches="tight")

    fig_optima, ax_optima = plt.subplots(1, 1, figsize=(6, 6))
    plot_level_from_array(ax_optima, levels[max_iteration - 1][0])
    fig_optima.savefig(
        PLOTS_DIR / f"level_{max_iteration - 1}.png", dpi=120, bbox_inches="tight"
    )
    # assert jumps.argmax() == 25

    # plt.show()
    plt.close()


if __name__ == "__main__":
    for max_iteration in range(1, 26):
        plot_trace(max_iteration=max_iteration)
