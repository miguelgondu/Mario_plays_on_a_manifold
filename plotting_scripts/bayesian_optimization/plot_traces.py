"""
Loads the traces in ./data/bayesian_optimization/traces
and plots the evolution, showing the latent space and
acquisition functions.
"""

from pathlib import Path
from typing import Tuple

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def load_traces(
    exp_path: Path, model_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the trace, returning (latent codes, playabilities, jumps).
    """
    initial_trace_path = ROOT_DIR / "data" / "bayesian_optimization" / "initial_traces"
    if "restricted" in exp_path.stem:
        initial_trace = np.load(
            initial_trace_path / f"playability_and_jumps_from_graph_{model_id}.npz"
        )
    elif "vanilla" in exp_path.stem:
        initial_trace = np.load(
            initial_trace_path / f"playabilities_and_jumps_{model_id}.npz"
        )
    else:
        raise ValueError(...)

    # TODO: add the loading of the initial traces.
    arr = np.load(exp_path)

    latent_codes = np.concatenate((initial_trace["zs"], arr["zs"]))
    playability = np.concatenate(
        (initial_trace["playability"].reshape(-1, 1), arr["playability"])
    )
    jumps = np.concatenate((initial_trace["jumps"].reshape(-1, 1), arr["jumps"]))

    return (
        latent_codes,
        playability,
        jumps,
    )


def plot_latent_space(
    latent_codes: np.ndarray,
    jumps: np.ndarray,
    playability: np.ndarray,
    max_jumps: int = None,
):
    _, ax = plt.subplots(1, 1)
    ax.scatter(
        latent_codes[:, 0], latent_codes[:, 1], c=playability, vmin=0.0, vmax=1.0
    )
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.axis("off")


def fit_model(
    latent_codes: t.Tensor, playabilities: t.Tensor, jumps: t.Tensor
) -> ExactGP:
    jumps[playabilities == 0.0] = 0.0

    model = SingleTaskGP(latent_codes, jumps / 10.0)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    return model


def visualize_traces(exp_path: Path, model_id: int = 0):
    """
    Plots the traces one by one.
    """
    latent_codes, playabilities, jumps = load_traces(
        exp_path=exp_path, model_id=model_id
    )
    for i in range(50, len(latent_codes)):
        # Get the relevant slice
        curr_latent_codes = t.from_numpy(latent_codes[:i])
        curr_playabilities = t.from_numpy(playabilities[:i])
        curr_jumps = t.from_numpy(jumps[:i])

        # Fit for the slice
        model = fit_model(curr_latent_codes, curr_playabilities, curr_jumps)

        # Predict on domain
        zs = t.Tensor(
            [
                [x, y]
                for x in t.linspace(-5, 5, 100)
                for y in t.linspace(-5, 5, 100).__reversed__()
            ]
        )
        dist_ = model(zs)
        jump_predictions = dist_.mean
        acq_values = UpperConfidenceBound(model, beta=3.0)(zs.unsqueeze(1))

        # Plot the predictions as a grid.
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot = ax.imshow(
            jump_predictions.cpu().detach().numpy().reshape(100, 100),
            extent=[-5, 5, -5, 5],
        )
        ax.scatter(
            [latent_codes[i, 0]], [latent_codes[i, 1]], marker="d", c="red", s=20
        )
        plt.colorbar(plot, ax=ax)
        plot_acq = ax_acq.imshow(
            acq_values.cpu().detach().numpy().reshape(100, 100), extent=[-5, 5, -5, 5]
        )
        ax_acq.scatter(
            [latent_codes[i, 0]], [latent_codes[i, 1]], marker="d", c="red", s=20
        )
        plt.colorbar(plot_acq, ax=ax_acq)
        fig.savefig(f"{exp_path.stem}_{i:03d}.png")
        plt.close(fig)
        ...


if __name__ == "__main__":
    model_id = 1
    iteration = 0
    visualize_traces(
        ROOT_DIR
        / "data"
        / "bayesian_optimization"
        / "traces"
        / f"vanilla_bo_{model_id}_{iteration}.npz",
        model_id=model_id,
    )
