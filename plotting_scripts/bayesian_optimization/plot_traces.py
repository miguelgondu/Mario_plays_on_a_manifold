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
from botorch.acquisition import ExpectedImprovement


def load_traces(exp_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the trace, returning (latent codes, playabilities, jumps).
    """
    arr = np.load(exp_path)
    return (
        arr["latent_codes"],
        arr["playabilities"],
        arr["jumps"],
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

    model.eval()



def visualize_traces(exp_path: Path):
    """
    Plots the traces one by one.
    """
    latent_codes, playabilities, jumps = load_traces(exp_path=exp_path)
    for i in range(1, len(latent_codes)):
        # Get the relevant slice
        curr_latent_codes = latent_codes[:i]
        curr_playabilities = playabilities[:i]
        curr_jumps = jumps[:i]

        # Fit for the slice
        model = fit_model(curr_latent_codes, curr_playabilities, curr_jumps)

        # Predict on domain
        zs = t.Tensor(
            [[x, y] for x in t.linspace(-5, 5, 100) for y in t.linspace(-5, 5, 100)]
        )
        dist_ = model(zs)
        acq_values = ExpectedImprovement(model, jumps.max())(zs.unsqueeze(1))

        # Plot the predictions as a grid.
        ...
