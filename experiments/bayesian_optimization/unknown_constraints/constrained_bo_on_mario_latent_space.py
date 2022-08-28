"""
Implements Bayesian Optimization with unknown constraints [ref],
and runs it in the latent space of SMB.

[ref]: ...
"""
from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import torch as t
import numpy as np

import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.simulator.interface import (
    test_level_from_decoded_tensor,
    test_level_from_int_tensor,
)
from utils.acquisition.constrained_expected_improvement import (
    ConstrainedExpectedImprovement,
)
from utils.visualization.latent_space import plot_prediction

from utils.experiment import load_model


def bayesian_optimization_iteration(
    latent_codes: t.Tensor, jumps: t.Tensor, plot_latent_space: bool = False
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    vae = load_model()

    model = SingleTaskGP(latent_codes, jumps)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    cEI = ConstrainedExpectedImprovement(model, max(jumps))

    bounds = t.stack([t.Tensor([-5, -5]), t.Tensor([5, 5])])
    candidate, _ = optimize_acqf(
        cEI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    print(candidate)
    level = vae.decode(candidate).probs.argmax(dim=-1)
    print(level)
    results = test_level_from_int_tensor(level[0], visualize=True)

    if plot_latent_space:
        fig, ax = plt.subplots(1, 1)
        plot_prediction(model, ax)

        ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="black", marker="x")
        ax.scatter(candidate[:, 0], candidate[:, 1], c="red", marker="d")

        plt.show()
        plt.close(fig)

    return candidate, t.Tensor([[results["jumpActionsPerformed"]]])
