"""
Implements vanilla Bayesian Optimization (without constraints)
in the latent space of our SMB VAEs. For now, I will
try to optimize for the number of jumps coming out of the
simulator while constraining that the level should be playable.
"""
from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import torch as t

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.simulator.interface import (
    test_level_from_int_tensor,
)
from utils.visualization.latent_space import plot_prediction
from utils.experiment import load_model
from utils.experiment.bayesian_optimization import run_first_samples

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


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

    EI = ExpectedImprovement(model, max(jumps))

    bounds = t.stack([t.Tensor([-5, -5]), t.Tensor([5, 5])])
    candidate, _ = optimize_acqf(
        EI,
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


def run_experiment():
    vae = load_model()

    # Get some first samples and save them.
    latent_codes, _, jumps = run_first_samples(vae)
    jumps = jumps.type(t.float32).unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    for iteration in range(20):
        if (iteration + 1) % 5 == 0:
            plot_latent_space = True
        else:
            plot_latent_space = False

        candidate, jump = bayesian_optimization_iteration(
            latent_codes, jumps, plot_latent_space=plot_latent_space
        )
        print(f"tested {candidate} and got {jump}")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))


if __name__ == "__main__":
    run_experiment()
