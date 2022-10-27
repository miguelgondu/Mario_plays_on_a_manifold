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
import numpy as np

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.simulator.interface import (
    test_level_from_int_tensor,
)
from utils.visualization.latent_space import plot_prediction
from utils.experiment import load_model
from utils.experiment.bayesian_optimization import run_first_samples

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)


def bayesian_optimization_iteration(
    latent_codes: t.Tensor, jumps: t.Tensor, plot_latent_space: bool = False
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    vae = load_model()

    model = SingleTaskGP(latent_codes, jumps / 10.0)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    # EI = UpperConfidenceBound(model, beta=3.0)
    EI = ExpectedImprovement(model, jumps.max() / 10.0)

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
    results = test_level_from_int_tensor(level[0], visualize=False)

    if plot_latent_space:
        fig, ax = plt.subplots(1, 1)
        plot_prediction(model, ax)

        ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="black", marker="x")
        ax.scatter(candidate[:, 0], candidate[:, 1], c="red", marker="d")

        plt.show()
        plt.close(fig)

    return (
        candidate,
        t.Tensor([[results["marioStatus"]]]),
        t.Tensor([[results["jumpActionsPerformed"]]]),
    )


def run_experiment():
    # Hyperparameters
    n_iterations = 50

    # Loading the VAE
    vae = load_model()

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples(vae)
    jumps = jumps.type(t.float32).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    try:
        for i in range(n_iterations):
            candidate, playability, jump = bayesian_optimization_iteration(
                latent_codes, jumps, plot_latent_space=False
            )
            print(
                f"(Iteration {i+1}) tested {candidate} and got {jump} (p={playability})"
            )

            if playability == 0.0:
                jump = t.zeros_like(jump)

            latent_codes = t.vstack((latent_codes, candidate))
            jumps = t.vstack((jumps, jump))
            playabilities = t.vstack((playabilities, playability))
    except Exception as e:
        print(f"Couldn't continue. Stopped at iteration {i+1}")
        print(e)

    # Saving the trace
    np.savez(
        "./data/bayesian_optimization/traces/vanilla_bo_EI.npz",
        zs=latent_codes.detach().numpy(),
        playability=playabilities.detach().numpy(),
        jumps=jumps.detach().numpy(),
    )


if __name__ == "__main__":
    run_experiment()
