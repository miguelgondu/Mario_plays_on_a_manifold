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
from utils.visualization.latent_space import plot_prediction, plot_acquisition
from utils.experiment import load_model
from utils.experiment.bayesian_optimization import run_first_samples

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)


def bayesian_optimization_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    plot_latent_space: bool = False,
    model_id: int = 0,
    iteration: int = 0,
    img_save_folder: Path = None,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    vae = load_model(model_id=model_id)

    model = SingleTaskGP(latent_codes, jumps / 10.0)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    # acq_function = UpperConfidenceBound(model, beta=3.0)
    acq_function = ExpectedImprovement(model, jumps.max() / 10.0)

    zs = t.Tensor(
        [
            [x, y]
            for x in t.linspace(-5, 5, 100)
            for y in reversed(t.linspace(-5, 5, 100))
        ]
    )
    acq_values = acq_function(zs.unsqueeze(1))
    candidate = zs[acq_values.argmax()]

    # bounds = t.stack([t.Tensor([-5, -5]), t.Tensor([5, 5])])
    # candidate, _ = optimize_acqf(
    #     EI,
    #     bounds=bounds,
    #     q=1,
    #     num_restarts=5,
    #     raw_samples=20,
    # )

    print(candidate, acq_values[acq_values.argmax()])
    level = vae.decode(candidate).probs.argmax(dim=-1)
    print(level)
    results = test_level_from_int_tensor(level[0], visualize=False)

    if plot_latent_space:
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot_prediction(model, ax)
        plot_acquisition(acq_function, ax_acq)
        # ax_acq.imshow(acq_values.cpu().detach().numpy().reshape(100, 100))

        ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="black", marker="x")
        ax.scatter([candidate[0]], [candidate[1]], c="red", marker="d")

        ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="black", marker="x")
        ax.scatter([candidate[0]], [candidate[1]], c="red", marker="d")

        if img_save_folder is not None:
            img_save_folder.mkdir(exist_ok=True, parents=True)
            fig.savefig(img_save_folder / f"{iteration:04d}.png")
        # plt.show()
        plt.close(fig)

    return (
        candidate,
        t.Tensor([[results["marioStatus"]]]),
        t.Tensor([[results["jumpActionsPerformed"]]]),
    )


def run_experiment(exp_id: int = 0, model_id: int = 0):
    # Hyperparameters
    n_iterations = 50

    # Loading the VAE
    vae = load_model(model_id=model_id)

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples(
        vae, model_id=model_id, n_samples=10, force=False
    )
    jumps = jumps.type(t.float32).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    # try:
    img_save_folder = (
        ROOT_DIR
        / "data"
        / "plots"
        / "bayesian_optimization"
        / f"vanilla_bo_{model_id}_{exp_id}"
    )
    for i in range(n_iterations):
        candidate, playability, jump = bayesian_optimization_iteration(
            latent_codes,
            jumps,
            plot_latent_space=True,
            iteration=i,
            img_save_folder=img_save_folder,
        )
        print(f"(Iteration {i+1}) tested {candidate} and got {jump} (p={playability})")

        if playability == 0.0:
            jump = t.zeros_like(jump)

        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playabilities = t.vstack((playabilities, playability))
    # except Exception as e:
    #     print(f"Couldn't continue. Stopped at iteration {i+1}")
    #     print(e)

    # Saving the trace
    np.savez(
        f"./data/bayesian_optimization/traces/vanilla_bo_{model_id}_{exp_id}.npz",
        zs=latent_codes.detach().numpy(),
        playability=playabilities.detach().numpy(),
        jumps=jumps.detach().numpy(),
    )


if __name__ == "__main__":
    model_id = 2

    for exp_id in range(10):
        run_experiment(exp_id=exp_id, model_id=model_id)
