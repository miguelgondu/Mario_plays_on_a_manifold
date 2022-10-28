"""
Implements Bayesian Optimization with unknown constraints [ref],
and runs it in the latent space of SMB, optimizing the number of
jumps.

TODO: other experiments we could perform are optimizing the number
of gaps. In this one, the optimization should be pretty skewed
towards unplayable levels, while the constrained/graph approach
would work greatly!

[ref]: ...
"""
from typing import Tuple
from matplotlib import pyplot as plt

import torch as t
import numpy as np

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf


from utils.simulator.interface import (
    test_level_from_int_tensor,
)
from utils.acquisition.constrained_expected_improvement import (
    ConstrainedExpectedImprovement,
)
from utils.visualization.latent_space import plot_prediction

from utils.experiment import load_model
from utils.experiment.bayesian_optimization import run_first_samples
from utils.gp_models.bernoulli_gp import GPClassificationModel


def bayesian_optimization_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    playabilities: t.Tensor,
    plot_latent_space: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    We use constrained Expected Improvement in this example.
    """
    vae = load_model()

    model = SingleTaskGP(latent_codes, jumps)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    success_model = GPClassificationModel(latent_codes)
    optimizer = t.optim.Adam(success_model.parameters(), lr=0.1)
    mll_success = gpytorch.mlls.VariationalELBO(
        success_model.likelihood, success_model, playabilities.numel()
    )

    # Training the success model
    for i in range(100):
        optimizer.zero_grad()
        output = success_model(latent_codes)
        loss = -mll_success(output, playabilities)
        loss.backward()
        # print("Iter %d/%d - Loss: %.3f" % (i + 1, 50, loss.item()))
        optimizer.step()

    cEI = ConstrainedExpectedImprovement(model, success_model, max(jumps))

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

    return (
        candidate,
        t.Tensor([[results["jumpActionsPerformed"]]]),
        t.Tensor([[results["marioStatus"]]]),
    )


def run_experiment(exp_id: int = 0):
    # Hyperparameters
    n_iterations = 50

    # Loading the model
    vae = load_model()

    # Get some first samples and save them.
    latent_codes, playability, jumps = run_first_samples(vae)
    jumps = jumps.type(t.float32).unsqueeze(1)
    playability = playability.unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    try:
        for i in range(n_iterations):
            # if (iteration + 1) % 5 == 0:
            #     plot_latent_space = True
            # else:
            #     plot_latent_space = False

            candidate, jump, p = bayesian_optimization_iteration(
                latent_codes, jumps, playability.squeeze(1), plot_latent_space=False
            )
            print(
                f"(Iteration {i+1}) tested {candidate} and got {jump.item()} (p={p.item()})"
            )

            if p == 0.0:
                jump = t.zeros_like(jump)

            latent_codes = t.vstack((latent_codes, candidate))
            jumps = t.vstack((jumps, jump))
            playability = t.vstack((playability, p))
    except Exception as e:
        print(f"Couldn't continue. Stopped at iteration {i+1}")
        print(e)

    # Saving the trace
    np.savez(
        f"./data/bayesian_optimization/traces/constrained_bo_{exp_id}.npz",
        zs=latent_codes.detach().numpy(),
        playability=playability.detach().numpy(),
        jumps=jumps.detach().numpy(),
    )


if __name__ == "__main__":
    for exp_id in range(10):
        run_experiment(exp_id)
