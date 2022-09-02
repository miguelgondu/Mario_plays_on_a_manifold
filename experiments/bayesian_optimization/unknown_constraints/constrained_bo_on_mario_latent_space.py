"""
Implements Bayesian Optimization with unknown constraints [ref],
and runs it in the latent space of SMB.

[ref]: ...
"""
from typing import Tuple
from matplotlib import pyplot as plt

import torch as t

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
from utils.gp_models.bernoulli_gp import GPClassificationModel

from experiments.bayesian_optimization.unknown_constraints.vanilla_bo_on_mario_latent_space import (
    run_first_samples,
)


def bayesian_optimization_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    playabilities: t.Tensor,
    plot_latent_space: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    vae = load_model()

    model = SingleTaskGP(latent_codes, jumps)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # success_model = DirichletGPModel(latent_codes, playabilities)
    success_model = GPClassificationModel(latent_codes)
    optimizer = t.optim.Adam(success_model.parameters(), lr=0.1)
    mll_success = gpytorch.mlls.VariationalELBO(
        success_model.likelihood, success_model, playabilities.numel()
    )

    for i in range(50):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = success_model(latent_codes)
        # Calc loss and backprop gradients
        loss = -mll_success(output, playabilities)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, 50, loss.item()))
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
        t.Tensor([results["marioStatus"]]),
    )


def run_experiment():
    vae = load_model()

    # Get some first samples and save them.
    latent_codes, playability, jumps = run_first_samples(vae)
    jumps = jumps.type(t.float32).unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    for iteration in range(20):
        if (iteration + 1) % 5 == 0:
            plot_latent_space = True
        else:
            plot_latent_space = False

        candidate, jump, p = bayesian_optimization_iteration(
            latent_codes, jumps, playability, plot_latent_space=plot_latent_space
        )
        print(f"tested {candidate} and got {jump}")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playability = t.hstack((playability, p))


if __name__ == "__main__":
    run_experiment()
