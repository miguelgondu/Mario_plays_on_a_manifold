"""
Implements Bayesian Optimization on a restricted domain
of the latent space of our SMB VAEs.
"""
from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import torch as t
import numpy as np

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel

from utils.simulator.interface import (
    test_level_from_int_tensor,
)
from utils.visualization.latent_space import plot_prediction, plot_acquisition
from utils.experiment import load_model
from utils.experiment.bayesian_optimization import (
    run_first_samples,
    load_geometry,
    run_first_samples_from_graph,
)

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-3)
if t.cuda.is_available():
    t.set_default_tensor_type(t.cuda.FloatTensor)


def fitness_function(jumps: t.Tensor):
    return jumps / 10.0


def bayesian_optimization_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    plot_latent_space: bool = False,
    model_id: int = 0,
    iteration: int = 0,
    img_save_folder: Path = None,
    mean_scale: float = 1.3,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    vae = load_model(model_id=model_id)
    discrete_geometry = load_geometry(
        mean_scale=mean_scale,
        model_id=model_id,
        name=f"bo_for_model_{model_id}_scale_{int(mean_scale*100)}",
    )
    restricted_domain = discrete_geometry.restricted_domain.to(vae.device)

    # kernel = ScaleKernel(MaternKernel())
    kernel = None
    model = SingleTaskGP(latent_codes, fitness_function(jumps), covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    acq_function = ExpectedImprovement(model, fitness_function(jumps).max())
    # acq_function = UpperConfidenceBound(model, beta=3.0)
    acq_on_restricted_domain = acq_function(restricted_domain.unsqueeze(1))
    candidate = restricted_domain[acq_on_restricted_domain.argmax()]

    print(candidate, acq_on_restricted_domain.max())
    level = vae.decode(candidate).probs.argmax(dim=-1)
    print(level)
    results = test_level_from_int_tensor(level[0], visualize=False)

    if plot_latent_space:
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot_prediction(model, ax)
        plot_acquisition(acq_function, ax_acq)
        # ax_acq.imshow(acq_values.cpu().detach().numpy().reshape(100, 100))

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter([candidate[0].cpu()], [candidate[1].cpu()], c="red", marker="d")

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter([candidate[0].cpu()], [candidate[1].cpu()], c="red", marker="d")

        if img_save_folder is not None:
            img_save_folder.mkdir(exist_ok=True, parents=True)
            fig.tight_layout()
            fig.savefig(img_save_folder / f"{iteration:04d}.png")
        # plt.show()
        plt.close(fig)

    return (
        candidate.to(vae.device),
        t.Tensor([[results["marioStatus"]]]).to(vae.device),
        t.Tensor([[results["jumpActionsPerformed"]]]).to(vae.device),
        level.to(vae.device),
    )


def run_experiment(exp_id: int = 0, model_id: int = 0, mean_scale: float = 1.0):
    # Hyperparameters
    n_iterations = 50

    # Loading the VAE
    vae = load_model(model_id=model_id)
    # dg = load_geometry(model_id=model_id)

    # Get some first samples and save them.
    # latent_codes, playabilities, jumps = run_first_samples_from_graph(
    #     vae, dg, model_id=model_id, n_samples=10, force=False
    # )
    latent_codes, playabilities, jumps = run_first_samples(
        vae, model_id=model_id, n_samples=2, force=True
    )
    jumps = jumps.type(t.float32).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)

    jumps[playabilities == 0.0] = 0.0

    latent_codes = latent_codes.to(vae.device)
    jumps = jumps.to(vae.device)
    playabilities = playabilities.to(vae.device)

    # Initialize the GPR model for the predicted number
    # of jumps.
    levels = []
    img_save_folder = (
        ROOT_DIR
        / "data"
        / "plots"
        / "bayesian_optimization"
        / f"restricted_bo_{model_id}_{exp_id}"
    )
    try:
        for i in range(n_iterations):
            candidate, playability, jump, level = bayesian_optimization_iteration(
                latent_codes,
                jumps,
                plot_latent_space=True,
                model_id=model_id,
                iteration=i,
                img_save_folder=img_save_folder,
                mean_scale=mean_scale,
            )
            print(
                f"(Iteration {i+1}) tested {candidate} and got {jump} (p={playability})"
            )
            latent_codes = t.vstack((latent_codes, candidate))

            if playability == 0.0:
                jump = t.zeros_like(jump)

            jumps = t.vstack((jumps, jump))
            playabilities = t.vstack((playabilities, playability))
            levels.append(level)
    except Exception as e:
        print(f"Couldn't continue. Stopped at iteration {i+1}")
        print(e)
        raise e

    # Saving the trace
    np.savez(
        f"./data/bayesian_optimization/traces/restricted_bo_{int(mean_scale*100)}_{model_id}_{exp_id}.npz",
        zs=latent_codes.cpu().detach().numpy(),
        playability=playabilities.cpu().detach().numpy(),
        jumps=jumps.cpu().detach().numpy(),
        levels=t.cat([lvl.unsqueeze(0) for lvl in levels]).cpu().detach().numpy(),
    )


if __name__ == "__main__":
    model_id = 1
    mean_scale = 1.3

    for exp_id in range(20):
        run_experiment(exp_id=exp_id, model_id=model_id, mean_scale=mean_scale)
