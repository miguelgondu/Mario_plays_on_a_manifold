"""
Implements Bayesian Optimization with unknown constraints
[...] in the latent space of our SMB VAEs. For now, I will
try to optimize for the number of jumps coming out of the
simulator while constraining that the level should be playable.
"""
from typing import Tuple
from pathlib import Path

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
from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


def load_model() -> VAEMarioHierarchical:
    model_name = "vae_mario_hierarchical_id_0"
    vae = VAEMarioHierarchical()
    vae.load_state_dict(
        t.load(f"./trained_models/ten_vaes/{model_name}.pt", map_location=vae.device)
    )
    vae.eval()

    return vae


# Implement EI.


# Run first samples
def run_first_samples(
    vae: VAEMarioHierarchical, n_samples: int = 10, force: bool = False
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    data_path = (
        ROOT_DIR
        / "data"
        / "bayesian_optimization"
        / "initial_traces"
        / "playability_and_jumps.npz"
    )
    if not force and data_path.exists():
        array = np.load(data_path)
        latent_codes = t.from_numpy(array["zs"])
        playability = t.from_numpy(array["playability"])
        jumps = t.from_numpy(array["jumps"])

        return latent_codes, playability, jumps

    latent_codes = 5.0 * vae.p_z.sample((n_samples,))
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for level in levels:
        results = test_level_from_int_tensor(level, visualize=True)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])

    # Saving the array
    np.savez(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps.npz",
        zs=latent_codes.detach().numpy(),
        playability=np.array(playability),
        jumps=np.array(jumps),
    )

    # Returning.
    return latent_codes, t.Tensor(playability), t.Tensor(jumps)


def bayesian_optimization_iteration(
    latent_codes: t.Tensor, jumps: t.Tensor
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

    return candidate, t.Tensor([[results["jumpActionsPerformed"]]])


def run_experiment():
    vae = load_model()

    # Get some first samples and save them.
    latent_codes, playability, jumps = run_first_samples(vae)
    jumps = jumps.type(t.float32).unsqueeze(1)

    # Initialize the GPR model for the predicted number
    # of jumps.
    for _ in range(20):
        candidate, jump = bayesian_optimization_iteration(latent_codes, jumps)
        print(f"tested {candidate} and got {jump}")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))


if __name__ == "__main__":
    run_experiment()
