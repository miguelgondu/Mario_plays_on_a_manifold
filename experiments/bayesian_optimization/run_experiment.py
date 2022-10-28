import torch as t

from utils.experiment import load_model
from utils.experiment.bayesian_optimization import (
    run_first_samples,
    load_geometry,
    run_first_samples_from_graph,
)


def run_experiment(exp_id: int = 0):
    # Hyperparameters
    n_iterations = 50

    # Loading the VAE
    vae = load_model()
    dg = load_geometry()

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples_from_graph(vae, dg)
    jumps = jumps.type(t.float32).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)

    latent_codes = latent_codes.to(vae.device)
    jumps = jumps.to(vae.device)
    playabilities = playabilities.to(vae.device)

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
            latent_codes = t.vstack((latent_codes, candidate))

            if playability == 0.0:
                jump = t.zeros_like(jump)

            jumps = t.vstack((jumps, jump))
            playabilities = t.vstack((playabilities, playability))
    except Exception as e:
        print(f"Couldn't continue. Stopped at iteration {i+1}")
        print(e)
        raise e

    # Saving the trace
    np.savez(
        f"./data/bayesian_optimization/traces/restricted_bo_{exp_id}.npz",
        zs=latent_codes.cpu().detach().numpy(),
        playability=playabilities.cpu().detach().numpy(),
        jumps=jumps.cpu().detach().numpy(),
    )
