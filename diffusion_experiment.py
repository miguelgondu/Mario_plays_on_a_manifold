import click
import numpy as np
import torch as t
import matplotlib.pyplot as plt

from diffusions.geometric_difussion import GeometricDifussion
from diffusions.normal_diffusion import NormalDifussion
from diffusions.baseline_diffusion import BaselineDiffusion

from inspect_model import load_model


@click.command()
@click.argument("model-name", type=str, default=None)
@click.option("--extrapolation", type=str, default="dirichlet")
@click.option("--only-playable/--not-only-playable", default=False)
@click.option("--n-points", type=int, default=100)
@click.option("--n-runs", type=int, default=10)
@click.option("--seed", type=int, default=0)
def run(model_name, extrapolation, only_playable, n_points, n_runs, seed):
    vae = load_model(extrapolation, model_name, only_playable=only_playable)
    g_scales = [0.5, 1.0, 1.5]
    b_step_sizes = [0.5, 1.0, 1.5]
    n_scales = [0.5, 1.0, 1.5]

    np.random.seed(seed)

    # Geometric diffusion experiments
    for g_scale in g_scales:
        geometric_diffusion = GeometricDifussion(n_points, scale=g_scale)
        for r in range(n_runs):
            try:
                zs_g = geometric_diffusion.run(vae).detach().numpy()
                levels_g = vae.decode(t.from_numpy(zs_g)).probs.argmax(dim=-1)
                array_name = f"geodesic_diffusion_model_{model_name}_extrapolation_{extrapolation}_scale_{g_scale}_run_{r}"
                np.savez(
                    f"./data/arrays/{array_name}.npz",
                    zs=zs_g,
                    levels=levels_g,
                )
                print(f"Saved {array_name}.")
            except Exception as e:
                print(f"Couldn't diffuse: {e}")

    # Baseline diffusion experiments
    for b_step_size in b_step_sizes:
        baseline_diffusion = BaselineDiffusion(n_points, step_size=b_step_size)
        for r in range(n_runs):
            zs_b = baseline_diffusion.run(vae).detach().numpy()
            levels_b = vae.decode(t.from_numpy(zs_b)).probs.argmax(dim=-1)

            array_name = f"baseline_diffusion_model_{model_name}_extrapolation_{extrapolation}_stepsize_{b_step_size}_run_{r}"
            np.savez(
                f"./data/arrays/{array_name}.npz",
                zs=zs_b,
                levels=levels_b,
            )

            print(f"Saved {array_name}.")

    # Normal diffusion experiments
    for n_scale in n_scales:
        normal_diffusion = NormalDifussion(n_points, scale=n_scale)
        for r in range(n_runs):
            zs = normal_diffusion.run(vae).detach().numpy()
            levels = vae.decode(t.from_numpy(zs)).probs.argmax(dim=-1)

            array_name = f"normal_diffusion_model_{model_name}_extrapolation_{extrapolation}_scale_{n_scale}_run_{r}"
            np.savez(
                f"./data/arrays/{array_name}.npz",
                zs=zs,
                levels=levels,
            )
            print(f"Saved {array_name}.")


if __name__ == "__main__":
    run()
