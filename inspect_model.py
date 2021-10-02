import click
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_base import VAEGeometryBase
from vae_geometry_dirichlet import VAEGeometryDirichlet
from vae_geometry_hierarchical import VAEGeometryHierarchical
from vae_geometry_uniform import VAEGeometryUniform

from diffusions.geometric_difussion import GeometricDifussion
from metric_approximation_with_jacobians import plot_approximation


def load_model(
    extrapolation: str, model_name: str, beta: float = None, only_playable: bool = False
) -> VAEGeometryBase:
    if extrapolation == "dirichlet":
        if beta is None:
            beta = -3.5
        Model = VAEGeometryDirichlet
        update_hyperparams = {
            "beta": beta,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "hierarchical":
        if beta is None:
            beta = -3.0
        Model = VAEGeometryHierarchical
        update_hyperparams = {
            "beta": beta,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "uniform":
        if beta is None:
            beta = -3.5
        Model = VAEGeometryUniform
        update_hyperparams = {
            "beta": beta,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    else:
        raise ValueError(
            f"Unexpected extrapolation {extrapolation}, expected 'dirichlet', 'hierarchical' or 'uniform'"
        )

    vae = Model()
    vae.load_state_dict(t.load(f"./models/{model_name}.pt", map_location="cpu"))
    vae.update_cluster_centers(**update_hyperparams)

    return vae


def plot_circle(vae, ax):
    angles = t.rand((100,)) * 2 * np.pi
    encodings = 3.0 * t.vstack((t.cos(angles), t.sin(angles))).T
    vae.update_cluster_centers(encodings=encodings)
    try:
        vae.plot_w_geodesics(ax=ax, plot_points=False)
    except Exception as e:
        print(f"couldn't get geodesics for reason {e}")


def plot_geometric_diffusion(vae, ax):
    """
    Runs the geometric diffusion 5 times and plots the results.
    """
    geometric_diffusion = GeometricDifussion(50)
    vae.plot_latent_space(ax=ax, plot_points=False)
    for _ in range(5):
        try:
            zs = geometric_diffusion.run(vae).detach().numpy()
            ax.scatter(zs[:, 0], zs[:, 1], c="g", marker="x")
            ax.scatter(zs[:1, 0], zs[:1, 1], c="c", marker="o", zorder=10)
        except Exception as e:
            print(f"Couldn't do diffusion: {e}")

    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))


@click.command()
@click.argument("model_name", type=str)
@click.option("--extrapolation", type=str, default="dirichlet")
@click.option("--beta", type=float, default=None)
@click.option("--only-playable/--not-only-playable", default=False)
def inspect(model_name, extrapolation, beta, only_playable):
    """
    Plots the latent space, a grid, and
    a summary of a given model.
    """
    vae = load_model(extrapolation, model_name, beta=beta, only_playable=only_playable)
    fig, axes = plt.subplots(2, 2, figsize=(2 * 7, 2 * 7))
    axes = axes.flatten()

    axes[0].set_title("Latent space & geodesics")
    vae.plot_w_geodesics(ax=axes[0])

    axes[1].set_title("Geometric diffusion")
    plot_geometric_diffusion(vae, axes[1])

    axes[2].set_title("Approximation of the metric")
    plot_approximation(vae, ax=axes[2])

    axes[3].set_title("On a circle")
    plot_circle(vae, ax=axes[3])

    beta_in_vae = vae.translated_sigmoid.beta.item()
    fig.suptitle(f"Model: {model_name} - {extrapolation} - beta {beta_in_vae}")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/model_inspections/{model_name}_extrapolation_{extrapolation}_playable_{only_playable}.png"
    )
    plt.show()


if __name__ == "__main__":
    inspect()
