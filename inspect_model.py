import click
import torch as t
import matplotlib.pyplot as plt

from vae_geometry_base import VAEGeometryBase
from vae_geometry_dirichlet import VAEGeometryDirichlet
from vae_geometry_hierarchical import VAEGeometryHierarchical
from vae_geometry_uniform import VAEGeometryUniform

from metric_approximation_with_jacobians import plot_approximation


def load_model(
    extrapolation: str, model_name: str, only_playable: bool = False
) -> VAEGeometryBase:
    if extrapolation == "dirichlet":
        Model = VAEGeometryDirichlet
        update_hyperparams = {
            "beta": -2.2,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "hierarchical":
        Model = VAEGeometryHierarchical
        update_hyperparams = {
            "beta": -2.5,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "uniform":
        Model = VAEGeometryUniform
        update_hyperparams = {
            "beta": -1.5,
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


@click.command()
@click.argument("model_name", type=str)
@click.option("--extrapolation", type=str, default="dirichlet")
@click.option("--only-playable/--not-only-playable", default=False)
def inspect(model_name, extrapolation, only_playable):
    """
    Plots the latent space, a grid, and
    a summary of a given model.
    """
    vae = load_model(extrapolation, model_name, only_playable=only_playable)
    _, axes = plt.subplots(1, 3, figsize=(3 * 7, 7))
    axes = axes.flatten()

    axes[0].set_title("Latent space & geodesics")
    vae.plot_w_geodesics(ax=axes[0])

    axes[1].set_title("Grid of levels")
    vae.plot_grid(ax=axes[1])

    axes[2].set_title("Approximation of the metric")
    plot_approximation(vae, ax=axes[2])
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"./data/plots/model_inspections/{model_name}_extrapolation_{extrapolation}_playable_{only_playable}.png"
    )
    plt.show()


if __name__ == "__main__":
    inspect()
