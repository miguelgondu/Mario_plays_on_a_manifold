import click
import torch as t
import matplotlib.pyplot as plt

from vae_geometry_base import VAEGeometryBase
from vae_geometry_dirichlet import VAEGeometryDirichlet
from vae_geometry_hierarchical import VAEGeometryHierarchical
from vae_geometry_uniform import VAEGeometryUniform

from diffusions.geometric_difussion import GeometricDifussion
from diffusions.normal_diffusion import NormalDifussion


def load_model(
    extrapolation: str, model_name: str, only_playable: bool = False
) -> VAEGeometryBase:
    if extrapolation == "dirichlet":
        Model = VAEGeometryDirichlet
        update_hyperparams = {
            "beta": -1.5,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "hierarchical":
        Model = VAEGeometryHierarchical
        update_hyperparams = {
            "beta": -1.5,
            "n_clusters": 500,
            "only_playable": only_playable,
        }
    elif extrapolation == "Uniform":
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
@click.argument("model-name", type=str, default=None)
@click.option("--extrapolation", type=str, default="dirichlet")
@click.option("--only-playable/--not-only-playable", default=False)
@click.option("--n-points", type=int, default=100)
@click.option("--n-runs", type=int, default=10)
def run(model_name, extrapolation, only_playable, n_points, n_runs):
    vae = load_model(extrapolation, model_name, only_playable=only_playable)
    geometric_diffusion = GeometricDifussion(n_points)
    normal_diffusion = NormalDifussion(n_points)

    _, ax = plt.subplots(1, 1)
    vae.plot_latent_space(ax=ax, plot_points=False)
    for _ in range(n_runs):
        zs = normal_diffusion.run(vae).detach().numpy()
        ax.scatter(zs[:, 0], zs[:, 1], c="r", marker="x")

    plt.show()


if __name__ == "__main__":
    run()
