"""
A suite of tools for visualizing the graph
discretization in latent space.
"""
import matplotlib.pyplot as plt

from utils.gp_models.graph_gp import GraphBasedGP
from geometries.discretized_geometry import DiscretizedGeometry


def visualize_graph_gp_in_ax(
    ax: plt.Axes,
    graph_gp: GraphBasedGP,
    discretized_geometry: DiscretizedGeometry,
    **scatter_kwargs
):
    """
    Plots the Gaussian Process posterior predictions associated with a
    certain discretized geometry in the axis.

    It modifies the graph GP, putting it in eval mode.
    """
    # The indices of the graph
    if graph_gp.training:
        graph_gp.eval()

    graph_indices = graph_gp.train_inputs[0]
    posterior = graph_gp(graph_indices)
    predictions = posterior.mean

    # Transforms them into latent codes.
    latent_positions = discretized_geometry.from_graph_idx_to_latent_code(graph_indices)

    ax.scatter(
        latent_positions[:, 0].detach().numpy(),
        latent_positions[:, 1].detach().numpy(),
        c=predictions.detach().numpy(),
        **scatter_kwargs
    )
