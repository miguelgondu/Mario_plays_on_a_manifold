"""
A suite of tools for visualizing the graph
discretization in latent space.
"""
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.gp_models.graph_gp import GraphBasedGP
from geometries.discretized_geometry import DiscretizedGeometry


def visualize_discretized_graph_nodes_and_values_in_ax(
    ax: plt.Axes,
    values: np.ndarray,
    discretized_geometry: DiscretizedGeometry,
    **scatter_kwargs
):
    nodes = np.array(discretized_geometry.to_graph().nodes)
    graph_idxs = (
        np.array(
            [
                discretized_geometry.graph_nodes.index((int(i), int(j)))
                for (i, j) in nodes
            ]
        )
        .astype(int)
        .reshape(-1, 1)
    )
    latent_codes = discretized_geometry.from_graph_idx_to_latent_code(graph_idxs)

    assert len(values) == len(latent_codes)

    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c=values)


def visualize_discretized_graph_gp_in_ax(
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
    posterior = graph_gp(*graph_gp.train_inputs)
    predictions = posterior.mean

    # Transforms them into latent codes.
    latent_positions = discretized_geometry.from_graph_idx_to_latent_code(graph_indices)

    ax.scatter(
        latent_positions[:, 0],
        latent_positions[:, 1],
        c=predictions.detach().numpy(),
        **scatter_kwargs
    )


def visualize_graph_gp_in_ax(
    ax: plt.Axes,
    graph_gp: GraphBasedGP,
    graph: nx.Graph,
    from_graph_idx_to_node: Callable[[int], int] = lambda x: x,
    **scatter_kwargs
):
    """
    A more generic function that plots a graph using networkx's tools
    and colors the nodes according to the graph GP predictions.

    TODO: what if the GP was trained on a subset/different ordering
    of the graph indices? This needs to be accounted for.
    """
    # The indices of the graph
    if graph_gp.training:
        graph_gp.eval()

    graph_indices = graph_gp.train_inputs[0]
    posterior = graph_gp(graph_indices)
    predictions = posterior.mean

    # These graph indices might not be the entire graph.
    colors = np.zeros(len(graph.nodes))
    colors[graph_indices.detach().numpy().astype(int)] = predictions.detach().numpy()
    print(graph_gp.train_targets)
    print(predictions)

    # This assumes that the nodes are ordered in graph_indices.
    # Is that the case?
    nx.draw(graph, ax=ax, node_color=colors, with_labels=True)
