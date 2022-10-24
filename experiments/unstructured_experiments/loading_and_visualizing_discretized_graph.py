"""
To check whether all the indexing is working, this script
loads and checks the discretized geometry's inner graph,
making sure we are properly building the graph.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.experiment import load_model
from utils.experiment.bayesian_optimization import load_geometry


def main():
    vae = load_model()
    dg = load_geometry()

    graph = dg.to_graph()
    print(type(graph))

    latent_codes = np.array([dg.from_graph_node_to_latent_code(n) for n in graph.nodes])
    _, (ax, ax2) = plt.subplots(1, 2)
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1])
    ax2.scatter(dg.zs[:, 0], dg.zs[:, 1])
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
