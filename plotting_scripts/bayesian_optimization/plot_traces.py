"""
Loads the traces in ./data/bayesian_optimization/traces
and plots the evolution, showing the latent space and
acquisition functions.
"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_traces() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def plot_latent_space(
    latent_codes: np.ndarray,
    jumps: np.ndarray,
    playability: np.ndarray,
    max_jumps: int = None,
):
    _, ax = plt.subplots(1, 1)
    ax.scatter(
        latent_codes[:, 0], latent_codes[:, 1], c=playability, vmin=0.0, vmax=1.0
    )
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.axis("off")


def visualize_traces():
    """
    Plots the traces into videos.
    """
    latent_codes, playability, jumps = load_traces()
    for i in range(1, len(latent_codes)):
        ...
