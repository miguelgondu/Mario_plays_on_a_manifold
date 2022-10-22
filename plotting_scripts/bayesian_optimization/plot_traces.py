"""
Loads the traces in ./data/bayesian_optimization/traces
and plots the evolution, showing the latent space and
acquisition functions.
"""

from typing import Tuple

import numpy as np


def load_traces() -> Tuple[np.ndarray, np.ndarray]:
    ...


def plot_latent_space(
    latent_codes: np.ndarray, jumps: np.ndarray, playability: np.ndarray
):
    ...


def visualize_traces():
    """
    Plots the traces into videos.
    """
    ...
