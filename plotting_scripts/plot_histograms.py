"""
Loads an experiment and plots a histogram
of the counts of a given column in the .csv
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_scripts.utils import load_experiment_csv_paths


def get_all_values(files: List[Path], column: str):
    """
    Loads and concatenates all values of a certain column
    """
    values = []
    for p in files:
        df = pd.read_csv(p, index_col=0)
        vals = df[column].values
        values.append(vals)

    values = np.concatenate(values)

    return values


def analyse(exp_name: str, column: str):
    """
    Saves two histograms, one for interpolations and
    one for diffusions of the given experiment
    """
    interps, diffs = load_experiment_csv_paths(exp_name)
    v_interps = get_all_values(interps, column)
    v_diffs = get_all_values(diffs, column)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 7, 7))

    fig.suptitle(f"{exp_name} - {column}")
    ax1.hist(v_interps[v_interps < 10], bins=10)
    ax1.set_title("Interpolations")
    ax2.hist(v_diffs[v_diffs < 10], bins=10)
    ax2.set_title("Random Walks")

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    analyse("baseline_no_jump_gt", "jumpActionsPerformed")
    analyse("discrete_no_jump_gt", "jumpActionsPerformed")
    analyse("continuous_no_jump_gt", "jumpActionsPerformed")

    analyse("baseline_no_jump_gt", "marioStatus")
    analyse("discrete_no_jump_gt", "marioStatus")
    analyse("continuous_no_jump_gt", "marioStatus")
