"""
Loads an experiment and plots a histogram
of the counts of a given column in the .csv
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.other_utils import load_experiment_csv_paths


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


def bar_chart_comparison(column, experiments, filter_, title):
    fig, (ax_interps, ax_diffs) = plt.subplots(1, 2, figsize=(7 * 2, 7))
    table_interps = []
    table_diffs = []
    for exp_name in experiments:
        interps, diffs = load_experiment_csv_paths(exp_name)
        v_interps = get_all_values(interps, column)
        n_values_interps = len(v_interps)
        v_interps = list(filter(filter_, v_interps))
        table_interps.append({"experiment": exp_name, "amount": len(v_interps)})

        v_diffs = get_all_values(diffs, column)
        n_values_diffs = len(v_diffs)
        v_diffs = list(filter(filter_, v_diffs))
        table_diffs.append({"experiment": exp_name, "amount": len(v_diffs)})

    table_interps = pd.DataFrame(table_interps)
    table_diffs = pd.DataFrame(table_diffs)

    ax_interps = sns.barplot(
        x="experiment", y="amount", data=table_interps, ax=ax_interps
    )
    ax_diffs = sns.barplot(x="experiment", y="amount", data=table_diffs, ax=ax_diffs)

    # ax_interps.bar_label(
    #     [f"{c/n_values_interps:.2f}" for c in ax_interps.containers[0]]
    # )
    # ax_diffs.bar_label([f"{c/n_values_diffs:.2f}" for c in ax_diffs.containers[0]])

    ax_interps.set_title(f"Out of {n_values_interps} simulations.")
    ax_diffs.set_title(f"Out of {n_values_diffs} simulations.")

    fig.suptitle(title)

    plt.show()


def dist_comparison(column, experiments, filter_, title):
    facet_table = []
    # _, ax = plt.subplots(1, 1, figsize=(7, 7))
    for exp_name in experiments:
        interps, diffs = load_experiment_csv_paths(exp_name)
        v_interps = get_all_values(interps, column)
        v_interps = list(filter(filter_, v_interps))
        for v in v_interps:
            facet_table.append({"experiment": exp_name, column: v})

    g = sns.FacetGrid(pd.DataFrame(facet_table), row="experiment")
    g.map(sns.histplot, column)
    plt.show()


if __name__ == "__main__":
    # analyse("baseline_no_jump_gt", "jumpActionsPerformed")
    # analyse("discrete_no_jump_gt", "jumpActionsPerformed")
    # analyse("continuous_no_jump_gt", "jumpActionsPerformed")

    # analyse("baseline_no_jump_gt", "marioStatus")
    # analyse("discrete_no_jump_gt", "marioStatus")
    # analyse("continuous_no_jump_gt", "marioStatus")
    # experiments = [
    #     "normal_strict_gt",
    #     "baseline_strict_gt",
    #     "continuous_strict_gt",
    #     "discrete_strict_gt",
    # ]
    # bar_chart_comparison(
    #     "marioStatus",
    #     experiments,
    #     filter_=lambda x: x == 0.0,
    #     title="Non-playable levels",
    # )
    experiments = [
        # "normal_strict_gt",
        "baseline_force_jump_gt",
        # "continuous_force_jump_gt",
        "discrete_force_jump_gt",
    ]
    # bar_chart_comparison(
    #     "jumpActionsPerformed",
    #     experiments,
    #     filter_=lambda x: x == 0.0,
    #     title="Levels without jumps",
    # )
    dist_comparison(
        "jumpActionsPerformed", experiments, lambda x: x < 16, "Dist comparison"
    )
