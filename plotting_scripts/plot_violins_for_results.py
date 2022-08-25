from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.utils import (
    get_mean,
    get_mean_diversities,
    get_mean_diversities_of_levels,
    get_mean_playability,
)
from plotting_scripts.plot_banner_w_grid import BIGGER_SIZE, MEDIUM_SIZE


def load_experiment(exp_folder: str, exp_name: str, id_: int) -> Tuple[List[Path]]:
    if exp_folder == "ten_vaes":
        end_ = "csv"
    else:
        end_ = "npz"
    interp_res_paths = (
        Path(f"./data/array_simulation_results/{exp_folder}/interpolations") / exp_name
    ).glob(f"*.{end_}")
    diff_res_paths = (
        Path(f"./data/array_simulation_results/{exp_folder}/diffusions") / exp_name
    ).glob(f"*.{end_}")

    filter_ = lambda x: f"final_{id_}" in x.name or f"id_{id_}" in x.name
    return list(filter(filter_, interp_res_paths)), list(
        filter(filter_, diff_res_paths)
    )


def get_all_means(exp: List[Path]) -> Tuple[List[float]]:
    """
    Loads up arrays, computes mean, std playability
    and mean, std diversity.
    """
    all_ps = []
    all_ds = []
    for path in exp:
        if path.name.endswith("npz"):
            array = np.load(path)
            all_ps.append(np.mean(array["ps"]))
            all_ds.append(np.mean(get_mean_diversities_of_levels(array["levels"])))
        elif path.name.endswith("csv"):
            all_ps.append(get_mean_playability([path]))
            all_ds.append(get_mean_diversities([path]))

    return all_ps, all_ds


def parse_exp_name(exp_name: str) -> str:
    if "baseline" in exp_name:
        first = "Baseline"
    elif "discretized" in exp_name:
        first = "Ours"
    elif "normal" in exp_name:
        first = "Normal"
    else:
        raise ValueError(f"Unkown experiment {exp_name}")

    return first


def plot_violins_for_dataviz_course(experiments, vae_ids):
    rows_interps = []
    for exp_folder, exp_name in experiments:
        for id_ in vae_ids:
            interps, _ = load_experiment(exp_folder, exp_name, id_)
            all_ps, all_ds = get_all_means(interps)
            rows_interps.extend(
                [
                    {
                        "id": id_,
                        "playability": m,
                        "experiment": parse_exp_name(exp_name),
                        "diversity": d,
                    }
                    for m, d in zip(all_ps, all_ds)
                ]
            )

    p_interp = pd.DataFrame(rows_interps)
    fig, ax_interp_p = plt.subplots(1, 1, figsize=(7, 4))
    sns.violinplot(
        data=p_interp,
        x="experiment",
        y="playability",
        ax=ax_interp_p,
        cut=1.0,
        palette="Blues",
    )
    ax_interp_p.set_ylabel("Playability (I)", fontsize=14)
    ax_interp_p.set_xlabel("")
    ax_interp_p.tick_params("x", labelsize=BIGGER_SIZE)
    fig.tight_layout()
    fig.savefig(
        "./data/plots/data_viz_course/violins.png", dpi=100, bbox_inches="tight"
    )


def plot_violins(
    experiments,
    vae_ids,
    ax_interp_p,
    ax_diff_p,
    ax_interp_d,
    ax_diff_d,
    points=False,
    palette="Blues",
):
    rows_interps = []
    rows_diffs = []
    for exp_folder, exp_name in experiments:
        for id_ in vae_ids:
            print(f"({exp_folder}) {exp_name} - {id_}")
            interps, diffs = load_experiment(exp_folder, exp_name, id_)
            all_ps, all_ds = get_all_means(interps)
            print("interps")
            print(np.mean(all_ps), np.mean(all_ds))
            rows_interps.extend(
                [
                    {
                        "id": id_,
                        "playability": m,
                        "experiment": parse_exp_name(exp_name),
                        "diversity": d,
                    }
                    for m, d in zip(all_ps, all_ds)
                ]
            )

            all_ps, all_ds = get_all_means(diffs)
            print("diffs")
            print(np.mean(all_ps), np.mean(all_ds))
            rows_diffs.extend(
                [
                    {
                        "id": id_,
                        "playability": m,
                        "experiment": parse_exp_name(exp_name),
                        "diversity": d,
                    }
                    for m, d in zip(all_ps, all_ds)
                ]
            )

    p_interp = pd.DataFrame(rows_interps)
    p_diff = pd.DataFrame(rows_diffs)

    if points:
        function_that_plots = sns.pointplot
    else:
        function_that_plots = sns.violinplot

    # _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7 * 2, 7 * 2), sharey=True)
    function_that_plots(
        data=p_interp,
        x="experiment",
        y="playability",
        ax=ax_interp_p,
        cut=1.0,
        palette=palette,
    )
    function_that_plots(
        data=p_diff,
        x="experiment",
        y="playability",
        ax=ax_diff_p,
        cut=1.0,
        palette=palette,
    )
    function_that_plots(
        data=p_interp,
        x="experiment",
        y="diversity",
        ax=ax_interp_d,
        cut=0.0,
        palette=palette,
    )
    function_that_plots(
        data=p_diff,
        x="experiment",
        y="diversity",
        ax=ax_diff_d,
        cut=0.0,
        palette=palette,
    )

    for ax in [ax_interp_p, ax_interp_d, ax_diff_d, ax_diff_p]:
        ax.set_xlabel("")
        ax.set_ylabel("")


def plot_all_violins():
    # First plots for playability
    fig, axes = plt.subplots(4, 3, figsize=(3 * 6, 2 * 5), sharex=True, sharey="row")
    # fig_d, axes_d = plt.subplots(2, 3, figsize=(3 * 7, 2 * 4), sharex=True, sharey=True)
    axes_p = axes[:2, :]
    axes_d = axes[2:, :]

    # First column: SMB
    ax_1, ax_2 = axes_p[:, 0]
    ax_3, ax_4 = axes_d[:, 0]

    experiments = [
        ("ten_vaes", "discretized_strict_gt"),
        ("ten_vaes", "baseline_strict_gt"),
        ("ten_vaes", "normal_strict_gt"),
    ]
    plot_violins(experiments, range(10), ax_1, ax_2, ax_3, ax_4)
    # First column: SMB
    ax_1, ax_2 = axes_p[:, 0]
    ax_3, ax_4 = axes_d[:, 0]

    experiments = [
        ("ten_vaes", "discretized_strict_gt"),
        ("ten_vaes", "baseline_strict_gt"),
        ("ten_vaes", "normal_strict_gt"),
    ]
    plot_violins(experiments, range(10), ax_1, ax_2, ax_3, ax_4)

    # Second column: Zelda
    ax_1, ax_2 = axes_p[:, 2]
    ax_3, ax_4 = axes_d[:, 2]

    experiments = [
        ("zelda", "zelda_discretized_grammar_gt"),
        ("zelda", "zelda_baseline_grammar_gt"),
        ("zelda", "zelda_normal_grammar_gt"),
    ]
    plot_violins(experiments, [0, 3, 5, 6], ax_1, ax_2, ax_3, ax_4)

    # First column: SMB (jumping)
    ax_1, ax_2 = axes_p[:, 1]
    ax_3, ax_4 = axes_d[:, 1]

    experiments = [
        ("ten_vaes", "discretized_force_jump_2_gt"),
        ("ten_vaes", "baseline_force_jump_2_gt"),
        ("ten_vaes", "normal_force_jump_2_gt"),
    ]
    plot_violins(experiments, range(10), ax_1, ax_2, ax_3, ax_4)

    # Setting up titles
    axes[0, 0].set_title("SMB", fontsize=BIGGER_SIZE)
    # axes_d[0, 0].set_title("SMB", fontsize=BIGGER_SIZE)

    axes[0, 1].set_title("SMB (Jump)", fontsize=BIGGER_SIZE)
    # axes_d[0, 1].set_title("SMB (Jump)", fontsize=BIGGER_SIZE)

    axes[0, 2].set_title("Zelda", fontsize=BIGGER_SIZE)
    # axes_d[0, 2].set_title("Zelda", fontsize=BIGGER_SIZE)

    # Setting up x labels
    axes[-1, 1].set_xlabel("\nExperiment", fontsize=BIGGER_SIZE)
    # axes_d[1, 1].set_xlabel("\nExperiment", fontsize=BIGGER_SIZE)

    # Setting up y labels
    axes_p[0, 0].set_ylabel("Playability (I)", fontsize=14)
    axes_p[1, 0].set_ylabel("Playability (RW)", fontsize=14)

    axes_d[0, 0].set_ylabel("Diversity (I)", fontsize=14)
    axes_d[1, 0].set_ylabel("Diversity (RW)", fontsize=14)

    # Setting up tick label sizes
    axes[-1, 0].tick_params(axis="x", labelsize=BIGGER_SIZE)
    axes[-1, 1].tick_params(axis="x", labelsize=BIGGER_SIZE)
    axes[-1, 2].tick_params(axis="x", labelsize=BIGGER_SIZE)

    # plt.show()
    # fig_p.tight_layout()
    # fig_d.tight_layout()
    # fig_p.savefig("./data/plots/ten_vaes/paper_ready/violin_plots_playability.png")
    # fig_d.savefig("./data/plots/ten_vaes/paper_ready/violin_plots_diversity.png")
    fig.tight_layout()
    fig.savefig("./violin_plots_for_presentation.png", dpi=120)
    # plt.show()


def plot_violins_for_cog_presentation():
    """
    Plots the violin comparison for playability and
    diversity independently
    """
    experiments = [
        ("ten_vaes", "discretized_strict_gt"),
        ("ten_vaes", "baseline_strict_gt"),
        ("ten_vaes", "normal_strict_gt"),
    ]

    fig_p, (ax_interp_p, ax_diff_p) = plt.subplots(1, 2, figsize=(2 * 5, 5))
    fig_d, (ax_interp_d, ax_diff_d) = plt.subplots(1, 2, figsize=(2 * 5, 5))
    plot_violins(
        experiments,
        range(10),
        ax_interp_p,
        ax_diff_p,
        ax_interp_d,
        ax_diff_d,
        points=True,
        palette="bright",
    )

    fig_p.savefig("./violins_playability.png")
    fig_d.savefig("./violins_diversity.png")

    plt.close()


if __name__ == "__main__":
    # experiments = [
    #     ("zelda", "zelda_discretized_grammar_gt"),
    #     ("zelda", "zelda_baseline_grammar_gt"),
    #     ("zelda", "zelda_normal_grammar_gt"),
    # ]
    # plot_violins_for_dataviz_course(experiments, [0, 3, 5, 6])
    # plot_all_violins()
    plot_violins_for_cog_presentation()
