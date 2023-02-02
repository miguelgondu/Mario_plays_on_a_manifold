from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.bayesian_optimization.check_traces import load_traces

from plotting_scripts.plot_banner_w_grid import BIGGER_SIZE, MEDIUM_SIZE

ROOT_DIR = Path(__file__).parent.parent.resolve()
PLOTS_DIR = ROOT_DIR / "data" / "plots" / "bayesian_optimization" / "paper_ready"

PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def plot_bayesian_optimization_traces():
    df, _ = load_traces()

    fig, (ax_jumps, ax_safety) = plt.subplots(
        2, 1, figsize=(1 * 6, 2 * 3.0), sharex=True
    )
    sns.stripplot(
        data=df,
        x="Experiment",
        y="Max. jump",
        ax=ax_jumps,
        size=8,
        edgecolor="black",
        linewidth=1.2,
        palette="Blues",
    )
    sns.stripplot(
        data=df,
        x="Experiment",
        y="Avg. playabilities",
        ax=ax_safety,
        size=8,
        edgecolor="black",
        linewidth=1.2,
        palette="Blues",
    )

    # Set up label sizes
    # x ticks
    ax_safety.tick_params(axis="x", labelsize=12)

    # y labels
    ax_jumps.set_ylabel("Max. jump", fontsize=12)
    ax_safety.set_ylabel("Avg. playabilities", fontsize=12)

    # Set up limits
    ax_jumps.set_ylim([0, 52])

    # Drawing a red line at the max of random
    # ax_jumps.axhline(df[df["Experiment"] == "Random"]["Max. jump"].max())

    # Cleaning x-labels
    for ax in [ax_jumps, ax_safety]:
        ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bayesian_optimization_traces.png", dpi=120)

    print("Summary:")
    for experiment in df["Experiment"].unique():
        print(experiment)
        print(
            "mean", df[df["Experiment"] == experiment]["Max. jump"].clip(0, 300).mean()
        )
        print("std", df[df["Experiment"] == experiment]["Max. jump"].clip(0, 300).std())
        print("max", df[df["Experiment"] == experiment]["Max. jump"].clip(0, 300).max())
        print(
            "nr. of fully playable",
            sum(df[df["Experiment"] == experiment]["Avg. playabilities"] == 1.0),
        )
        print()

    # plt.show()


if __name__ == "__main__":
    plot_bayesian_optimization_traces()
