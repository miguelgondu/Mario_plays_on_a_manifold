"""
Loads up the array's results
and gets a table.
"""

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.other_utils import (
    get_mean_diversities,
    get_mean_diversities_of_levels,
)


def load_experiment(exp_name: str, id_: int) -> Tuple[List[Path]]:
    interp_res_paths = (
        Path("./data/array_simulation_results/zelda/interpolations") / exp_name
    ).glob("*.npz")
    diff_res_paths = (
        Path("./data/array_simulation_results/zelda/diffusions") / exp_name
    ).glob("*.npz")

    filter_ = lambda x: f"final_{id_}" in x.name
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
        array = np.load(path)
        all_ps.append(np.mean(array["ps"]))
        all_ds.append(np.mean(get_mean_diversities_of_levels(array["levels"])))

    return all_ps, all_ds


def build_table_layout() -> pd.DataFrame:
    geometries = ["discretized geometry", "baseline", "normal"]

    # index = []
    # for geometry in geometries:
    #     index.append((geometry))
    # index = pd.MultiIndex.from_tuples(index, names=["Geometry", "grid"])
    index = geometries

    columns = [
        (r"$\mathbb{E}[\text{playability}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{playability}]$", "Random Walks"),
        (r"$\mathbb{E}[\text{diversity}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{diversity}]$", "Random Walks"),
    ]
    columns = pd.MultiIndex.from_tuples(columns)
    data = np.zeros((len(index), 4))
    table = pd.DataFrame(
        data,
        index=index,
        columns=columns,
    )

    print("Table layout:")
    print(table.to_latex(escape=False, float_format="%1.3f"))
    return table


def build_table() -> Tuple[pd.DataFrame]:
    experiments = [
        "zelda_baseline_grammar_gt",
        "zelda_normal_grammar_gt",
        "zelda_discretized_grammar_gt",
    ]
    rows_interps = []
    rows_diffs = []
    for experiment in experiments:
        for id_ in [0, 3, 5, 6]:
            print(f"{experiment} - {id_}")
            interps, diffs = load_experiment(experiment, id_)
            all_ps, all_ds = get_all_means(interps)
            print("interps")
            print(np.mean(all_ps), np.mean(all_ds))
            rows_interps.extend(
                [
                    {
                        "id": id_,
                        "playability": m,
                        "experiment": experiment,
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
                        "experiment": experiment,
                        "diversity": d,
                    }
                    for m, d in zip(all_ps, all_ds)
                ]
            )

    p_interp = pd.DataFrame(rows_interps)
    p_diff = pd.DataFrame(rows_diffs)

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(7 * 2, 7 * 2), sharey=True
    )
    sns.violinplot(data=p_interp, x="experiment", y="playability", ax=ax1, cut=1.0)
    sns.violinplot(data=p_diff, x="experiment", y="playability", ax=ax2, cut=1.0)
    sns.violinplot(data=p_interp, x="experiment", y="diversity", ax=ax3)
    sns.violinplot(data=p_diff, x="experiment", y="diversity", ax=ax4)
    # plt.show()

    rows_i = []
    rows_d = []
    for experiment in experiments:
        slice_ = p_interp[p_interp["experiment"] == experiment]
        rows_i.append(
            {
                "experiment": experiment,
                "mean_playability": slice_["playability"].mean(),
                "std_playability": slice_["playability"].std(),
                "mean_diversity": slice_["diversity"].mean(),
                "std_diversity": slice_["diversity"].std(),
            }
        )
        slice_ = p_diff[p_diff["experiment"] == experiment]
        rows_d.append(
            {
                "experiment": experiment,
                "mean_playability": slice_["playability"].mean(),
                "std_playability": slice_["playability"].std(),
                "mean_diversity": slice_["diversity"].mean(),
                "std_diversity": slice_["diversity"].std(),
            }
        )

    return pd.DataFrame(rows_i), pd.DataFrame(rows_d)


if __name__ == "__main__":
    table = build_table_layout()
    E_playability = r"$\mathbb{E}[\text{playability}]$"
    E_diversity = r"$\mathbb{E}[\text{diversity}]$"

    df, df2 = build_table()
    print(df.to_latex())
    print(df2.to_latex())

    # for interps
    mp_interp_discretized = df[df["experiment"] == "zelda_discretized_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_interp_discretized = df[df["experiment"] == "zelda_discretized_grammar_gt"][
        "std_playability"
    ].values[0]
    md_interp_discretized = df[df["experiment"] == "zelda_discretized_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_interp_discretized = df[df["experiment"] == "zelda_discretized_grammar_gt"][
        "std_diversity"
    ].values[0]

    mp_interp_normal = df[df["experiment"] == "zelda_normal_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_interp_normal = df[df["experiment"] == "zelda_normal_grammar_gt"][
        "std_playability"
    ].values[0]
    md_interp_normal = df[df["experiment"] == "zelda_normal_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_interp_normal = df[df["experiment"] == "zelda_normal_grammar_gt"][
        "std_diversity"
    ].values[0]

    mp_interp_baseline = df[df["experiment"] == "zelda_baseline_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_interp_baseline = df[df["experiment"] == "zelda_baseline_grammar_gt"][
        "std_playability"
    ].values[0]
    md_interp_baseline = df[df["experiment"] == "zelda_baseline_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_interp_baseline = df[df["experiment"] == "zelda_baseline_grammar_gt"][
        "std_diversity"
    ].values[0]

    # for diffs
    mp_diff_discretized = df2[df2["experiment"] == "zelda_discretized_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_diff_discretized = df2[df2["experiment"] == "zelda_discretized_grammar_gt"][
        "std_playability"
    ].values[0]
    md_diff_discretized = df2[df2["experiment"] == "zelda_discretized_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_diff_discretized = df2[df2["experiment"] == "zelda_discretized_grammar_gt"][
        "std_diversity"
    ].values[0]

    mp_diff_normal = df2[df2["experiment"] == "zelda_normal_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_diff_normal = df2[df2["experiment"] == "zelda_normal_grammar_gt"][
        "std_playability"
    ].values[0]
    md_diff_normal = df2[df2["experiment"] == "zelda_normal_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_diff_normal = df2[df2["experiment"] == "zelda_normal_grammar_gt"][
        "std_diversity"
    ].values[0]

    mp_diff_baseline = df2[df2["experiment"] == "zelda_baseline_grammar_gt"][
        "mean_playability"
    ].values[0]
    sp_diff_baseline = df2[df2["experiment"] == "zelda_baseline_grammar_gt"][
        "std_playability"
    ].values[0]
    md_diff_baseline = df2[df2["experiment"] == "zelda_baseline_grammar_gt"][
        "mean_diversity"
    ].values[0]
    sd_diff_baseline = df2[df2["experiment"] == "zelda_baseline_grammar_gt"][
        "std_diversity"
    ].values[0]

    table.loc[("discretized geometry"), (E_playability, "Interpolation")] = (
        f"{mp_interp_discretized:.3f}" + r"$\pm$" + f"{sp_interp_discretized:.3f}"
    )

    table.loc[("discretized geometry"), (E_playability, "Random Walks")] = (
        f"{mp_diff_discretized:.3f}" + r"$\pm$" + f"{sp_diff_discretized:.3f}"
    )

    table.loc[("discretized geometry"), (E_diversity, "Interpolation")] = (
        f"{md_interp_discretized:.3f}" + r"$\pm$" + f"{sd_interp_discretized:.3f}"
    )

    table.loc[("discretized geometry"), (E_diversity, "Random Walks")] = (
        f"{md_diff_discretized:.3f}" + r"$\pm$" + f"{sd_diff_discretized:.3f}"
    )

    table.loc[("normal"), (E_playability, "Interpolation")] = (
        f"{mp_interp_normal:.3f}" + r"$\pm$" + f"{sp_interp_normal:.3f}"
    )

    table.loc[("normal"), (E_playability, "Random Walks")] = (
        f"{mp_diff_normal:.3f}" + r"$\pm$" + f"{sp_diff_normal:.3f}"
    )

    table.loc[("normal"), (E_diversity, "Interpolation")] = (
        f"{md_interp_normal:.3f}" + r"$\pm$" + f"{sd_interp_normal:.3f}"
    )

    table.loc[("normal"), (E_diversity, "Random Walks")] = (
        f"{md_diff_normal:.3f}" + r"$\pm$" + f"{sd_diff_normal:.3f}"
    )

    table.loc[("baseline"), (E_playability, "Interpolation")] = (
        f"{mp_interp_baseline:.3f}" + r"$\pm$" + f"{sp_interp_baseline:.3f}"
    )

    table.loc[("baseline"), (E_playability, "Random Walks")] = (
        f"{mp_diff_baseline:.3f}" + r"$\pm$" + f"{sp_diff_baseline:.3f}"
    )

    table.loc[("baseline"), (E_diversity, "Interpolation")] = (
        f"{md_interp_baseline:.3f}" + r"$\pm$" + f"{sd_interp_baseline:.3f}"
    )

    table.loc[("baseline"), (E_diversity, "Random Walks")] = (
        f"{md_diff_baseline:.3f}" + r"$\pm$" + f"{sd_diff_baseline:.3f}"
    )

    print(table.to_latex(escape=False))
