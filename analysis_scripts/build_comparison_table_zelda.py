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

from analysis_scripts.utils import get_mean_diversities, get_mean_diversities_of_levels


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
    df, df2 = build_table()
    print(df.to_latex())
    print(df2.to_latex())
