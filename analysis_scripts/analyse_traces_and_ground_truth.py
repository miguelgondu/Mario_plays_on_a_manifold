from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from evolving_playability import get_ground_truth

traces = Path("./data/evolution_traces").glob("trace_*.npz")

for f in traces:
    array = np.load(f)
    all_zs = array["zs"]
    all_playabilities = array["playabilities"]

    gt_img = get_ground_truth()
    for m in range(500):
        # print(str(f), len(zs))
        print(f"{f.name}, {m:03d}/500", flush=True, end="\r")
        zs = all_zs[: 100 + m]
        playabilities = all_playabilities[: 100 + m]

        gpc = GaussianProcessClassifier()
        gpc.fit(zs, playabilities)

        z1s = np.linspace(-5, 5, 50)
        z2s = np.linspace(-5, 5, 50)

        bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
        res, var = gpc.predict_proba(bigger_grid, return_var=True)
        predictions = gpc.predict(bigger_grid)

        p_dict = {(z[0], z[1]): r[1] for z, r in zip(bigger_grid, res)}
        var_dict = {(z[0], z[1]): v for z, v in zip(bigger_grid, var)}
        pred_dict = {(z[0], z[1]): pred for z, pred in zip(bigger_grid, predictions)}

        positions = {
            (x, y): (i, j)
            for j, x in enumerate(z1s)
            for i, y in enumerate(reversed(z2s))
        }

        var_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            var_img[i, j] = var_dict[z]

        p_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            p_img[i, j] = p_dict[z]

        pred_img = np.zeros((len(z2s), len(z1s)))
        for z, (i, j) in positions.items():
            pred_img[i, j] = pred_dict[z]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7 * 2))

        plt.title(f"{f.name}, m={m:03d}")
        ax1.imshow(gt_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
        ax2.imshow(p_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
        ax2.scatter(
            zs[:, 0], zs[:, 1], c=playabilities, cmap="Wistia", vmin=0.0, vmax=1.0
        )
        fig.tight_layout()
        fig.savefig(
            f"./data/plots/evolving_playability/for_videos/{m:03d}_{f.name.replace('.npz', '.png')}"
        )
        plt.close("all")
