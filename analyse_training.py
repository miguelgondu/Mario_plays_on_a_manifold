"""
This script loads the json files
from the training and analyses them.
"""
import json
from operator import itemgetter
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

training_data_path = Path("./data/results")
all_results = training_data_path.glob("*.json")
# print(list(all_results))

rows = []
for result in all_results:
    with open(result) as fp:
        losses = json.load(fp)
    row = {
        "model_name": result.name.replace(".json", ""),
        "test_losses": losses["test_losses"],
    }
    rows.append(row)

final_fig, ax_f = plt.subplots(1, 1, figsize=(14, 10))

colors = np.random.rand(18, 3).tolist()
# colors = [
#     (0, 0, k) for k in np.linspace(0, 1, 18, dtype=float)
# ]
print(colors)

lowest_test_loss = {}
for z_dim in [2, 3, 8, 16, 32, 38]:
    # Filter the rows by this z_dim
    _, ax_z = plt.subplots(1, 1, figsize=(10, 10))
    for h_dims in [[256, 128], [512, 256, 128], [1024, 512, 256]]:
        print(f"z dim: {z_dim}, h dims: {h_dims}")

        def model_filter(d):
            cond = f"_z_dim_{z_dim}_" in d["model_name"]
            h_dims_str = "_h_dims"
            for h_dim in h_dims:
                h_dims_str += f"_{h_dim}"

            cond = cond and (h_dims_str in d["model_name"])
            return cond

        z_dim_rows = filter(model_filter, rows)

        # print(f"z_dim_rows: {z_dim_rows}")
        # Compute mean and variance per epoch
        test_losses_by_epoch = defaultdict(list)
        for row in z_dim_rows:
            for i, loss in enumerate(row["test_losses"]):
                test_losses_by_epoch[i].append(loss)

        print(test_losses_by_epoch)
        cutpoint = None
        for i, loss_l in test_losses_by_epoch.items():
            cutpoint = i
            if len(loss_l) < 2:
                break

        test_loss_mean = {i: np.mean(loss) for i, loss in test_losses_by_epoch.items()}
        test_loss_std = {i: np.std(loss) for i, loss in test_losses_by_epoch.items()}
        test_loss_mean = np.array(sorted(test_loss_mean.items(), key=lambda x: x[0]))[
            :, 1
        ]
        test_loss_std = np.array(sorted(test_loss_std.items(), key=lambda x: x[0]))[
            :, 1
        ]
        # print(test_loss_mean)
        # print(test_loss_std)
        lowest_test_loss[f"z dim: {z_dim}, h dims: {h_dims}"] = test_loss_mean[
            : min(100, len(test_loss_mean))
        ].min()

        color = colors.pop()
        for ax in [ax_f, ax_z]:
            ax.plot(
                test_loss_mean, label=f"z dim: {z_dim}, h dims: {h_dims}", color=color
            )
            ax.fill_between(
                np.arange(len(test_loss_mean)),
                test_loss_mean - test_loss_std,
                test_loss_mean + test_loss_std,
                alpha=0.2,
                color=color,
            )
    ax_z.legend()

# ax.axvline(x=cutpoint, color="k", label="< 2 models.")
ax_f.legend()
ax_f.set_xlabel("Epoch")
ax_f.set_ylabel("Test loss")
ax_f.set_title("Training results for gridsearch")
# plt.show()
final_fig.savefig("training_results.png")

print(lowest_test_loss)
print(sorted(lowest_test_loss.items(), key=itemgetter(1)))
print(sorted(lowest_test_loss.items(), key=itemgetter(1))[0])


# print(rows)
