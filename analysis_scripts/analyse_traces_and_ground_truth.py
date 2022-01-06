from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from evolving_playability import get_ground_truth


array = np.load("./data/evolution_traces/bigger_trace.npz")
zs = array["zs"]
playabilities = array["playabilities"]

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

positions = {(x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(z2s)}

var_img = np.zeros((len(z2s), len(z1s)))
for z, (i, j) in positions.items():
    var_img[i, j] = var_dict[z]

p_img = np.zeros((len(z2s), len(z1s)))
for z, (i, j) in positions.items():
    p_img[i, j] = p_dict[z]

pred_img = np.zeros((len(z2s), len(z1s)))
for z, (i, j) in positions.items():
    pred_img[i, j] = pred_dict[z]

_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7 * 2, 7 * 2))

gt_img = get_ground_truth()
ax1.imshow(gt_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
ax2.imshow(pred_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
ax3.imshow(p_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
ax3.scatter(zs[:, 0], zs[:, 1], c=playabilities, cmap="Wistia", vmin=0.0, vmax=1.0)
ax4.imshow(var_img, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
plt.show()
