import torch as t
import matplotlib.pyplot as plt

from vae_text import VAEText

vaetext = VAEText()
vaetext.load_state_dict(t.load(f"./models/test_text_final.pt"))

img = vaetext.plot_syntactic_correctness()
_, ax = plt.subplots(1, 1)
ax.imshow(img, extent=[-5, 5, -5, 5], cmap="Blues")
latent_codes, _ = vaetext.forward(vaetext.train_tensor)
latent_codes = latent_codes.mean.detach().numpy()
ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")
plt.show()
