from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.parallel

from mario_utils.plotting import get_img_from_level

# Taken from the MarioGAN repo
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(
            "initial-{0}-{1}-convt".format(nz, cngf),
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial-{0}-batchnorm".format(cngf), nn.BatchNorm2d(cngf))
        main.add_module("initial-{0}-relu".format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                "pyramid-{0}-{1}-convt".format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid-{0}-batchnorm".format(cngf // 2), nn.BatchNorm2d(cngf // 2)
            )
            main.add_module("pyramid-{0}-relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}-{1}-conv".format(t, cngf),
                nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}-{1}-batchnorm".format(t, cngf), nn.BatchNorm2d(cngf)
            )
            main.add_module("extra-layers-{0}-{1}-relu".format(t, cngf), nn.ReLU(True))

        main.add_module(
            "final-{0}-{1}-convt".format(cngf, nc),
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        main.add_module(
            "final-{0}-tanh".format(nc), nn.ReLU()
        )  # nn.Softmax(1))    #Was TANH nn.Tanh())#
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # print (output[0,:,0,0])
        # exit()
        return output

    def get_level(self, z: t.Tensor) -> t.Tensor:
        """
        Gets the important part of the level
        """
        b, z_dim = z.shape
        return self.forward(z.view(b, z_dim, 1, 1)).argmax(dim=1)[:, :14, :28]

    def plot_grid(self) -> np.ndarray:
        x_lims = y_lims = (-15, 15)
        n_cols = n_rows = 15
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])
        zs_ = zs.reshape(n_cols * n_rows, 2, 1, 1)

        images_onehot = self.forward(torch.from_numpy(zs_).type(torch.float))

        # slicing the important path of the array
        images = images_onehot.argmax(dim=1)[:, :14, :28]

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        # pixels = 16
        img_height = images.shape[1]
        img_width = images.shape[2]
        final_img = np.zeros((n_cols * img_height, n_rows * img_width, 3))
        for z, (i, j) in positions.items():
            final_img[
                i * img_height : (i + 1) * img_height,
                j * img_width : (j + 1) * img_width,
            ] = img_dict[z]

        final_img = final_img.astype(int)

        return final_img


if __name__ == "__main__":
    map_size = 32
    nz = 2
    z_dims = 10
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    generator = DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)

    generator.load_state_dict(t.load(f"./models/MarioGAN/netG_epoch_5800_0_{nz}.pth"))
    z = 3.0 * t.randn((64, nz))
    levels = generator(z.reshape(z.shape[0], z.shape[1], 1, 1)).argmax(dim=1)[
        :, :14, :28
    ]
    _, axes = plt.subplots(8, 8, figsize=(8 * 14, 8 * 7))
    for lvl, ax in zip(levels, axes.flatten()):
        ax.imshow(get_img_from_level(lvl.detach().numpy()))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "./data/plots/MarioGAN/random_samples.png", dpi=100, bbox_inches="tight"
    )
    # plt.show()
    plt.close()

    _, ax = plt.subplots(1, 1, figsize=(14, 7))
    img = generator.plot_grid()
    ax.imshow(img, extent=[-15, 15, -15, 15])
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("./data/plots/MarioGAN/grid.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
