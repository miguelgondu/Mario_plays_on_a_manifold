import matplotlib.pyplot as plt
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
    levels = generator(z.reshape(z.shape[0], z.shape[1], 1, 1)).argmax(dim=1)
    _, axes = plt.subplots(8, 8, figsize=(8 * 7, 8 * 7))
    for lvl, ax in zip(levels, axes.flatten()):
        ax.imshow(get_img_from_level(lvl.detach().numpy()))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "./data/plots/MarioGAN/random_samples.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()
