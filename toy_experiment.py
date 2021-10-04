from typing import List, Tuple
import json
from geoml.curve import CubicSpline

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_text import parse_syntactically
from vae_geometry_text import VAEGeometryText
from vae_geometry_hierarchical_text import VAEGeometryHierarchicalText

from geoml.discretized_manifold import DiscretizedManifold

from interpolations.base_interpolation import BaseInterpolation
from interpolations.linear_interpolation import LinearInterpolation
from interpolations.geodesic_interpolation import GeodesicInterpolation

from diffusions.normal_diffusion import NormalDifussion
from diffusions.baseline_diffusion import BaselineDiffusion
from diffusions.geometric_difussion import GeometricDifussion


def get_random_pairs(
    encodings: t.Tensor, n_pairs: int = 100, seed: int = 17
) -> List[t.Tensor]:
    np.random.seed(seed)
    idx1 = np.random.choice(len(encodings), size=n_pairs, replace=False)
    idx2 = np.random.choice(len(encodings), size=n_pairs, replace=False)
    while np.any(idx1 == idx2):
        idx2 = np.random.choice(len(encodings), size=n_pairs, replace=False)

    pairs_1 = encodings[idx1]
    pairs_2 = encodings[idx2]

    return pairs_1, pairs_2


def get_interpolations(
    vae, n_points_in_line=10
) -> Tuple[LinearInterpolation, GeodesicInterpolation]:
    """
    Returns the interpolations for the experiment. Here
    is where the hyperparameters can be found.
    """
    # Linear interpolation
    li = LinearInterpolation(n_points_in_line=n_points_in_line)

    # Geodesic interpolation
    grid = [t.linspace(-5, 5, 50), t.linspace(-5, 5, 50)]
    Mx, My = t.meshgrid(grid[0], grid[1])
    grid2 = t.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)
    DM = DiscretizedManifold(vae, grid2, use_diagonals=True)
    gi = GeodesicInterpolation(DM, n_points_in_line=n_points_in_line)

    return li, gi


def get_diffusions() -> Tuple[NormalDifussion, BaselineDiffusion, GeometricDifussion]:
    n_points = 50

    normal_diffusion = NormalDifussion(n_points, scale=0.5)
    geometric_diffusion = GeometricDifussion(n_points, scale=0.01)
    baseline_diffusion = BaselineDiffusion(n_points, step_size=0.5)

    return normal_diffusion, baseline_diffusion, geometric_diffusion


def get_expected_coherences(points: t.Tensor, vae) -> Tuple[List[float], List[str]]:
    """
    Grabs a tensor of points ([n_points, 2] tensor) and returns a list
    with the expected coherences of each zi.
    """
    expected_coherences = []
    all_sequences = []
    for z in points:
        dist = vae.decode(z)
        samples = dist.sample((100,))
        sequences_in_samples = [vae.int_sequence_to_text(s[0]) for s in samples]
        coherences_at_z = [parse_syntactically(seq) for seq in sequences_in_samples]
        expected_coherences.append(np.mean(coherences_at_z))
        all_sequences += sequences_in_samples

    return expected_coherences, all_sequences


def inspect_model(vae):
    """
    Plots the latent space and coherence
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7 * 1))
    coherence = vae.get_correctness_img("syntactic", sample=True)
    zs = vae.encodings.detach().numpy()
    ax1.scatter(zs[:, 0], zs[:, 1], marker="x", c="k")
    ax1.imshow(coherence, extent=[-5, 5, -5, 5], cmap="Blues")

    vae.plot_latent_space(ax=ax2)
    plt.tight_layout()
    plt.show()


def interpolation_experiment(vae) -> List[t.Tensor]:
    z_0s, z_1s = get_random_pairs(vae.encodings, n_pairs=50, seed=0)
    li, gi = get_interpolations(vae)

    fifty_lines = [li.interpolate(z_0, z_1) for z_0, z_1 in zip(z_0s, z_1s)]
    fifty_geodesics_splines = [
        gi.interpolate_and_return_geodesic(z_0, z_1) for z_0, z_1 in zip(z_0s, z_1s)
    ]

    domain = t.linspace(0, 1, gi.n_points_in_line)
    fifty_geodesics = [c(domain) for c in fifty_geodesics_splines]

    all_sequences_in_lines = []
    all_sequences_in_geodesics = []
    expected_coherences_in_lines = []
    expected_coherences_in_geodesics = []

    for line in fifty_lines:
        expected_coherences, all_sequences = get_expected_coherences(line, vae)
        expected_coherences_in_lines.append(expected_coherences)
        all_sequences_in_lines.append(all_sequences)

    for geodesic in fifty_geodesics:
        expected_coherences, all_sequences = get_expected_coherences(geodesic, vae)
        expected_coherences_in_geodesics.append(expected_coherences)
        all_sequences_in_geodesics.append(all_sequences)

    print("Mean coherence in lines")
    print(np.mean(expected_coherences_in_lines))
    print("Mean coherence in geodesics")
    print(np.mean(expected_coherences_in_geodesics))
    unpacked_sequences_in_lines = [s for seq in all_sequences_in_lines for s in seq]
    unpacked_sequences_in_geodesics = [
        s for seq in all_sequences_in_geodesics for s in seq
    ]
    print(
        f"Unique sec. percentage in lines: {len(set(unpacked_sequences_in_lines)) / len(unpacked_sequences_in_lines)}"
    )
    print(
        f"Unique sec. percentage in geodesics: {len(set(unpacked_sequences_in_geodesics)) / len(unpacked_sequences_in_geodesics)}"
    )

    # Plot the first 5.
    # _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    # correctness = vae.get_correctness_img("syntactic", sample=True)
    # ax1.imshow(correctness, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
    # ax2.imshow(correctness, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
    # for line, geodesic, i in zip(fifty_lines, fifty_geodesics, range(15)):
    #     line = line.detach().numpy()
    #     geodesic = geodesic.detach().numpy()

    #     coh_line = expected_coherences_in_lines[i]
    #     coh_geodesic = expected_coherences_in_geodesics[i]

    #     ax1.scatter(line[:, 0], line[:, 1], c=coh_line, vmin=0.0, vmax=1.0)
    #     ax2.scatter(geodesic[:, 0], geodesic[:, 1], c=coh_geodesic, vmin=0.0, vmax=1.0)

    # ax1.set_title("Lines")
    # ax2.set_title("Geodesics")
    # plt.tight_layout()
    # plt.show()

    return (
        fifty_lines,
        fifty_geodesics_splines,
        expected_coherences_in_lines,
        expected_coherences_in_geodesics,
        all_sequences_in_lines,
        all_sequences_in_geodesics,
    )


def diffusion_experiment(vae):
    n_runs = 10
    normal_diff, baseline_diff, geometric_diff = get_diffusions()

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    vae.plot_latent_space(ax=(ax1, ax2, ax3))

    mean_coherences_n = []
    mean_coherences_b = []
    mean_coherences_g = []
    all_sequences_n = []
    all_sequences_b = []
    all_sequences_g = []
    for _ in range(n_runs):
        zs_n = normal_diff.run(vae)
        zs_b = baseline_diff.run(vae)
        zs_g = geometric_diff.run(vae)

        coherences_n, sequences_n = get_expected_coherences(zs_n, vae)
        coherences_b, sequences_b = get_expected_coherences(zs_b, vae)
        coherences_g, sequences_g = get_expected_coherences(zs_g, vae)

        mean_coherences_n.append(np.mean(coherences_n))
        mean_coherences_b.append(np.mean(coherences_b))
        mean_coherences_g.append(np.mean(coherences_g))
        all_sequences_n += sequences_n
        all_sequences_b += sequences_b
        all_sequences_g += sequences_g

        zs_n = zs_n.detach().numpy()
        zs_b = zs_b.detach().numpy()
        zs_g = zs_g.detach().numpy()

        ax1.scatter(zs_n[:, 0], zs_n[:, 1], c=coherences_n, vmin=0.0, vmax=1.0)
        ax2.scatter(zs_b[:, 0], zs_b[:, 1], c=coherences_b, vmin=0.0, vmax=1.0)
        ax3.scatter(zs_g[:, 0], zs_g[:, 1], c=coherences_g, vmin=0.0, vmax=1.0)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim((-5, 5))
        ax.axis("off")

    print(f"Mean coherence (Normal): {np.mean(mean_coherences_n)}")
    print(f"Mean coherence (Baseline): {np.mean(mean_coherences_b)}")
    print(f"Mean coherence (Geometric): {np.mean(mean_coherences_g)}")

    print(
        f"Unique seq. percentage (Normal): {len(set(all_sequences_n)) / len(all_sequences_n)} ({len(set(all_sequences_n))} / {len(all_sequences_n)})"
    )
    print(
        f"Unique seq. percentage (Baseline): {len(set(all_sequences_b)) / len(all_sequences_b)} ({len(set(all_sequences_b))} / {len(all_sequences_b)})"
    )
    print(
        f"Unique seq. percentage (Geometric): {len(set(all_sequences_g)) / len(all_sequences_g)} ({len(set(all_sequences_g))} / {len(all_sequences_g)})"
    )

    with open("./data/processed/sequences_n.json", "w") as fp:
        json.dump(all_sequences_n, fp)

    with open("./data/processed/sequences_b.json", "w") as fp:
        json.dump(all_sequences_b, fp)

    with open("./data/processed/sequences_g.json", "w") as fp:
        json.dump(all_sequences_g, fp)

    ax1.set_title("Normal")
    ax2.set_title("Baseline")
    ax3.set_title("Geometric")
    plt.tight_layout()
    plt.show()


def figure_latent_codes(vae: VAEGeometryHierarchicalText):
    """
    Gets an image of the latent space with the latent codes,
    illuminated by coherence after 100% samples.
    """
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    x_lims = (-4, 4)
    y_lims = (-4, 4)
    img = vae.get_correctness_img(
        "syntactic", x_lims=x_lims, y_lims=y_lims, sample=True
    )
    plot = ax.imshow(img, extent=[*x_lims, *y_lims], cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)

    zs = vae.encodings.detach().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], c="#DC851F", marker="x")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("./data/plots/final/equation_model_latent_codes.png", dpi=100)
    plt.show()
    plt.close()


def plot_line_w_color(ax, line, color):
    pass


def figure_interpolations(vae: VAEGeometryHierarchicalText):
    """
    Saves the figure for metric volume and interpolations, as well
    as a table for expected syntactc coherence.
    """
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    # vae.plot_metric_volume(ax=ax, x_lims=(-4, 4), y_lims=(-4, 4))
    img = vae.get_correctness_img(
        "syntactic", x_lims=(-4, 4), y_lims=(-4, 4), sample=True
    )
    _ = ax.imshow(img, extent=[-4, 4, -4, 4], cmap="Blues", vmin=0.0, vmax=1.0)

    li, gi = get_interpolations(vae)
    # z_0 = t.Tensor([-2.5749447, 0.6269546])
    # z_1 = t.Tensor([2.54, -0.17])
    # z_0 = t.Tensor([-1.0, 1.98])
    # z_1 = t.Tensor([1.0, -1.92])
    z_0 = t.Tensor([2.35, -1.0])
    z_1 = t.Tensor([-2.55, 0.5])
    # z_0 = t.Tensor([-0.14, -2.1])
    # z_1 = t.Tensor([0.05, 1.88])

    line = li.interpolate(z_0, z_1)
    geodesic = gi.interpolate_and_return_geodesic(z_0, z_1)
    domain = t.linspace(0, 1, li.n_points_in_line)
    geodesic_t = geodesic(domain)

    coh_line, _ = get_expected_coherences(line, vae)
    coh_geodesic, _ = get_expected_coherences(geodesic_t, vae)

    print(f"Mean coherence (Line): {np.mean(coh_line)}")
    print(f"Mean coherence (Geodesic): {np.mean(coh_geodesic)}")

    line = line.detach().numpy()
    ax.plot(line[:, 0], line[:, 1], label="Linear")
    geodesic_arr = geodesic(domain).detach().numpy()
    geodesic.plot(ax=ax, N=gi.n_points_in_line, label="Geodesic")
    plot = ax.scatter(line[:, 0], line[:, 1], c=coh_line, vmin=0.0, vmax=1.0, zorder=5)
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter(
        geodesic_arr[:, 0],
        geodesic_arr[:, 1],
        c=coh_geodesic,
        vmin=0.0,
        vmax=1.0,
        zorder=5,
    )
    ax.legend()

    # (
    #     fifty_lines,
    #     fifty_geodesics_splines,
    #     expected_coherences_in_lines,
    #     expected_coherences_in_geodesics,
    #     all_sequences_in_lines,
    #     all_sequences_in_geodesics,
    # ) = interpolation_experiment(vae)

    # iterator = zip(
    #     fifty_lines,
    #     fifty_geodesics_splines,
    #     expected_coherences_in_lines,
    #     expected_coherences_in_geodesics,
    #     range(3),
    # )
    # _, gi = get_interpolations(vae)
    # domain = t.linspace(0, 1, gi.n_points_in_line)
    # for line, geodesic, coh_line, coh_geodesic, _ in iterator:
    #     line = line.detach().numpy()
    #     ax.plot(line[:, 0], line[:, 1])
    #     geodesic_arr = geodesic(domain).detach().numpy()
    #     geodesic.plot(ax=ax, N=gi.n_points_in_line)
    #     ax.scatter(line[:, 0], line[:, 1], c=coh_line, vmin=0.0, vmax=1.0, zorder=5)
    #     ax.scatter(
    #         geodesic_arr[:, 0],
    #         geodesic_arr[:, 1],
    #         c=coh_geodesic,
    #         vmin=0.0,
    #         vmax=1.0,
    #         zorder=5,
    #     )

    # TODO: finish implementing this plot
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("./data/plots/final/equation_model_interpolation_example.png", dpi=100)
    plt.show()
    plt.close()


if __name__ == "__main__":
    vaeh = VAEGeometryHierarchicalText()
    vaeh.load_state_dict(t.load("./models/text/hierarchical_vae_text_final.pt"))
    vaeh.update_cluster_centers(beta=-3.5)

    # inspect_model(vaeh)
    # interpolation_experiment(vaeh)
    # diffusion_experiment(vaeh)

    figure_latent_codes(vaeh)
    figure_interpolations(vaeh)
