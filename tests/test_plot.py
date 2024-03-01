import matplotlib.pyplot as plt
import pytest
import scipy.stats

import lsbi.stats
from lsbi.plot import (
    make_1d_axes,
    make_2d_axes,
    pdf_plot_1d,
    pdf_plot_2d,
    plot_1d,
    plot_2d,
    scatter_plot_2d,
)


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


dists = [
    lsbi.stats.multivariate_normal(),
    lsbi.stats.multivariate_normal(dim=5),
    scipy.stats.multivariate_normal(),
]


@pytest.mark.parametrize("dist", dists)
def test_pdf_plot_1d(dist):
    fig, ax = plt.subplots()
    pdf_plot_1d(ax, dist)
    pdf_plot_1d(ax, dist, edgecolor="k")
    pdf_plot_1d(ax, dist, facecolor=True)
    pdf_plot_1d(ax, dist, density=True)


@pytest.mark.parametrize("dist", dists)
def test_pdf_plot_2d(dist):
    fig, ax = plt.subplots()
    pdf_plot_2d(ax, dist)
    pdf_plot_2d(ax, dist, facecolor=None, ec="k")


@pytest.mark.parametrize("dist", dists)
def test_scatter_plot_2d(dist):
    fig, ax = plt.subplots()
    scatter_plot_2d(ax, dist)


@pytest.mark.parametrize("dist", dists)
def test_plot_1d(dist):
    plot_1d(dist)
    if dist.dim > 1:
        plot_1d(dist, [0, 2, 4])
    fig, ax = make_1d_axes(list(range(dist.dim)))
    plot_1d(dist, ax)


@pytest.mark.parametrize("dist", dists)
def test_plot_2d(dist):
    plot_2d(dist)
    if dist.dim > 1:
        plot_2d(dist, [0, 2, 4])
    fig, ax = make_2d_axes(list(range(dist.dim)))
    plot_2d(dist, ax)
