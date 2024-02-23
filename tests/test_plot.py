import matplotlib.pyplot as plt
import pytest
import scipy.stats

import lsbi.stats
from lsbi.plot import pdf_plot_1d, pdf_plot_2d


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


dists = [lsbi.stats.multivariate_normal(), scipy.stats.multivariate_normal()]


@pytest.mark.parametrize("dist", dists)
def test_pdf_plot_1d(dist):
    fig, ax = plt.subplots()
    pdf_plot_1d(ax, dist)
    pdf_plot_1d(ax, dist, edgecolor="k")
    pdf_plot_1d(ax, dist, facecolor=True)
    pdf_plot_1d(ax, dist, density=True)


@pytest.mark.parametrize("dist", dists)
def test_pdf_plot_2d(dist):
    dist = scipy.stats.multivariate_normal([0.1, 0.2])
    fig, ax = plt.subplots()
    pdf_plot_2d(ax, dist)
    pdf_plot_2d(ax, dist, facecolor=None, ec="k")


@pytest.mark.parametrize("dist", dists)
def test_scatter_plot_2d(dist):
    fig, ax = plt.subplots()
    scatter_plot_2d(ax, dist)


@pytest.mark.parametrize("dist", dists)
def test_plot_1d(dist):
    fig, ax = plt.subplots()
    plot_1d(dist, ax)


@pytest.mark.parametrize("dist", dists)
def test_plot_2d(dist):
    fig, ax = plt.subplots()
    plot_2d(dist, ax)
