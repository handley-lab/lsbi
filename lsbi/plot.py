"""anesthetic-style plotting functions for distributions."""

import matplotlib.cbook as cbook
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import make_1d_axes, make_2d_axes
from anesthetic.plot import (
    AxesDataFrame,
    AxesSeries,
    basic_cmap,
    normalize_kwargs,
    quantile_plot_interval,
)
from anesthetic.plot import scatter_plot_2d as anesthetic_scatter_plot_2d
from anesthetic.plot import set_colors
from anesthetic.utils import (
    iso_probability_contours_from_samples,
    match_contour_to_contourf,
)
from matplotlib.colors import LinearSegmentedColormap


def pdf_plot_1d(ax, dist, *args, **kwargs):
    """Plot a 1D probability density estimate.

    This is in the same style as anesthetic, but since we have analytic expressions for the marginal densities we can plot the pdf directly

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`, with
    a kernel density estimation computation provided by
    :class:`scipy.stats.gaussian_kde` in-between. All remaining keyword
    arguments are passed onwards.

    Parameters
    ----------
    ax: :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    dist: statistical distribution to plot
        This should have a `logpdf` method and a `rvs` method, operating on
        one-dimensional inputs

    levels : list
        Values at which to draw iso-probability lines.
        Default: [0.95, 0.68]

    facecolor : bool or string, default=False
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).
    """
    kwargs = normalize_kwargs(kwargs)
    nplot = kwargs.get("nplot_1d", 10000)

    levels = kwargs.pop("levels", [0.95, 0.68])
    density = kwargs.pop("density", False)

    cmap = kwargs.pop("cmap", None)
    color = kwargs.pop(
        "color",
        (ax._get_lines.get_next_color() if cmap is None else plt.get_cmap(cmap)(0.68)),
    )
    facecolor = kwargs.pop("facecolor", False)
    if "edgecolor" in kwargs:
        edgecolor = kwargs.pop("edgecolor")
        if edgecolor:
            color = edgecolor
    else:
        edgecolor = color

    x = dist.rvs(nplot)
    logpdf = dist.logpdf(x)
    logpdfmin = np.sort(logpdf)[::-1][int(0.997 * nplot)]
    x = np.atleast_2d(x)[..., 0]
    i = np.argsort(x)
    x = x[i]
    logpdf = logpdf[i]
    logpdf[logpdf < logpdfmin] = np.nan
    if not density:
        logpdf -= np.nanmax(logpdf)
    pdf = np.exp(logpdf)
    ans = ax.plot(x, pdf, color=color, *args, **kwargs)

    if facecolor and facecolor not in [None, "None", "none"]:
        if facecolor is True:
            facecolor = color

        c = iso_probability_contours_from_samples(pdf, contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for j in range(len(c) - 1):
            fill.append(
                ax.fill_between(
                    x, pdf, where=pdf >= c[j], color=cmap(c[j]), edgecolor=edgecolor
                )
            )

        ans = ans, fill

    if density:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1.1)

    return ans


def pdf_plot_2d(ax, dist, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This is in the same style as anesthetic, but since we have analytic expressions for the marginal densities we can plot the pdf directly

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.contour`
    and :meth:`matplotlib.axes.Axes.contourf` with a kernel density
    estimation (KDE) computation provided by :class:`scipy.stats.gaussian_kde`
    in-between. All remaining keyword arguments are passed onwards to both
    functions.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    dist: statistical distribution to plot
        This should have a `logpdf` method and a `rvs` method, operating on
        two-dimensional inputs

    levels : list, optional
        Amount of mass within each iso-probability contour.
        Has to be ordered from outermost to innermost contour.
        Default: [0.95, 0.68]

    nplot_2d : int, default=1000
        Number of plotting points to use.

    Returns
    -------
    c : :class:`matplotlib.contour.QuadContourSet`
        A set of contourlines or filled regions.

    """
    kwargs = normalize_kwargs(
        kwargs,
        dict(
            linewidths=["linewidth", "lw"],
            linestyles=["linestyle", "ls"],
            color=["c"],
            facecolor=["fc"],
            edgecolor=["ec"],
        ),
    )

    nplot = kwargs.pop("nplot_2d", 10000)
    label = kwargs.pop("label", None)
    zorder = kwargs.pop("zorder", 1)
    levels = kwargs.pop("levels", [0.95, 0.68])

    color = kwargs.pop("color", ax._get_lines.get_next_color())
    facecolor = kwargs.pop("facecolor", True)
    edgecolor = kwargs.pop("edgecolor", None)
    cmap = kwargs.pop("cmap", None)
    facecolor, edgecolor, cmap = set_colors(
        c=color, fc=facecolor, ec=edgecolor, cmap=cmap
    )

    x = dist.rvs(nplot)
    P = dist.pdf(x)
    levels = iso_probability_contours_from_samples(P, contours=levels)
    y = np.atleast_1d(x[..., 1])
    x = np.atleast_1d(x[..., 0])

    if facecolor not in [None, "None", "none"]:
        linewidths = kwargs.pop("linewidths", 0.5)
        contf = ax.tricontourf(
            x,
            y,
            P,
            levels=levels,
            cmap=cmap,
            zorder=zorder,
            vmin=0,
            vmax=P.max(),
            *args,
            **kwargs,
        )
        contf.set_cmap(cmap)
        ax.add_patch(
            plt.Rectangle(
                (0, 0), 0, 0, lw=2, label=label, fc=cmap(0.999), ec=cmap(0.32)
            )
        )
        cmap = None
    else:
        linewidths = kwargs.pop("linewidths", plt.rcParams.get("lines.linewidth"))
        contf = None
        fc = "None" if cmap is None else cmap(0.999)
        ec = edgecolor if cmap is None else cmap(0.32)
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, lw=2, label=label, fc=fc, ec=ec))

    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=P.max())
    cont = ax.tricontour(
        x,
        y,
        P,
        levels=levels,
        zorder=zorder,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        colors=edgecolor,
        cmap=cmap,
        *args,
        **kwargs,
    )

    return contf, cont


def scatter_plot_2d(ax, dist, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`,
    enforcing any prior bounds. All remaining keyword arguments are passed
    onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        axis object to plot on

    dist: statistical distribution to plot
        This should have a `logpdf` method and a `rvs` method, operating on
        two-dimensional inputs

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).
    """
    nplot = kwargs.pop("nplot_2d", 1000)
    x = dist.rvs(nplot)
    y = x[:, 1]
    x = x[:, 0]
    return anesthetic_scatter_plot_2d(ax, x, y, *args, **kwargs)


def plot_1d(dist, axes=None, *args, **kwargs):
    """Create an array of 1D plots.

    Parameters
    ----------
    dist: statistical distribution to plot
        This should have a `logpdf` method and a `rvs` method, operating on
        one-dimensional inputs

    axes : plotting axes, optional
        Can be:

        * list(str) or str
        * :class:`pandas.Series` of :class:`matplotlib.axes.Axes`

        If a :class:`pandas.Series` is provided as an existing set of axes,
        then this is used for creating the plot. Otherwise, a new set of
        axes are created using the list or lists of strings.

        If not provided, then all parameters are plotted. This is intended
        for plotting a sliced array (e.g. `samples[['x0','x1]].plot_1d()`.

    Returns
    -------
    axes : :class:`pandas.Series` of :class:`matplotlib.axes.Axes`
        Pandas array of axes objects

    """
    if axes is None:
        axes = list(range(dist.dim))
    if not isinstance(axes, AxesSeries):
        fig, axes = make_1d_axes(axes)
    for i, ax in enumerate(axes):
        d = dist[i]
        pdf_plot_1d(ax, d, *args, **kwargs)
    return axes


def plot_2d(dist, axes=None, *args, **kwargs):
    """Create an array of 2D plots.

    To avoid interfering with y-axis sharing, one-dimensional plots are
    created on a separate axis, which is monkey-patched onto the argument
    ax as the attribute ax.twin.

    Parameters
    ----------
    dist : statistical distribution to plot
        This should have a `logpdf` method and a `rvs` method, operating on
        two-dimensional inputs

    axes : plotting axes, optional
        Can be:
            - list(str) if the x and y axes are the same
            - [list(str),list(str)] if the x and y axes are different
            - :class:`pandas.DataFrame` of :class:`matplotlib.axes.Axes`

        If a :class:`pandas.DataFrame` is provided as an existing set of
        axes, then this is used for creating the plot. Otherwise, a new set
        of axes are created using the list or lists of strings.

        If not provided, then all parameters are plotted. This is intended
        for plotting a sliced array (e.g. `samples[['x0','x1]].plot_2d()`.
        It is not advisible to plot an entire frame, as it is
        computationally expensive, and liable to run into linear algebra
        errors for degenerate derived parameters.

    diagonal_kwargs, lower_kwargs, upper_kwargs : dict, optional
        kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
        useful when there is a conflict of kwargs for different kinds of
        plots.  Note that any kwargs directly passed to plot_2d will
        overwrite any kwarg with the same key passed to <sub>_kwargs.
        Default: {}

    Returns
    -------
    axes : :class:`pandas.DataFrame` of :class:`matplotlib.axes.Axes`
        Pandas array of axes objects

    """
    if axes is None:
        axes = list(range(dist.dim))
    if not isinstance(axes, AxesDataFrame):
        fig, axes = make_2d_axes(axes)
    for y, row in axes.iterrows():
        for x, ax in row.items():
            if ax.position == "diagonal":
                pdf_plot_1d(ax.twin, dist[[x]], *args, **kwargs)
            elif ax.position == "lower":
                pdf_plot_2d(ax, dist[[x, y]], *args, **kwargs)
            elif ax.position == "upper":
                scatter_plot_2d(ax, dist[[x, y]], *args, **kwargs)
    return axes
