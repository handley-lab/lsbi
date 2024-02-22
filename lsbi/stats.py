"""Extensions to scipy.stats functions."""

from copy import deepcopy

import numpy as np
import scipy.stats
from numpy.linalg import cholesky, inv
from scipy.special import erf, logsumexp

from lsbi.utils import bisect, logdet


class multivariate_normal(object):
    """Vectorised multivariate normal distribution.

    This extends scipy.stats.multivariate_normal to allow for vectorisation across
    the distribution parameters mean and cov.

    Implemented with the same style as scipy.stats.multivariate_normal, except that
    results are not squeezed.

    mean and cov are lazily broadcasted to the same shape to improve performance.

    Parameters
    ----------
    mean : array_like, shape `(..., dim)`
        Mean of each component.

    cov array_like, shape `(..., dim, dim)`if diagonal is False else shape `(..., dim)`
        Covariance matrix of each component.

    shape: tuple, optional, default=()
        Shape of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    dim: int, optional, default=0
        Dimension of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov dimensions

    diagonal: bool, optional, default=False
        If True, cov is interpreted as the diagonal of the covariance matrix.
    """

    def __init__(self, mean=0, cov=1, shape=(), dim=1, diagonal=False):
        self.mean = mean
        self.cov = cov
        self._shape = shape
        self._dim = dim
        self.diagonal = diagonal
        if len(np.shape(self.cov)) < 2:
            self.diagonal = True

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.mean)[:-1],
            np.shape(self.cov)[: -2 + self.diagonal],
            self._shape,
        )

    @property
    def dim(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.mean)[-1:],
                *np.shape(self.cov)[-2 + self.diagonal :],
                self._dim,
            ]
        )

    def logpdf(self, x, broadcast=False):
        """Log of the probability density function.

        Parameters
        ----------
        x : array_like, shape `(*size, dim)`
            Points at which to evaluate the log of the probability density
            function.
        broadcast : bool, optional, default=False
            If True, broadcast x across the shape of the distribution

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x.
            if not broadcast: shape `(*size, *self.shape)`
            else: shape broadcast of `(*size,) and `self.shape`
        """
        x = np.array(x)
        if broadcast:
            dx = x - self.mean
        else:
            size = x.shape[:-1]
            mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
            dx = x.reshape(*size, *np.ones_like(self.shape), self.dim) - mean
        if self.diagonal:
            chi2 = (dx**2 / self.cov).sum(axis=-1)
            norm = -np.log(2 * np.pi * np.ones(self.dim) * self.cov).sum(axis=-1) / 2
        else:
            chi2 = np.einsum("...j,...jk,...k->...", dx, inv(self.cov), dx)
            norm = -logdet(2 * np.pi * self.cov) / 2
        return norm - chi2 / 2

    def pdf(self, x, broadcast=False):
        """Probability density function.

        Parameters
        ----------
        x : array_like, shape `(*size, dim)`
            Points at which to evaluate the probability density function.
        broadcast : bool, optional, default=False
            If True, broadcast x across the distribution parameters.

        Returns
        -------
        pdf : array_like
            Probability density function evaluated at x.
            if not broadcast: shape `(*size, *self.shape)`
            else: shape broadcast of `(*size,) and `self.shape`
        """
        return np.exp(self.logpdf(x, broadcast=broadcast))

    def rvs(self, size=(), broadcast=False):
        """Draw random samples from the distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional, default=()
            Number of samples to draw.
        broadcast : bool, optional, default=False
            If True, broadcast x across the distribution parameters.

        Returns
        -------
        rvs : ndarray, shape `(*size, *shape, dim)`
            Random samples from the distribution.
        """
        size = np.atleast_1d(size)
        if broadcast:
            x = np.random.randn(*size, self.dim)
        else:
            x = np.random.randn(*size, *self.shape, self.dim)
        if self.diagonal:
            return self.mean + np.sqrt(self.cov) * x
        else:
            return self.mean + np.einsum("...jk,...k->...j", cholesky(self.cov), x)

    def predict(self, A=1, b=0, diagonal=False):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ N(μ, Σ)
        then:  Ax + b ~ N(A μ + b, A Σ A^T)

        Parameters
        ----------
        A : array_like, shape `(..., k, dim)`
            Linear transformation matrix.
        b : array_like, shape `(..., k)`, optional
            Linear transformation vector.
        diagonal : bool, optional, default=False
            If True, A is interpreted as the diagonal of the transformation matrix.

        where self.shape is broadcastable to ...

        Returns
        -------
        transformed distribution shape `(..., k)`
        """
        if len(np.shape(A)) < 2:
            diagonal = True
        dist = deepcopy(self)
        if diagonal:
            dist.mean = A * self.mean + b
            if self.diagonal:
                dist.cov = A * self.cov * A
            else:
                dist.cov = (
                    self.cov
                    * np.atleast_1d(A)[..., None]
                    * np.atleast_1d(A)[..., None, :]
                )
        else:
            dist.mean = (
                np.einsum("...qn,...n->...q", A, np.ones(self.dim) * self.mean) + b
            )
            if self.diagonal:
                dist.cov = np.einsum(
                    "...qn,...pn->...qp", A, A * np.atleast_1d(self.cov)[..., None, :]
                )
                dist.diagonal = False
            else:
                dist.cov = np.einsum("...qn,...nm,...pm->...qp", A, self.cov, A)
            dist._dim = np.shape(A)[-2]
        return dist

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        marginalised distribution, shape `(*shape, dim - len(indices))`
        """
        dist = deepcopy(self)
        i = self._bar(indices)
        dist.mean = (np.ones(self.dim) * self.mean)[..., i]

        if self.diagonal:
            dist.cov = (np.ones(self.dim) * self.cov)[..., i]
        else:
            dist.cov = self.cov[..., i, :][..., i]

        dist._dim = sum(i)
        return dist

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like shape `(..., len(indices))`
            Values to condition on.

        where where self.shape is broadcastable to ...

        Returns
        -------
        conditioned distribution shape `(..., len(indices))`
        """
        i = self._bar(indices)
        k = indices
        dist = deepcopy(self)
        dist.mean = (np.ones(self.dim) * self.mean)[..., i]

        if self.diagonal:
            dist.cov = (np.ones(self.dim) * self.cov)[..., i]
            dist._shape = np.broadcast_shapes(self.shape, values.shape[:-1])
        else:
            dist.mean = dist.mean + np.einsum(
                "...ja,...ab,...b->...j",
                self.cov[..., i, :][..., :, k],
                inv(self.cov[..., k, :][..., :, k]),
                values - (np.ones(self.dim) * self.mean)[..., k],
            )
            dist.cov = self.cov[..., i, :][..., :, i] - np.einsum(
                "...ja,...ab,...bk->...jk",
                self.cov[..., i, :][..., :, k],
                inv(self.cov[..., k, :][..., :, k]),
                self.cov[..., k, :][..., :, i],
            )
        dist._dim = sum(i)
        return dist

    def _bar(self, indices):
        """Return the indices not in the given indices."""
        k = np.ones(self.dim, dtype=bool)
        k[indices] = False
        return k

    def bijector(self, x, inverse=False):
        """Bijector between U([0, 1])^d and the distribution.

        - x in [0, 1]^d is the hypercube space.
        - theta in R^d is the physical space.

        Computes the transformation from x to theta or theta to x depending on
        the value of inverse.

        Parameters
        ----------
        x : array_like, shape `(..., dim)`
            if inverse: x is theta
            else: x is x
        inverse : bool, optional, default=False
            If True: compute the inverse transformation from physical to
            hypercube space.

        where self.shape is broadcastable to ...

        Returns
        -------
        transformed x or theta: array_like, shape (..., dim)
        """
        x = np.array(x)
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        if inverse:
            if self.diagonal:
                y = (x - mean) / np.sqrt(self.cov)
            else:
                y = np.einsum("...jk,...k->...j", inv(cholesky(self.cov)), x - mean)
            return scipy.stats.norm.cdf(y)
        else:
            y = scipy.stats.norm.ppf(x)
            if self.diagonal:
                return mean + np.sqrt(self.cov) * y
            else:
                L = cholesky(self.cov)
                return mean + np.einsum("...jk,...k->...j", L, y)

    def __getitem__(self, arg):
        """Access a subset of the distributions.

        Parameters
        ----------
        arg : int or slice or tuple of ints or tuples
            Indices to access.

        Returns
        -------
        dist : distribution
            A subset of the distribution

        Examples
        --------
        >>> dist = multivariate_normal(shape=(2, 3), dim=4)
        >>> dist.shape
        (2, 3)
        >>> dist.dim
        4
        >>> dist[0].shape
        (3,)
        >>> dist[0, 0].shape
        ()
        >>> dist[:, 0].shape
        (2,)
        """
        dist = deepcopy(self)
        dist.mean = np.broadcast_to(self.mean, (*self.shape, self.dim))[arg]
        if self.diagonal:
            dist.cov = np.broadcast_to(self.cov, (*self.shape, self.dim))[arg]
        else:
            dist.cov = np.broadcast_to(self.cov, (*self.shape, self.dim, self.dim))[arg]
        dist._shape = dist.mean.shape[:-1]
        dist._dim = dist.mean.shape[-1]
        return dist


class mixture_normal(multivariate_normal):
    """Mixture of multivariate normal distributions.

    Broadcastable multivariate mixture model.

    Parameters
    ----------
    mean : array_like, shape `(..., n, dim)`
        Mean of each component.

    cov: array_like, shape `(..., n, dim, dim)`
        Covariance matrix of each component.

    logw: array_like, shape `(..., n)`
        Log of the mixing weights.

    shape: tuple, optional, default=()
        Shape of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    dim: int, optional, default=0
        Dimension of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    diagonal: bool, optional, default=False
        If True, cov is interpreted as the diagonal of the covariance matrix.
    """

    def __init__(self, logw=0, mean=0, cov=1, shape=(), dim=1, diagonal=False):
        self.logw = logw
        super().__init__(mean, cov, shape, dim, diagonal)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(np.shape(self.logw), super().shape)

    @property
    def k(self):
        """Number of components."""
        if self.shape == ():
            return 1
        return self.shape[-1]

    def logpdf(self, x, broadcast=False, joint=False):
        """Log of the probability density function.

        Parameters
        ----------
        x : array_like, shape `(*size, dim)`
            Points at which to evaluate the log of the probability density
            function.

        broadcast : bool, optional, default=False
            If True, broadcast x across the distribution parameters.

        joint : bool, optional, default=False
            If True, return the joint logpdf of the mixture P(x, N)

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x.
            if not broadcast and not joint: shape `(*size, *shape[:-1])`
            elif not broadcast and joint: shape `(*size, *shape)`
            elif not joint: shape the broadcast of `(*size,) and `shape[:-1]`
            else: shape the broadcast of `(*size,) and `shape`
        """
        if broadcast:
            x = np.expand_dims(x, -2)
        logpdf = super().logpdf(x, broadcast=broadcast)
        if self.shape == ():
            return logpdf
        logw = np.broadcast_to(self.logw, self.shape).copy()
        logw = logw - logsumexp(logw, axis=-1, keepdims=True)
        if joint:
            return logpdf + logw
        return logsumexp(logpdf + logw, axis=-1)

    def pdf(self, x, broadcast=False, joint=False):
        """Probability density function.

        Parameters
        ----------
        x : array_like, shape `(*size, dim)`
            Points at which to evaluate the probability density function.

        broadcast : bool, optional, default=False
            If True, broadcast x across the distribution parameters.

        joint : bool, optional, default=False
            If True, return the joint pdf of the mixture P(x, N)

        Returns
        -------
        pdf :
            Probability density function evaluated at x.
            if not broadcast and not joint: shape `(*size, *shape[:-1])`
            elif not broadcast and joint: shape `(*size, *shape)`
            elif not joint: shape the broadcast of `(*size,) and `shape[:-1]`
            else: shape the broadcast of `(*size,) and `shape`
        """
        return np.exp(self.logpdf(x, broadcast=broadcast, joint=joint))

    def rvs(self, size=(), broadcast=False):
        """Draw random samples from the distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional, default=1
            Number of samples to draw.
        broadcast : bool, optional, default=False
            If True, broadcast x across the distribution parameters.

        Returns
        -------
        rvs : array_like, shape `(*size, *shape[:-1], dim)`
        """
        if self.shape == ():
            return super().rvs(size=size, broadcast=broadcast)
        size = np.atleast_1d(size)
        logw = np.broadcast_to(self.logw, self.shape).copy()
        logw = logw - logsumexp(logw, axis=-1, keepdims=True)
        p = np.exp(logw)
        cump = np.cumsum(p, axis=-1)
        if broadcast:
            u = np.random.rand(*size).reshape(-1, *p.shape[:-1])
        else:
            u = np.random.rand(np.prod(size, dtype=int), *p.shape[:-1])
        i = np.argmax(np.array(u)[..., None] < cump, axis=-1)
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        mean = np.take_along_axis(np.moveaxis(mean, -2, 0), i[..., None], axis=0)
        if broadcast:
            x = np.random.randn(*size, self.dim)
        else:
            x = np.random.randn(np.prod(size, dtype=int), *self.shape[:-1], self.dim)
        if self.diagonal:
            L = np.sqrt(self.cov)
            L = np.broadcast_to(L, (*self.shape, self.dim))
            L = np.take_along_axis(np.moveaxis(L, -2, 0), i[..., None], axis=0)
            rvs = mean + L * x
        else:
            L = cholesky(self.cov)
            L = np.broadcast_to(L, (*self.shape, self.dim, self.dim))
            L = np.take_along_axis(np.moveaxis(L, -3, 0), i[..., None, None], axis=0)
            rvs = mean + np.einsum("...ij,...j->...i", L, x)
        if broadcast:
            return rvs.reshape(*size, self.dim)
        else:
            return rvs.reshape(*size, *self.shape[:-1], self.dim)

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like shape `(..., len(indices))`
            Values to condition on.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        conditioned distribution, shape `(*shape, len(indices))`
        """
        dist = super().condition(indices, np.expand_dims(values, -2))
        dist.__class__ = mixture_normal
        marg = self.marginalise(self._bar(indices))
        dist.logw = marg.logpdf(values, broadcast=True, joint=True)
        dist.logw = dist.logw - logsumexp(dist.logw, axis=-1, keepdims=True)
        return dist

    def bijector(self, x, inverse=False):
        """Bijector between U([0, 1])^d and the distribution.

        - x in [0, 1]^d is the hypercube space.
        - theta in R^d is the physical space.

        Computes the transformation from x to theta or theta to x depending on
        the value of inverse.

        Parameters
        ----------
        x : array_like, shape `(..., d)`
            if inverse: x is theta
            else: x is x
        inverse : bool, optional, default=False
            If True: compute the inverse transformation from physical to
            hypercube space.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        transformed x or theta: array_like, shape `(..., d)`
        """
        x = np.array(x)
        theta = np.empty(np.broadcast_shapes(x.shape, self.shape[:-1] + (self.dim,)))

        if inverse:
            theta[:] = x
            x = np.empty(np.broadcast_shapes(x.shape, self.shape[:-1] + (self.dim,)))

        for i in range(self.dim):
            dist = self.marginalise(np.s_[i + 1 :]).condition(
                np.s_[:-1], theta[..., :i]
            )
            m = np.atleast_1d(dist.mean)[..., 0]
            if dist.diagonal:
                c = np.atleast_1d(dist.cov)[..., 0]
            else:
                c = np.atleast_2d(dist.cov)[..., 0, 0]
            A = np.exp(dist.logw - logsumexp(dist.logw, axis=-1)[..., None])
            m = np.broadcast_to(m, dist.shape)

            def f(t):
                return (A * 0.5 * (1 + erf((t[..., None] - m) / np.sqrt(2 * c)))).sum(
                    axis=-1
                ) - y

            if inverse:
                y = 0
                x[..., i] = f(theta[..., i])
            else:
                y = x[..., i]
                a = (m - 10 * np.sqrt(c)).min(axis=-1)
                b = (m + 10 * np.sqrt(c)).max(axis=-1)
                theta[..., i] = bisect(f, a, b)
        if inverse:
            return x
        else:
            return theta

    def __getitem__(self, arg):  # noqa: D105
        dist = super().__getitem__(arg)
        dist.__class__ = mixture_normal
        dist.logw = np.broadcast_to(self.logw, self.shape)[arg]
        return dist


def dkl(p, q, n=0):
    """Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p : lsbi.stats.multivariate_normal
    q : lsbi.stats.multivariate_normal
    n : int, optional, default=0
        Number of samples to mcmc estimate the divergence.

    Returns
    -------
    dkl : array_like
        Kullback-Leibler divergence between p and q.
    """
    shape = np.broadcast_shapes(p.shape, q.shape)
    if n:
        x = p.rvs(size=(n, *shape), broadcast=True)
        return (p.logpdf(x, broadcast=True) - q.logpdf(x, broadcast=True)).mean(axis=0)
    dkl = -p.dim * np.ones(shape)
    dkl = dkl + logdet(q.cov * np.ones(q.dim), q.diagonal)
    dkl = dkl - logdet(p.cov * np.ones(p.dim), p.diagonal)
    pq = (p.mean - q.mean) * np.ones(p.dim)
    if q.diagonal:
        dkl = dkl + (pq**2 / q.cov).sum(axis=-1)
        if p.diagonal:
            dkl = dkl + (p.cov / q.cov * np.ones(q.dim)).sum(axis=-1)
        else:
            dkl = dkl + (np.diagonal(p.cov, 0, -2, -1) / q.cov).sum(axis=-1)
    else:
        invq = inv(q.cov)
        dkl = dkl + np.einsum("...i,...ij,...j->...", pq, invq, pq)
        if p.diagonal:
            dkl = dkl + (p.cov * np.diagonal(invq, 0, -2, -1)).sum(axis=-1)
        else:
            dkl = dkl + np.einsum("...ij,...ji->...", invq, p.cov)

    return dkl / 2
