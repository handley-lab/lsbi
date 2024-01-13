"""Extensions to scipy.stats functions."""
from copy import deepcopy

import numpy as np
import scipy.stats
from numpy.linalg import cholesky, inv
from scipy.special import erf, logsumexp

from lsbi.utils import bisect, choice, logdet


class multivariate_normal(object):
    """Vectorised multivariate normal distribution.

    This extends scipy.stats.multivariate_normal to allow for vectorisation across
    the distribution parameters mean and cov.

    Implemented with the same style as scipy.stats.multivariate_normal, except that
    results are not squeezed.

    mean and cov are lazily broadcasted to the same shape to improve performance.

    Parameters
    ----------
    mean : array_like, shape (..., dim)
        Mean of each component.

    cov array_like, shape (..., dim, dim)
        Covariance matrix of each component.

    shape: tuple, optional, default=()
        Shape of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    dim: int, optional, default=0
        Dimension of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    diagonal_cov: bool, optional, default=False
        If True, cov is interpreted as the diagonal of the covariance matrix.
    """

    def __init__(self, mean=0, cov=1, shape=(), dim=0, diagonal_cov=False):
        self.mean = mean
        self.cov = cov
        self._shape = shape
        self._dim = dim
        self.diagonal_cov = diagonal_cov
        if len(np.shape(self.cov)) < 2:
            self.diagonal_cov = True

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.mean)[:-1],
            np.shape(self.cov)[: -2 + self.diagonal_cov],
            self._shape,
        )

    @property
    def dim(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.mean)[-1:],
                *np.shape(self.cov)[-2 + self.diagonal_cov :],
                self._dim,
            ]
        )

    def logpdf(self, x):
        """Log of the probability density function.

        Parameters
        ----------
        x : array_like, shape (*size, dim)
            Points at which to evaluate the log of the probability density
            function.

        Returns
        -------
        logpdf : array_like, shape (*size, *shape)
            Log of the probability density function evaluated at x.
        """
        x = np.array(x)
        size = x.shape[:-1]
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        dx = x.reshape(*size, *np.ones_like(self.shape), self.dim) - mean
        if self.diagonal_cov:
            chi2 = (dx**2 / self.cov).sum(axis=-1)
            norm = -np.log(2 * np.pi * np.ones(self.dim) * self.cov).sum() / 2
        else:
            chi2 = np.einsum("...j,...jk,...k->...", dx, inv(self.cov), dx)
            norm = -logdet(2 * np.pi * self.cov) / 2
        return norm - chi2 / 2

    def rvs(self, size=()):
        """Draw random samples from the distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional, default=()
            Number of samples to draw.

        Returns
        -------
        rvs : ndarray, shape (*size, *shape, dim)
            Random samples from the distribution.
        """
        size = np.atleast_1d(size)
        x = np.random.randn(*size, *self.shape, self.dim)
        if self.diagonal_cov:
            return self.mean + np.sqrt(self.cov) * x
        else:
            return self.mean + np.einsum("...jk,...k->...j", cholesky(self.cov), x)

    def predict(self, A=1, b=0):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ N(mu, Sigma)
        then:  Ax + b ~ N(A mu + b, A Sigma A^T)

        Parameters
        ----------
        A : array_like, shape (..., k, dim)
            Linear transformation matrix.
        b : array_like, shape (..., k), optional
            Linear transformation vector.

        where self.shape is broadcastable to ...

        Returns
        -------
        multivariate_normal shape (..., k)
        """
        diagonal_cov = self.diagonal_cov
        if len(np.shape(A)) > 1:
            mean = np.einsum("...qn,...n->...q", A, np.ones(self.dim) * self.mean) + b
            if self.diagonal_cov:
                cov = np.einsum(
                    "...qn,...pn->...qp", A, A * np.atleast_1d(self.cov)[..., None, :]
                )
                diagonal_cov = False
            else:
                cov = np.einsum("...qn,...nm,...pm->...qp", A, self.cov, A)
        else:
            mean = A * self.mean + b
            if self.diagonal_cov:
                cov = A * self.cov * A
            else:
                cov = (
                    self.cov
                    * np.atleast_1d(A)[..., None]
                    * np.atleast_1d(A)[..., None, :]
                )
        dim = np.max([*np.shape(A)[-2:-1], *np.shape(b)[-1:], -1])
        if dim == -1:
            dim = self.dim
        return multivariate_normal(mean, cov, self.shape, dim, diagonal_cov)

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        multivariate_normal shape (*shape, dim - len(indices))
        """
        i = self._bar(indices)
        mean = (np.ones(self.dim) * self.mean)[..., i]

        if self.diagonal_cov:
            cov = (np.ones(self.dim) * self.cov)[..., i]
        else:
            cov = self.cov[..., i, :][..., i]

        return multivariate_normal(mean, cov, self.shape, sum(i), self.diagonal_cov)

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like shape (..., len(indices))
            Values to condition on.

        where where self.shape is broadcastable to ...

        Returns
        -------
        multivariate_normal shape (..., len(indices))
        """
        i = self._bar(indices)
        k = indices
        mean = (np.ones(self.dim) * self.mean)[..., i]

        if self.diagonal_cov:
            cov = (np.ones(self.dim) * self.cov)[..., i]
            shape = np.broadcast_shapes(self.shape, values.shape[:-1])
        else:
            mean = mean + np.einsum(
                "...ja,...ab,...b->...j",
                self.cov[..., i, :][..., :, k],
                inv(self.cov[..., k, :][..., :, k]),
                values - (np.ones(self.dim) * self.mean)[..., k],
            )
            cov = self.cov[..., i, :][..., :, i] - np.einsum(
                "...ja,...ab,...bk->...jk",
                self.cov[..., i, :][..., :, k],
                inv(self.cov[..., k, :][..., :, k]),
                self.cov[..., k, :][..., :, i],
            )
            shape = self.shape

        return multivariate_normal(mean, cov, shape, sum(i), self.diagonal_cov)

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
        x : array_like, shape (..., dim)
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
            if self.diagonal_cov:
                y = (x - mean) / np.sqrt(self.cov)
            else:
                y = np.einsum("...jk,...k->...j", inv(cholesky(self.cov)), x - mean)
            return scipy.stats.norm.cdf(y)
        else:
            y = scipy.stats.norm.ppf(x)
            if self.diagonal_cov:
                return mean + np.sqrt(self.cov) * y
            else:
                L = cholesky(self.cov)
                return mean + np.einsum("...jk,...k->...j", L, y)


class mixture_normal(multivariate_normal):
    """Mixture of multivariate normal distributions.

    Broadcastable multivariate mixture model.

    Parameters
    ----------
    mean : array_like, shape (..., n, dim)
        Mean of each component.

    cov: array_like, shape (..., n, dim, dim)
        Covariance matrix of each component.

    logA: array_like, shape (..., n)
        Log of the mixing weights.

    shape: tuple, optional, default=()
        Shape of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    dim: int, optional, default=0
        Dimension of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    diagonal_cov: bool, optional, default=False
        If True, cov is interpreted as the diagonal of the covariance matrix.
    """

    def __init__(self, logA=0, mean=0, cov=1, shape=(), dim=0, diagonal_cov=False):
        self.logA = logA
        super().__init__(
            mean=mean, cov=cov, shape=shape, dim=dim, diagonal_cov=diagonal_cov
        )

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.logA),
            np.shape(self.mean)[:-1],
            np.shape(self.cov)[: -2 + self.diagonal_cov],
            self._shape,
        )

    def logpdf(self, x):
        """Log of the probability density function.

        Parameters
        ----------
        x : array_like, shape (*size, dim)
            Points at which to evaluate the log of the probability density
            function.

        Returns
        -------
        logpdf : array_like, shape (*size, *shape[:-1])
            Log of the probability density function evaluated at x.
        """
        logpdf = super().logpdf(x)
        if self.shape == ():
            return logpdf
        logA = self.logA - logsumexp(self.logA, axis=-1)[..., None]
        return logsumexp(logpdf + logA, axis=-1)

    def rvs(self, size=()):
        """Draw random samples from the distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional, default=1

        Returns
        -------
        rvs : array_like, shape (*size, *shape[:-1], dim)
        """
        if self.shape == ():
            return super().rvs(size=size)
        size = np.atleast_1d(np.array(size, dtype=int))
        p = np.exp(self.logA - logsumexp(self.logA, axis=-1)[..., None])
        p = np.broadcast_to(p, self.shape)
        i = choice(size, p)
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        mean = np.choose(i[..., None], np.moveaxis(mean, -2, 0))
        x = np.random.randn(*size, *self.shape[:-1], self.dim)
        if self.diagonal_cov:
            L = np.sqrt(self.cov)
            L = np.broadcast_to(L, (*self.shape, self.dim))
            L = np.choose(i[..., None], np.moveaxis(L, -2, 0))
            return mean + L * x
        else:
            L = cholesky(self.cov)
            L = np.broadcast_to(L, (*self.shape, self.dim, self.dim))
            L = np.choose(i[..., None, None], np.moveaxis(L, -3, 0))
            return mean + np.einsum("...ij,...j->...i", L, x)

    def predict(self, A=1, b=0):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ mixN(mu, Sigma, logA)
        then:  Ax + b ~ mixN(A mu + b, A Sigma A^T, logA)

        Parameters
        ----------
        A : array_like, shape (..., k, dim)
            Linear transformation matrix.
        b : array_like, shape (..., k), optional
            Linear transformation vector.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        mixture_normal shape (..., k)
        """
        if len(np.shape(A)) > 1:
            A = np.expand_dims(A, axis=-3)
        if len(np.shape(b)) > 0:
            b = np.expand_dims(b, axis=-2)
        dist = super().predict(A, b)
        return mixture_normal(
            self.logA, dist.mean, dist.cov, dist.shape, dist.dim, dist.diagonal_cov
        )

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        mixture_normal shape (*shape, dim - len(indices))
        """
        dist = super().marginalise(indices)
        return mixture_normal(
            self.logA, dist.mean, dist.cov, self.shape, dist.dim, dist.diagonal_cov
        )

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like shape (..., len(indices))
            Values to condition on.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        mixture_normal shape (*shape, len(indices))
        """
        dist = super().condition(indices, values[..., None, :])
        logA = self.marginalise(self._bar(indices)).weights(values)
        return mixture_normal(
            logA, dist.mean, dist.cov, dist.shape, dist.dim, dist.diagonal_cov
        )

    def weights(self, values):
        """Compute the conditional weights of the mixture.

        Parameters
        ----------
        values : array_like shape (..., dim)
            Values to condition on.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        weights : array_like shape (*shape, n)
        """
        copy = deepcopy(self)
        copy.mean = copy.mean - values[..., None, :]
        logA = super(copy.__class__, copy).logpdf(np.zeros(copy.dim))
        logA -= logsumexp(logA, axis=-1)[..., None]
        logA += self.logA
        return logA

    def bijector(self, x, inverse=False):
        """Bijector between U([0, 1])^d and the distribution.

        - x in [0, 1]^d is the hypercube space.
        - theta in R^d is the physical space.

        Computes the transformation from x to theta or theta to x depending on
        the value of inverse.

        Parameters
        ----------
        x : array_like, shape (..., d)
            if inverse: x is theta
            else: x is x
        inverse : bool, optional, default=False
            If True: compute the inverse transformation from physical to
            hypercube space.

        where self.shape[:-1] is broadcastable to ...

        Returns
        -------
        transformed x or theta: array_like, shape (..., d)
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
            if dist.diagonal_cov:
                c = np.atleast_1d(dist.cov)[..., 0]
            else:
                c = np.atleast_2d(dist.cov)[..., 0, 0]
            A = np.exp(dist.logA - logsumexp(dist.logA, axis=-1)[..., None])
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
