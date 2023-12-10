"""Extensions to scipy.stats functions."""
import numpy as np
import scipy.stats
from numpy.linalg import inv
from scipy.special import erf, logsumexp

from lsbi.utils import bisect, logdet


def choice(size, p):
    """Vectorised choice function."""
    cump = np.cumsum(p, axis=-1)
    u = np.random.rand(*size, *p.shape)
    return np.argmin(u > cump, axis=-1)


class multivariate_normal(object):
    """Vectorised multivariate normal distribution.

    This extends scipy.stats.multivariate_normal to allow for vectorisation across
    the distribution parameters. mean can be an array of shape (..., dim) and cov
    can be an array of shape (..., dim, dim) where ... represent arbitrary broadcastable
    shapes.

    Implemented with the same style as scipy.stats.multivariate_normal, except that
    results are not squeezed.

    Parameters
    ----------
    mean : array_like, shape (..., dim)
        Mean of each component.

    cov array_like, shape (..., dim, dim)
        Covariance matrix of each component.

    shape: tuple, optional, default=()
        Shape of the distribution. Useful for forcing a broadcast beyond that
        inferred by mean and cov shapes

    """

    def __init__(self, mean, cov, shape=()):
        self.mean = np.atleast_1d(mean)
        self.cov = np.atleast_2d(cov)
        self.shape = shape
        assert self.cov.shape[-2:] == (self.dim, self.dim)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            self.mean.shape[:-1], self.cov.shape[:-2], self._shape
        )

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self._shape = self.shape

    @property
    def dim(self):
        """Dimension of the distribution."""
        return self.mean.shape[-1]

    def logpdf(self, x):
        """Log of the probability density function."""
        x = np.array(x)
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        dx = x.reshape(*x.shape[:-1], *np.ones_like(self.shape), self.dim) - mean
        invcov = np.linalg.inv(self.cov)
        chi2 = np.einsum("...j,...jk,...k->...", dx, invcov, dx)
        norm = -logdet(2 * np.pi * self.cov) / 2
        return norm - chi2 / 2

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(size)
        x = np.random.randn(*size, *self.shape, self.dim)
        L = np.linalg.cholesky(self.cov)
        return self.mean + np.einsum("...jk,...k->...j", L, x)

    def predict(self, A, b=0):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ N(mu, Sigma)
        then:  Ax + b ~ N(A mu + b, A Sigma A^T)

        Parameters
        ----------
        A : array_like, shape (..., k, n)
            Linear transformation matrix.
        b : array_like, shape (..., k), optional
            Linear transformation vector.

        Returns
        -------
        predicted distribution: multivariate_normal
        shape (..., k)
        """
        mean = np.einsum("...qn,...n->...q", A, self.mean) + b
        cov = np.einsum("...qn,...nm,...pm->...qp", A, self.cov, A)
        return multivariate_normal(mean, cov, self.shape)

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        marginalised distribution: multivariate_normal
        """
        i = self._bar(indices)
        mean = self.mean[..., i]
        cov = self.cov[..., i, :][..., i]
        return multivariate_normal(mean, cov, self.shape)

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like
            Values to condition on.

        Returns
        -------
        conditional distribution: multivariate_normal
        """
        i = self._bar(indices)
        k = indices
        mean = self.mean[..., i] + np.einsum(
            "...ja,...ab,...b->...j",
            self.cov[..., i, :][..., :, k],
            inv(self.cov[..., k, :][..., :, k]),
            values - self.mean[..., k],
        )
        cov = self.cov[..., i, :][..., :, i] - np.einsum(
            "...ja,...ab,...bk->...jk",
            self.cov[..., i, :][..., :, k],
            inv(self.cov[..., k, :][..., :, k]),
            self.cov[..., k, :][..., :, i],
        )
        return multivariate_normal(mean, cov, self.shape)

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
        x : array_like, shape (..., d)
            if inverse: x is theta
            else: x is x
        inverse : bool, optional, default=False
            If True: compute the inverse transformation from physical to
            hypercube space.

        Returns
        -------
        transformed x or theta: array_like, shape (..., d)
        """
        x = np.array(x)
        L = np.linalg.cholesky(self.cov)
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        if inverse:
            invL = np.broadcast_to(inv(L), (*self.shape, self.dim, self.dim))
            y = np.einsum("...jk,...k->...j", invL, x - mean)
            return scipy.stats.norm.cdf(y)
        else:
            L = np.broadcast_to(L, (*self.shape, self.dim, self.dim))
            y = scipy.stats.norm.ppf(x)
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

    logA: array_like, shape (..., n,)
        Log of the mixing weights.
    """

    def __init__(self, logA, mean, cov, shape=()):
        self.logA = np.array(logA)
        super().__init__(mean, cov, shape)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            self.logA.shape, self.mean.shape[:-1], self.cov.shape[:-2], self._shape
        )

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self._shape = self.shape

    def logpdf(self, x):
        """Log of the probability density function."""
        logpdf = super().logpdf(x)
        if self.shape == ():
            return logpdf
        logA = self.logA - logsumexp(self.logA, axis=-1)[..., None]
        return logsumexp(logpdf + logA, axis=-1)

    def rvs(self, size=1):
        """Random variates."""
        if self.shape == ():
            return super().rvs(size)
        size = np.atleast_1d(np.array(size, dtype=int))
        p = np.exp(self.logA - logsumexp(self.logA, axis=-1)[..., None])
        p = np.broadcast_to(p, self.shape)
        i = choice(size, p)
        L = np.linalg.cholesky(self.cov)
        L = np.broadcast_to(L, (*self.shape, self.dim, self.dim))
        L = np.choose(i[..., None, None], np.moveaxis(L, -3, 0))
        mean = np.broadcast_to(self.mean, (*self.shape, self.dim))
        mean = np.choose(i[..., None], np.moveaxis(mean, -2, 0))
        x = np.random.randn(*size, *self.shape[:-1], self.dim)
        return mean + np.einsum("...ij,...j->...i", L, x)

    def predict(self, A, b=None):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ mixN(mu, Sigma, logA)
        then:  Ax + b ~ mixN(A mu + b, A Sigma A^T, logA)

        Parameters
        ----------
        A : array_like, shape (k, q, n)
            Linear transformation matrix.
        b : array_like, shape (k, q,), optional
            Linear transformation vector.

        Returns
        -------
        predicted distribution: multivariate_normal
        """
        dist = super().predict(A, b)
        return multivariate_normal(self.logA, dist.mean, dist.cov, dist.shape)

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        marginalised distribution: mixture_normal
        """
        dist = super().marginalise(indices)
        return mixture_normal(self.logA, dist.mean, dist.cov, self.shape)

    def condition(self, indices, values):
        """Condition on indices with values.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like
            Values to condition on.

        Returns
        -------
        conditional distribution: mixture_normal
        """
        dist = super().condition(indices, values[..., None, :])
        marginal = self.marginalise(self._bar(indices))
        marginal.mean = marginal.mean - values[..., None, :]
        logA = super(marginal.__class__, marginal).logpdf(np.zeros(marginal.dim))
        logA -= logsumexp(logA, axis=-1)[..., None]
        logA += self.logA
        return mixture_normal(logA, dist.mean, dist.cov, dist.shape)

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
            m = dist.mean[..., 0]
            c = dist.cov[..., 0, 0]
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
