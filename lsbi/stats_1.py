"""Extensions to scipy.stats functions."""
import numpy as np
import scipy.stats
from numpy.linalg import inv
from scipy.special import erf, logsumexp
from scipy.stats._multivariate import multivariate_normal_frozen

from lsbi.utils import bisect, logdet


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

    def _flatten(self, x, *args):
        """Flatten the distribution parameters."""
        return np.broadcast_to(x, (*self.shape, *args)).reshape(-1, *args)

    def logpdf(self, x):
        """Log of the probability density function."""
        dx = x.reshape(-1, 1, self.dim) - self._flatten(self.mean, self.dim)
        invcov = self._flatten(np.linalg.inv(self.cov), self.dim, self.dim)
        chi2 = np.einsum("xaj,ajk,xak->xa", dx, invcov, dx)
        norm = -logdet(2 * np.pi * self.cov) / 2
        logpdf = norm - chi2.reshape((*x.shape[:-1], *self.shape)) / 2
        return logpdf

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(np.array(size, dtype=int))
        x = np.random.randn(np.prod(size), np.prod(self.shape, dtype=int), self.dim)
        L = self._flatten(np.linalg.cholesky(self.cov), self.dim, self.dim)
        t = self._flatten(self.mean, self.dim) + np.einsum("ajk,xak->xaj", L, x)
        return t.reshape(*size, *self.shape, self.dim)

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        marginalised distribution: multimultivariate_normal
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
        conditional distribution: multimultivariate_normal
        """
        i = self._bar(indices)
        k = indices
        kdim = len(k)
        idim = self.dim - kdim
        old_shape = self.shape
        self.shape = np.broadcast_shapes(values.shape[:-1], self.shape)
        mean = self._flatten(self.mean[..., i], idim) + np.einsum(
            "ija,iab,ib->ij",
            self._flatten(self.cov[..., i, :][..., :, k], idim, kdim),
            self._flatten(inv(self.cov[..., k, :][..., :, k]), kdim, kdim),
            self._flatten(values, kdim) - self._flatten(self.mean[..., k], kdim),
        )
        cov = self._flatten(self.cov[..., i, :][..., :, i], idim, idim) - np.einsum(
            "ija,iab,ibk->ijk",
            self._flatten(self.cov[..., i, :][..., :, k], idim, kdim),
            self._flatten(inv(self.cov[..., k, :][..., :, k]), kdim, kdim),
            self._flatten(self.cov[..., k, :][..., :, i], kdim, idim),
        )
        mean = mean.reshape(*self.shape, idim)
        cov = cov.reshape(*self.shape, idim, idim)
        self.shape = old_shape
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
        L = np.linalg.cholesky(self.cov)
        old_shape = self.shape
        self.shape = np.broadcast_shapes(x.shape[:-1], self.shape)
        if inverse:
            invL = inv(L)
            y = np.einsum(
                "ajk,ak->aj",
                self._flatten(invL, self.dim, self.dim),
                self._flatten(x, self.dim) - self._flatten(self.mean, self.dim),
            )
            y = y.reshape(*self.shape, self.dim)
            self.shape = old_shape
            return scipy.stats.norm.cdf(y)
        else:
            y = scipy.stats.norm.ppf(x)
            z = self._flatten(self.mean, self.dim) + np.einsum(
                "ajk,ak->aj",
                self._flatten(L, self.dim, self.dim),
                self._flatten(y, self.dim),
            )
            z = z.reshape(*self.shape, self.dim)
            self.shape = old_shape
            return z

    def predict(self, A, b=None):
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
        predicted distribution: mixture_multivariate_normal
        shape (..., k)
        """
        k = A.shape[-2]
        old_shape = self.shape
        if b is None:
            b = np.zeros(A.shape[:-1])
        self.shape = np.broadcast_shapes(self.shape, A.shape[:-2], b.shape[:-1])
        A = self._flatten(A, k, self.dim)
        b = self._flatten(b, k)
        mean = np.einsum("kqn,kn->kq", A, self._flatten(self.mean, self.dim)) + b
        cov = np.einsum(
            "kqn,knm,kpm->kqp", A, self._flatten(self.cov, self.dim, self.dim), A
        )
        mean = mean.reshape(*self.shape, k)
        cov = cov.reshape(*self.shape, k, k)
        ans = multivariate_normal(mean, cov, self.shape)
        self.shape = old_shape
        return ans


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
        logA = self.logA - scipy.special.logsumexp(self.logA)
        return scipy.special.logsumexp(logpdf + logA, axis=-1)

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(size)
        p = np.exp(self.logA - self.logA.max())
        p /= p.sum()
        i = np.random.choice(len(p), size, p=p)
        x = np.random.randn(*size, self.means.shape[-1])
        choleskys = np.linalg.cholesky(self.covs)
        return np.squeeze(self.means[i, ..., None] + choleskys[i] @ x[..., None])

    def marginalise(self, indices):
        """Marginalise over indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.

        Returns
        -------
        marginalised distribution: mixture_multivariate_normal
        """
        i = self._bar(indices)
        means = self.means[:, i]
        covs = self.covs[:, i][:, :, i]
        logA = self.logA
        return mixture_multivariate_normal(means, covs, logA)

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
        conditional distribution: mixture_multivariate_normal
        """
        i = self._bar(indices)
        k = indices
        marginal = self.marginalise(i)

        means = self.means[:, i] + np.einsum(
            "ija,iab,ib->ij",
            self.covs[:, i][:, :, k],
            inv(self.covs[:, k][:, :, k]),
            (values - self.means[:, k]),
        )
        covs = self.covs[:, i][:, :, i] - np.einsum(
            "ija,iab,ibk->ijk",
            self.covs[:, i][:, :, k],
            inv(self.covs[:, k][:, :, k]),
            self.covs[:, k][:, :, i],
        )
        logA = (
            marginal.logpdf(values, reduce=False) + self.logA - marginal.logpdf(values)
        )
        return mixture_multivariate_normal(means, covs, logA)

    def _bar(self, indices):
        """Return the indices not in the given indices."""
        k = np.ones(self.means.shape[-1], dtype=bool)
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
        theta = np.empty_like(x)
        if inverse:
            theta[:] = x
            x = np.empty_like(x)

        for i in range(x.shape[-1]):
            m = self.means[..., :, i] + np.einsum(
                "ia,iab,...ib->...i",
                self.covs[:, i, :i],
                inv(self.covs[:, :i, :i]),
                theta[..., None, :i] - self.means[:, :i],
            )
            c = self.covs[:, i, i] - np.einsum(
                "ia,iab,ib->i",
                self.covs[:, i, :i],
                inv(self.covs[:, :i, :i]),
                self.covs[:, i, :i],
            )
            dist = mixture_multivariate_normal(
                self.means[:, :i], self.covs[:, :i, :i], self.logA
            )
            logA = (
                self.logA
                + dist.logpdf(theta[..., :i], reduce=False, keepdims=True)
                - dist.logpdf(theta[..., :i], keepdims=True)[..., None]
            )
            A = np.exp(logA - logsumexp(logA, axis=-1)[..., None])

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

    def _process_quantiles(self, x, dim):
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis, np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

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
        predicted distribution: mixture_multivariate_normal
        """
        if b is None:
            b = np.zeros(A.shape[:-1])
        means = np.einsum("kqn,kn->kq", A, self.means) + b
        covs = np.einsum("kqn,knm,kpm->kqp", A, self.covs, A)
        logA = self.logA
        return mixture_multivariate_normal(means, covs, logA)
