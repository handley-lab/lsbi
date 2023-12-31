"""Extensions to scipy.stats functions."""
import numpy as np
import scipy.stats
from numpy.linalg import inv
from scipy.special import erf, logsumexp
from scipy.stats._multivariate import multivariate_normal_frozen

from lsbi.utils import bisect, logdet


class multivariate_normal(multivariate_normal_frozen):  # noqa: D101
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
        mean = self.mean[i]
        cov = self.cov[i][:, i]
        return multivariate_normal(mean, cov)

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
        mean = self.mean[i] + self.cov[i][:, k] @ inv(self.cov[k][:, k]) @ (
            values - self.mean[k]
        )
        cov = (
            self.cov[i][:, i]
            - self.cov[i][:, k] @ inv(self.cov[k][:, k]) @ self.cov[k][:, i]
        )
        return multivariate_normal(mean, cov)

    def _bar(self, indices):
        """Return the indices not in the given indices."""
        k = np.ones(len(self.mean), dtype=bool)
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
        if inverse:
            Linv = inv(L)
            y = np.einsum("ij,...j->...i", Linv, x - self.mean)
            return scipy.stats.norm.cdf(y)
        else:
            y = scipy.stats.norm.ppf(x)
            return self.mean + np.einsum("ij,...j->...i", L, y)

    def predict(self, A, b=None):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ N(mu, Sigma)
        then:  Ax + b ~ N(A mu + b, A Sigma A^T)

        Parameters
        ----------
        A : array_like, shape (q, n)
            Linear transformation matrix.
        b : array_like, shape (q,), optional
            Linear transformation vector.

        Returns
        -------
        predicted distribution: multivariate_normal
        """
        if b is None:
            b = np.zeros(A.shape[0])
        mean = A @ self.mean + b
        cov = A @ self.cov @ A.T
        return multivariate_normal(mean, cov, allow_singular=True)


class multimultivariate_normal(object):
    """Multivariate normal distribution with multiple means and covariances.

    Implemented with the same style as scipy.stats.multivariate_normal

    Parameters
    ----------
    means : array_like, shape (n_components, n_features)
        Mean of each component.

    covs: array_like, shape (n_components, n_features, n_features)
        Covariance matrix of each component.

    """

    def __init__(self, means, covs):
        self.means = np.array([np.atleast_1d(m) for m in means])
        self.covs = np.array([np.atleast_2d(c) for c in covs])

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

    def logpdf(self, x):
        """Log of the probability density function."""
        x = self._process_quantiles(x, self.means.shape[-1])
        dx = self.means - x[..., :, :]
        invcovs = np.linalg.inv(self.covs)
        chi2 = np.einsum("...ij,ijk,...ik->...i", dx, invcovs, dx)
        norm = -logdet(2 * np.pi * self.covs) / 2
        logpdf = norm - chi2 / 2
        return np.squeeze(logpdf)

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(size)
        x = np.random.randn(*size, *self.means.shape)
        choleskys = np.linalg.cholesky(self.covs)
        return np.squeeze(self.means + np.einsum("ijk,...ik->...ij", choleskys, x))

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
        means = self.means[:, i]
        covs = self.covs[:, i][:, :, i]
        return multimultivariate_normal(means, covs)

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
        values = values.reshape(self.means[:, k].shape)
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
        return multimultivariate_normal(means, covs)

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
        Ls = np.linalg.cholesky(self.covs)
        if inverse:
            Linvs = inv(Ls)
            y = np.einsum("ijk,...ik->...ij", Linvs, x - self.means)
            return scipy.stats.norm.cdf(y)
        else:
            y = scipy.stats.norm.ppf(x)
            return self.means + np.einsum("ijk,...ik->...ij", Ls, y)

    def predict(self, A, b=None):
        """Predict the mean and covariance of a linear transformation.

        if:         x ~ N(mu, Sigma)
        then:  Ax + b ~ N(A mu + b, A Sigma A^T)

        Parameters
        ----------
        A : array_like, shape (k, q, n)
            Linear transformation matrix.
        b : array_like, shape (k, q), optional
            Linear transformation vector.

        Returns
        -------
        predicted distribution: mixture_multivariate_normal
        """
        if b is None:
            b = np.zeros(A.shape[:-1])
        means = np.einsum("kqn,kn->kq", A, self.means) + b
        covs = np.einsum("kpn,knm,kqm->kpq", A, self.covs, A)
        return multimultivariate_normal(means, covs)


class mixture_multivariate_normal(object):
    """Mixture of multivariate normal distributions.

    Implemented with the same style as scipy.stats.multivariate_normal

    Parameters
    ----------
    means : array_like, shape (n_components, n_features)
        Mean of each component.

    covs: array_like, shape (n_components, n_features, n_features)
        Covariance matrix of each component.

    logA: array_like, shape (n_components,)
        Log of the mixing weights.
    """

    def __init__(self, means, covs, logA):
        self.means = np.array([np.atleast_1d(m) for m in means])
        self.covs = np.array([np.atleast_2d(c) for c in covs])
        self.logA = np.atleast_1d(logA)

    def logpdf(self, x, reduce=True, keepdims=False):
        """Log of the probability density function."""
        x = self._process_quantiles(x, self.means.shape[-1])
        dx = self.means - x[..., None, :]
        invcovs = np.linalg.inv(self.covs)
        chi2 = np.einsum("...ij,ijk,...ik->...i", dx, invcovs, dx)
        norm = -logdet(2 * np.pi * self.covs) / 2
        logpdf = norm - chi2 / 2
        if reduce:
            logA = self.logA - scipy.special.logsumexp(self.logA)
            logpdf = np.squeeze(scipy.special.logsumexp(logpdf + logA, axis=-1))
        if not keepdims:
            logpdf = np.squeeze(logpdf)
        return logpdf

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
