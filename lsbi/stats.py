"""Extensions to scipy.stats functions."""
import numpy as np
import scipy.stats
from scipy.stats._multivariate import multivariate_normal_frozen
from numpy.linalg import inv


class multivariate_normal(multivariate_normal_frozen):  # noqa: D101
    def marginalise(self, indices):
        """Marginalise out the given indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.
        """
        i = self._bar(indices)
        mean = self.mean[i]
        cov = self.cov[i][:, i]
        return multivariate_normal(mean, cov)

    def condition(self, indices, values):
        """Condition over the given indices.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like
            Values to condition on.
        """
        i = self._bar(indices)
        k = indices
        mean = (self.mean[i] + self.cov[i][:, k] @
                inv(self.cov[k][:, k]) @ (values - self.mean[k]))
        cov = (self.cov[i][:, i] - self.cov[i][:, k] @
               inv(self.cov[k][:, k]) @ self.cov[k][:, i])
        return multivariate_normal(mean, cov)

    def _bar(self, indices):
        """Return the indices not in the given indices."""
        k = np.ones(len(self.mean), dtype=bool)
        k[indices] = False
        return k


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

    def logpdf(self, x, reduce=True):
        """Log of the probability density function."""
        process_quantiles = scipy.stats.multivariate_normal._process_quantiles
        x = process_quantiles(x, self.means.shape[-1])
        dx = self.means - x[..., None, :]
        invcovs = np.linalg.inv(self.covs)
        chi2 = np.einsum('...ij,ijk,...ik->...i', dx, invcovs, dx)
        norm = -np.linalg.slogdet(2*np.pi*self.covs)[1]/2
        logpdfs = norm - chi2/2
        if not reduce:
            return np.squeeze(logpdfs)
        logA = self.logA - scipy.special.logsumexp(self.logA)
        return np.squeeze(scipy.special.logsumexp(logpdfs+logA, axis=-1))

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(size)
        p = np.exp(self.logA-self.logA.max())
        p /= p.sum()
        i = np.random.choice(len(p), size, p=p)
        x = np.random.randn(*size, self.means.shape[-1])
        choleskys = np.linalg.cholesky(self.covs)
        return np.squeeze(self.means[i, ..., None]
                          + choleskys[i] @ x[..., None])

    def marginalise(self, indices):
        """Marginalise out the given indices.

        Parameters
        ----------
        indices : array_like
            Indices to marginalise.
        """
        i = self._bar(indices)
        means = self.means[:, i]
        covs = self.covs[:, i][:, :, i]
        logA = self.logA
        return mixture_multivariate_normal(means, covs, logA)

    def condition(self, indices, values):
        """Condition over the given indices.

        Parameters
        ----------
        indices : array_like
            Indices to condition over.
        values : array_like
            Values to condition on.
        """
        i = self._bar(indices)
        k = indices

        means = (self.means[:, i] +
                 np.einsum('ija,iab,ib->ij', self.covs[:, i][:, :, k],
                           inv(self.covs[:, k][:, :, k]),
                           (values - self.means[:, k])))
        covs = (self.covs[:, i][:, :, i] -
                np.einsum('ija,iab,ibk->ijk', self.covs[:, i][:, :, k],
                          inv(self.covs[:, k][:, :, k]),
                          self.covs[:, k][:, :, i]))
        logA = self.logA
        return mixture_multivariate_normal(means, covs, logA)

    def _bar(self, indices):
        """Return the indices not in the given indices."""
        k = np.ones(self.means.shape[-1], dtype=bool)
        k[indices] = False
        return k
