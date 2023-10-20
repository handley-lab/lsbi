"""Extensions to scipy.stats functions."""
import numpy as np
import scipy.stats


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
        self.choleskys = np.linalg.cholesky(self.covs)
        self.invcholeskys = np.linalg.inv(self.choleskys)

    def logpdf(self, x):
        """Log of the probability density function."""
        x = self._process_quantiles(x, self.means.shape[-1])
        dx = self.means - x[..., None, :]
        chi2 = np.einsum('...ij,ijk,...ik->...i', dx, self.invcholeskys, dx)
        norm = -np.linalg.slogdet(2*np.pi*self.covs)[1]/2
        logA = self.logA - scipy.special.logsumexp(self.logA)
        return np.squeeze(scipy.special.logsumexp(norm - chi2 + logA, axis=-1))

    def rvs(self, size=1):
        """Random variates."""
        size = np.atleast_1d(size)
        p = np.exp(self.logA-self.logA.max())
        p /= p.sum()
        i = np.random.choice(len(p), size, p=p)
        x = np.random.randn(*size, self.means.shape[-1])
        return np.squeeze(self.means[i, ..., None]
                          + self.choleskys[i] @ x[..., None])

    def _process_quantiles(self, x, dim):
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x
