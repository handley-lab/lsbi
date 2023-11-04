"""Gaussian models for linear Bayesian inference."""
import numpy as np
from functools import cached_property
from scipy.stats import multivariate_normal
from lsbi.stats import mixture_multivariate_normal
from numpy.linalg import solve, inv, slogdet


def logdet(A):
    """log(abs(det(A)))."""
    return slogdet(A)[1]


class LinearModel(object):
    """A linear model.

    D|theta ~ N(m + M theta, C)
    theta ~ N(mu, Sigma)

    Defined by:
    - Parameters: theta (n,)
    - Data: D (d,)
    - Prior mean: mu (n,)
    - Prior covariance: Sigma (n, n)
    - Data mean: m (d,)
    - Data covariance: C (d, d)

    Parameters
    ----------
    M : array_like, optional
        Model matrix, defaults to identity matrix
    m : array_like, optional
        Data mean, defaults to zero vector
    C : array_like, optional
        Data covariance, defaults to identity matrix
    mu : array_like, optional
        Prior mean, defaults to zero vector
    Sigma : array_like, optional
        Prior covariance, defaults to identity matrix

    the overall shape is attempted to be inferred from the input parameters.
    """

    def __init__(self, *args, **kwargs):
        # Rationalise input arguments
        M = self._atleast_2d(kwargs.pop('M', None))
        m = self._atleast_1d(kwargs.pop('m', None))
        C = self._atleast_2d(kwargs.pop('C', None))
        mu = self._atleast_1d(kwargs.pop('mu', None))
        Sigma = self._atleast_2d(kwargs.pop('Sigma', None))

        # Determine dimensions
        n = max([M.shape[1], mu.shape[0], Sigma.shape[0], Sigma.shape[1]])
        d = max([M.shape[0], m.shape[0], C.shape[0], C.shape[1]])
        if not n:
            raise ValueError('Unable to determine number of parameters n')
        if not d:
            raise ValueError('Unable to determine data dimensions d')

        # Set defaults if no argument was passed
        M = M if M.size else np.eye(d, n)
        m = m if m.size else np.zeros(d)
        C = C if C.size else np.eye(d)
        mu = mu if mu.size else np.zeros(n)
        Sigma = Sigma if Sigma.size else np.eye(n)

        # Broadcast to correct shape
        self.M = np.broadcast_to(M, (d, n))
        self.m = np.broadcast_to(m, (d,))
        self.C = np.broadcast_to(C, (d, d))
        self.mu = np.broadcast_to(mu, (n,))
        self.Sigma = np.broadcast_to(Sigma, (n, n))

    @classmethod
    def from_joint(cls, mean, cov, n):
        """Construct model from joint distribution."""
        mu = mean[-n:]
        Sigma = cov[-n:, -n:]
        M = solve(Sigma, cov[-n:, :-n]).T
        m = mean[:-n] - M @ mu
        C = cov[:-n, :-n] - M @ Sigma @ M.T

        return cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    @property
    def n(self):
        """Dimensionality of parameter space len(theta)."""
        return self.M.shape[1]

    @property
    def d(self):
        """Dimensionality of data space len(D)."""
        return self.M.shape[0]

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D ~ N( m + M theta, C )

        Parameters
        ----------
        theta : array_like, shape (n,)
        """
        return multivariate_normal(self.m + self.M @ theta, self.C)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta ~ N( mu, Sigma )
        """
        return multivariate_normal(self.mu, self.Sigma)

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta ~ N( mu + Sigma M'C^{-1}(D-m), Sigma - Sigma M' C^{-1} M Sigma )

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        Sigma = inv(self.invSigma + self.M.T @ self.invC @ self.M)
        D0 = self.m + self.M @ self.mu
        mu = self.mu + Sigma @ self.M.T @ self.invC @ (D-D0)
        return multivariate_normal(mu, Sigma)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D ~ N( m + M mu, C + M Sigma M' )
        """
        return multivariate_normal(self.m + self.M @ self.mu,
                                   self.C + self.M @ self.Sigma @ self.M.T)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta]    ( [   mu   ] , [   Sigma M'      Sigma ] )
        """
        evidence = self.evidence()
        prior = self.prior()
        mu = np.concatenate([evidence.mean, prior.mean])
        Sigma = np.block([[evidence.cov, self.M @ self.Sigma],
                          [self.Sigma @ self.M.T, prior.cov]])
        return multivariate_normal(mu, Sigma)

    def DKL(self, D):
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence.

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        cov_p = self.posterior(D).cov
        cov_q = self.prior().cov
        mu_p = self.posterior(D).mean
        mu_q = self.prior().mean
        return (- logdet(cov_p) + logdet(cov_q)
                + np.trace(inv(cov_q) @ cov_p - 1)
                + (mu_q - mu_p) @ inv(cov_q) @ (mu_q - mu_p))/2

    def reduce(self, D):
        """Reduce the model to a Gaussian in the parameters.

        Parameters
        ----------
        D : array_like, shape (d,)

        Returns
        -------
        ReducedLinearModel
        """
        Sigma_L = inv(self.M.T @ self.invC @ self.M)
        mu_L = Sigma_L @ self.M.T @ self.invC @ (D-self.m)
        logLmax = (- logdet(2 * np.pi * self.C)/2 - (D-self.m) @ self.invC @
                   (self.C - self.M @ Sigma_L @ self.M.T) @ self.invC @
                   (D-self.m)/2)
        return ReducedLinearModel(mu_L=mu_L, Sigma_L=Sigma_L, logLmax=logLmax,
                                  mu_pi=self.prior().mean,
                                  Sigma_pi=self.prior().cov)

    @cached_property
    def invSigma(self):
        """Inverse of prior covariance."""
        return inv(self.Sigma)

    @cached_property
    def invC(self):
        """Inverse of data covariance."""
        return inv(self.C)

    def _atleast_2d(self, x):
        if x is None:
            return np.zeros(shape=(0, 0))
        return np.atleast_2d(x)

    def _atleast_1d(self, x):
        if x is None:
            return np.zeros(shape=(0,))
        return np.atleast_1d(x)


class ReducedLinearModel(object):
    """A model with no data.

    If a Likelihood is Gaussian in the parameters, it is sometmise more
    clear/efficient to phrase it in terms of a parameter covariance, parameter
    mean and peak value:

    logL(theta) = logLmax - (theta - mu_L)' Sigma_L^{-1} (theta - mu_L)

    We can link this to a data-based model with the relations:

    Sigma_L = (M' C^{-1} M)^{-1}
    mu_L = Sigma_L M' C^{-1} (D-m)
    logLmax = - log|2 pi C|/2
              - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

    Parameters
    ----------
    mu_L : array_like
        Likelihood peak
    Sigma_L : array_like
        Likelihood covariance
    logLmax : float, optional
        Likelihood maximum, defaults to zero
    mu_pi : array_like, optional
        Prior mean, defaults to zero vector
    Sigma_pi : array_like, optional
        Prior covariance, defaults to identity matrix
    """

    def __init__(self, *args, **kwargs):
        self.mu_L = np.atleast_1d(kwargs.pop('mu_L'))
        self.Sigma_L = np.atleast_2d(kwargs.pop('Sigma_L', None))
        self.logLmax = kwargs.pop('logLmax', 0)
        self.mu_pi = np.atleast_1d(kwargs.pop('mu_pi',
                                              np.zeros_like(self.mu_L)))
        self.Sigma_pi = np.atleast_2d(kwargs.pop('Sigma_pi',
                                                 np.eye(len(self.mu_pi))))
        self.Sigma_P = inv(inv(self.Sigma_pi) + inv(self.Sigma_L))
        self.mu_P = self.Sigma_P @ (solve(self.Sigma_pi, self.mu_pi)
                                    + solve(self.Sigma_L, self.mu_L))

    def prior(self):
        """P(theta) as a scipy distribution object."""
        return multivariate_normal(self.mu_pi, self.Sigma_pi)

    def posterior(self):
        """P(theta|D) as a scipy distribution object."""
        return multivariate_normal(self.mu_P, self.Sigma_P)

    def logpi(self, theta):
        """P(theta) as a scalar."""
        return self.prior().logpdf(theta)

    def logP(self, theta):
        """P(theta|D) as a scalar."""
        return self.posterior().logpdf(theta)

    def logL(self, theta):
        """P(D|theta) as a scalar."""
        return (self.logLmax
                + multivariate_normal.logpdf(theta, self.mu_L, self.Sigma_L)
                + logdet(2 * np.pi * self.Sigma_L)/2)

    def logZ(self):
        """P(D) as a scalar."""
        return (self.logLmax + logdet(self.Sigma_P)/2 - logdet(self.Sigma_pi)/2
                - (self.mu_P - self.mu_pi
                   ) @ solve(self.Sigma_pi, self.mu_P - self.mu_pi)/2
                - (self.mu_P - self.mu_L
                   ) @ solve(self.Sigma_L, self.mu_P - self.mu_L)/2)

    def DKL(self):
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence."""
        return (logdet(self.Sigma_pi) - logdet(self.Sigma_P)
                + np.trace(inv(self.Sigma_pi) @ self.Sigma_P - 1)
                + (self.mu_P - self.mu_pi
                   ) @ solve(self.Sigma_pi, self.mu_P - self.mu_pi))/2


class ReducedLinearModelUniformPrior(object):
    """A model with no data.

    Gaussian likelihood in the parameters

    logL(theta) = logLmax - (theta - mu_L)' Sigma_L^{-1} (theta - mu_L)

    Uniform prior

    We can link this to a data-based model with the relations:

    Sigma_L = (M' C^{-1} M)^{-1}
    mu_L = Sigma_L M' C^{-1} (D-m)
    logLmax = -log|2 pi C|/2
              - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

    Parameters
    ----------
    mu_L : array_like
        Likelihood peak
    Sigma_L : array_like
        Likelihood covariance
    logLmax : float, optional
        Likelihood maximum, defaults to zero
    logV : float, optional
        log prior volume, defaults to zero
    """

    def __init__(self, *args, **kwargs):
        self.mu_L = np.atleast_1d(kwargs.pop('mu_L'))
        self.Sigma_L = np.atleast_2d(kwargs.pop('Sigma_L'))
        self.logLmax = kwargs.pop('logLmax', 0)
        self.logV = kwargs.pop('logV', 0)
        self.Sigma_P = self.Sigma_L
        self.mu_P = self.mu_L

    def posterior(self):
        """P(theta|D) as a scipy distribution object."""
        return multivariate_normal(self.mu_P, self.Sigma_P)

    def logpi(self, theta):
        """P(theta) as a scalar."""
        return - self.logV

    def logP(self, theta):
        """P(theta|D) as a scalar."""
        return self.posterior().logpdf(theta)

    def logL(self, theta):
        """P(D|theta) as a scalar."""
        return (self.logLmax + logdet(2 * np.pi * self.Sigma_L)/2
                + multivariate_normal.logpdf(theta, self.mu_L, self.Sigma_L))

    def logZ(self):
        """P(D) as a scalar."""
        return self.logLmax + logdet(2*np.pi*self.Sigma_P)/2 - self.logV

    def DKL(self):
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence."""
        return self.logV - logdet(2*np.pi*np.e*self.Sigma_P)/2


class LinearMixtureModel(object):
    """A linear mixture model.

    A linear mixture model is defined by

    D|theta, A ~ N(m + M theta, C)
    theta|A    ~ N(mu, Sigma)
    A          ~ categorical(exp(logA))

    Defined by:
    - Parameters: theta (n,)
    - Data: D (k, d)
    - Prior mean: mu (k, n)
    - Prior covariance: Sigma (k, n, n)
    - Data mean: m (k, d)
    - Data covariance: C (k, d, d)
    - log mixture weights: logA (k,)


    Parameters
    ----------
    M : array_like, optional
        Model matrix, defaults to identity matrix
    m : array_like, optional
        Data mean, defaults to zero vector
    C : array_like, optional
        Data covariance, defaults to identity matrix
    mu : array_like, optional
        Prior mean, defaults to zero vector
    Sigma : array_like, optional
        Prior covariance, defaults to identity matrix
    logA : array_like, optional
        Mixture log-weights, defaults to uniform weights

    the overall shape is attempted to be inferred from the input parameters.
    """

    def __init__(self, *args, **kwargs):
        # Rationalise input arguments
        M = self._atleast_3d(kwargs.pop('M', None))
        m = self._atleast_2d(kwargs.pop('m', None))
        C = self._atleast_3d(kwargs.pop('C', None))
        mu = self._atleast_2d(kwargs.pop('mu', None))
        Sigma = self._atleast_3d(kwargs.pop('Sigma', None))
        logA = self._atleast_1d(kwargs.pop('logA', None))

        # Determine dimensions
        n = max([M.shape[2], mu.shape[1], Sigma.shape[1], Sigma.shape[2]])
        d = max([M.shape[1], m.shape[1], C.shape[1], C.shape[2]])
        k = max([M.shape[0], m.shape[0], C.shape[0], mu.shape[0],
                 Sigma.shape[0], logA.shape[0]])
        if not n:
            raise ValueError('Unable to determine number of parameters n')
        if not d:
            raise ValueError('Unable to determine data dimensions d')
        if not k:
            raise ValueError('Unable to determine number of components k')

        # Set defaults if no argument was passed
        M = M if M.size else np.eye(d, n)
        m = m if m.size else np.zeros(d)
        C = C if C.size else np.eye(d)
        mu = mu if mu.size else np.zeros(n)
        Sigma = Sigma if Sigma.size else np.eye(n)
        logA = logA if logA.size else - np.log(k)

        # Broadcast to correct shape
        self.M = np.broadcast_to(M, (k, d, n))
        self.m = np.broadcast_to(m, (k, d))
        self.C = np.broadcast_to(C, (k, d, d))
        self.mu = np.broadcast_to(mu, (k, n))
        self.Sigma = np.broadcast_to(Sigma, (k, n, n))
        self.logA = np.broadcast_to(logA, (k,))

    @classmethod
    def from_joint(cls, means, covs, n):
        """Construct model from joint distribution."""
        mu = means[:, -n:]
        Sigma = covs[:, -n:, -n:]
        M = solve(Sigma, covs[:, -n:, :-n]).transpose(0, 2, 1)
        m = means[:, :-n] - np.einsum('ija,ia->ij', M, mu)
        C = covs[:, :-n, :-n] - np.einsum('ija,iab,ikb->ijk', M, Sigma, M)
        return cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    @property
    def n(self):
        """Dimensionality of parameter space len(theta)."""
        return self.M.shape[2]

    @property
    def d(self):
        """Dimensionality of data space len(D)."""
        return self.M.shape[1]

    @property
    def k(self):
        """Number of mixture components len(logA)."""
        return self.M.shape[0]

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D|theta,A ~ N( m + M theta, C )
        theta|A   ~ N( mu, Sigma )
        A         ~ categorical(exp(logA))

        Parameters
        ----------
        theta : array_like, shape (n,)
        """
        mu = self.m + np.einsum('ija,a->ij', self.M, theta)
        prior = self.prior()
        logA = (prior.logpdf(theta, reduce=False) + self.logA
                - prior.logpdf(theta))
        return mixture_multivariate_normal(mu, self.C, logA)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta|A ~ N( mu, Sigma )
        A       ~ categorical(exp(logA))
        """
        return mixture_multivariate_normal(self.mu, self.Sigma, self.logA)

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta|D, A ~ N( mu + S M'C^{-1}(D - m - M mu), S)
        D|A        ~ N( m + M mu, C + M Sigma M' )
        A          ~ categorical(exp(logA))
        S = (Sigma^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        Sigma = inv(self.invSigma + np.einsum('iaj,iab,ibk->ijk',
                                              self.M, self.invC, self.M))
        D0 = self.m + np.einsum('ija,ia->ij', self.M, self.mu)
        mu = self.mu + np.einsum('ija,iba,ibc,ic->ij',
                                 Sigma, self.M, self.invC, D-D0)
        evidence = self.evidence()
        logA = (evidence.logpdf(D, reduce=False) + self.logA
                - evidence.logpdf(D))
        return mixture_multivariate_normal(mu, Sigma, logA)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D|A ~ N( m + M mu, C + M Sigma M' )
        A   ~ categorical(exp(logA))
        """
        mu = self.m + np.einsum('ija,ia->ij', self.M, self.mu)
        Sigma = self.C + np.einsum('ija,iab,ikb->ijk',
                                   self.M, self.Sigma, self.M)
        return mixture_multivariate_normal(mu, Sigma, self.logA)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] | A ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta] |      ( [   mu   ] , [   Sigma M'      Sigma ] )

        A           ~ categorical(exp(logA))
        """
        evidence = self.evidence()
        prior = self.prior()
        mu = np.block([evidence.means, prior.means])
        corr = np.einsum('ija,ial->ijl', self.M, self.Sigma)
        Sigma = np.block([[evidence.covs, corr],
                          [corr.transpose(0, 2, 1), prior.covs]])
        return mixture_multivariate_normal(mu, Sigma, self.logA)

    @cached_property
    def invSigma(self):
        """Inverse of prior covariance."""
        return inv(self.Sigma)

    @cached_property
    def invC(self):
        """Inverse of data covariance."""
        return inv(self.C)

    def _atleast_3d(self, x):
        if x is None:
            return np.zeros(shape=(0, 0, 0))
        x = np.array(x)
        if x.size > 0:
            x = np.atleast_2d(x)
            if x.ndim == 2:
                x = x[None, ...]
        return x

    def _atleast_2d(self, x):
        if x is None:
            return np.zeros(shape=(0, 0))
        x = np.array(x)
        if x.size > 0:
            x = np.atleast_1d(x)
            if x.ndim == 1:
                x = x[None, ...]
        return x

    def _atleast_1d(self, x):
        if x is None:
            return np.zeros(shape=(0,))
        x = np.array(x)
        if x.size > 0:
            x = np.atleast_1d(x)
            if x.ndim == 0:
                x = x[None]
        return x
