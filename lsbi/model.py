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

    Model M:  D = m + M theta +/- sqrt(C)

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

        self.M = kwargs.pop('M', None)
        self.m = kwargs.pop('m', None)
        self.C = kwargs.pop('C', None)
        self.mu = kwargs.pop('mu', None)
        self.Sigma = kwargs.pop('Sigma', None)

        n, d = None, None

        if self.m is not None:
            self.m = np.atleast_1d(self.m)
            d, = self.m.shape
        if self.C is not None:
            self.C = np.atleast_2d(self.C)
            d, d = self.C.shape
        if self.Sigma is not None:
            self.Sigma = np.atleast_2d(self.Sigma)
            n, n = self.Sigma.shape
        if self.mu is not None:
            self.mu = np.atleast_1d(self.mu)
            n, = self.mu.shape
        if self.M is not None:
            self.M = np.atleast_2d(self.M)
            d, n = self.M.shape

        if n is None:
            raise ValueError('Unable to determine number of parameters n')
        if d is None:
            raise ValueError('Unable to determine data dimensions d')

        if self.M is None:
            self.M = np.eye(d, n)
        if self.m is None:
            self.m = np.zeros(d)
        if self.C is None:
            self.C = np.eye(d)
        if self.mu is None:
            self.mu = np.zeros(n)
        if self.Sigma is None:
            self.Sigma = np.eye(n)

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
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence."""
        cov_p = self.posterior(D).cov
        cov_q = self.prior().cov
        mu_p = self.posterior(D).mean
        mu_q = self.prior().mean
        return (- logdet(cov_p) + logdet(cov_q)
                + np.trace(inv(cov_q) @ cov_p - 1)
                + (mu_q - mu_p) @ inv(cov_q) @ (mu_q - mu_p))/2

    def reduce(self, D):
        """Reduce the model to a Gaussian in the parameters."""
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
    logAA : array_like, optional
        Mixture log-weights, defaults to uniform weights

    the overall shape is attempted to be inferred from the input parameters.
    """

    def __init__(self, *args, **kwargs):

        self.M = kwargs.pop('M', None)
        self.m = kwargs.pop('m', None)
        self.C = kwargs.pop('C', None)
        self.mu = kwargs.pop('mu', None)
        self.Sigma = kwargs.pop('Sigma', None)
        self.logA = kwargs.pop('logA', None)

        k, n, d = None, None, None

        if self.m is not None:
            self.m = np.atleast_2d(self.m)
            k, d = self.m.shape
        if self.C is not None:
            self.C = np.atleast_3d(self.C)
            k, d, d = self.C.shape
        if self.Sigma is not None:
            self.Sigma = np.atleast_3d(self.Sigma)
            k, n, n = self.Sigma.shape
        if self.mu is not None:
            self.mu = np.atleast_2d(self.mu)
            k, n, = self.mu.shape
        if self.M is not None:
            self.M = np.atleast_3d(self.M)
            k, d, n = self.M.shape
        if self.logA is not None:
            self.logA = np.atleast_1d(self.logA)
            k, = self.logA.shape

        if n is None:
            raise ValueError('Unable to determine number of parameters n')
        if d is None:
            raise ValueError('Unable to determine data dimensions d')
        if k is None:
            raise ValueError('Unable to determine number of components k')

        if self.logA is None:
            self.logA = np.zeros(k) - np.log(k)
        if self.M is None:
            self.M = np.broadcast_to(np.eye(d, n), (k, d, n))
        if self.m is None:
            self.m = np.broadcast_to(np.zeros(d), (k, d))
        if self.C is None:
            self.C = np.broadcast_to(np.eye(d), (k, d, d))
        if self.mu is None:
            self.mu = np.broadcast_to(np.zeros(n), (k, n))
        if self.Sigma is None:
            self.Sigma = np.broadcast_to(np.eye(n), (k, n, n))

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
        return self.M.shape[2]

    @property
    def d(self):
        """Dimensionality of data space len(D)."""
        return self.M.shape[1]

    @property
    def k(self):
        """Number of mixture components."""
        return self.M.shape[0]

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D ~ N( m + M theta, C )
        """
        mu = self.m + np.einsum('ijk,k->ij', self.M, theta)
        prior = self.prior()
        logA = np.squeeze(prior.logpdfs(theta) + self.logA
                          - prior.logpdf(theta))
        return mixture_multivariate_normal(mu, self.C, logA)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta ~ N( mu, Sigma )
        """
        return mixture_multivariate_normal(self.mu, self.Sigma, self.logA)

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta ~ N( mu + Sigma M'C^{-1}(D-m), Sigma - Sigma M' C^{-1} M Sigma )
        """
        Sigma = inv(self.invSigma +
                    np.einsum('iaj,iab,ibk->ijk', self.M, self.invC, self.M))
        D0 = self.m + np.einsum('iaj,ij->ia', self.M, self.mu)
        mu = self.mu + np.einsum('iab,icb,icf,if->ia',
                                 Sigma, self.M, self.invC, D-D0)
        evidence = self.evidence()
        logA = np.squeeze(evidence.logpdfs(D) + self.logA
                          - evidence.logpdf(D))
        return mixture_multivariate_normal(mu, Sigma, logA)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D ~ N( m + M mu, C + M Sigma M' )
        """
        mu = self.m + np.einsum('ijk,ik->ij', self.M, self.mu)
        Sigma = self.C + np.einsum('ija,iab,ikb->ijk',
                                   self.M, self.Sigma, self.M)
        return mixture_multivariate_normal(mu, Sigma, self.logA)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta]    ( [   mu   ] , [   Sigma M'      Sigma ] )
        """
        evidence = self.evidence()
        prior = self.prior()
        mu = np.block([evidence.means, prior.means])
        corr = np.einsum('ijk,ikl->ijl', self.M, self.Sigma)
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
