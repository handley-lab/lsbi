"""Gaussian models for linear Bayesian inference."""
import numpy as np
from numpy.linalg import inv, solve

from lsbi.stats_1 import mixture_normal, multivariate_normal
from lsbi.utils import logdet


class LinearModel(object):
    """A multilinear model.

    D|theta ~ N( m + M theta, C )
    theta   ~ N( mu, Sigma )

    Defined by:
        Parameters:       theta (..., n,)
        Data:             D     (..., d,)
        Model:            M     (..., d, n)
        Prior mean:       mu    (..., n,)
        Prior covariance: Sigma (..., n, n)
        Data mean:        m     (..., d,)
        Data covariance:  C     (..., d, d)

    where the ellipses indicate arbitrary (broadcastable) additional copies.

    Parameters
    ----------
    M : array_like, optional
        if ndim>=2: model matrices
        if ndim==1: model matrix with vector diagonal for all components
        if ndim==0: scalar * rectangular identity matrix for all components
        Defaults to rectangular identity matrix
    m : array_like, optional
        if ndim>=1: data means
        if ndim==0: scalar * unit vector for all components
        Defaults to 0 for all components
    C : array_like, optional
        if ndim>=2: data covariances
        if ndim==1: data covariance with vector diagonal for all components
        if ndim==0: scalar * identity matrix for all components
        Defaults to rectangular identity matrix
    mu : array_like, optional
        if ndim>=1: prior means
        if ndim==0: scalar * unit vector for all components
        Defaults to 0 for all components
        Prior mean, defaults to zero vector
    Sigma : array_like, optional
        if ndim>=2: prior covariances
        if ndim==1: prior covariance with vector diagonal for all components
        if ndim==0: scalar * identity matrix for all components
        Defaults to k copies of identity matrices
    n : int, optional
        Number of parameters, defaults to automatically inferred value
    d : int, optional
        Number of data dimensions, defaults to automatically inferred value
    shape : (), optional
        Number of mixture components, defaults to automatically inferred value
    """

    def __init__(self, M=1, m=0, C=1, mu=0, Sigma=1, shape=(), n=1, d=1):
        self.M = M
        self.m = m
        self.C = C
        self.mu = mu
        self.Sigma = Sigma
        self._shape = shape
        self._n = n
        self._d = d

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.atleast_2d(self.M).shape[:-2],
            np.atleast_1d(self.m).shape[:-1],
            np.atleast_2d(self.C).shape[:-2],
            np.atleast_1d(self.mu).shape[:-1],
            np.atleast_2d(self.Sigma).shape[:-2],
            self._shape,
        )

    @property
    def n(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.M)[-1:],
                *np.shape(self.Sigma)[-2:],
                *np.shape(self.mu)[-1:],
                self._n,
            ]
        )

    @property
    def d(self):
        """Dimensionality of data space len(D)."""
        return np.max(
            [
                *np.shape(self.M)[-2:-1],
                *np.shape(self.C)[-2:],
                *np.shape(self.m)[-1:],
                self._d,
            ]
        )

    @classmethod
    def from_joint(cls, means, covs, n):
        """Construct model from joint distribution."""
        mu = means[:, -n:]
        Sigma = covs[:, -n:, -n:]
        M = solve(Sigma, covs[:, -n:, :-n]).transpose(0, 2, 1)
        m = means[:, :-n] - np.einsum("ija,ia->ij", M, mu)
        C = covs[:, :-n, :-n] - np.einsum("ija,iab,ikb->ijk", M, Sigma, M)
        return cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D|theta ~ N( m + M theta, C )
        theta   ~ N( mu, Sigma )

        Parameters
        ----------
        theta : array_like, shape (k, n)
        """
        if len(np.shape(self.M)) > 1:
            M = self.M
        else:
            M = self.M * np.eye(self.d, self.n)

        mu = self.m + np.einsum("...ja,...a->...j", M, theta)
        return multivariate_normal(mu, self.C, self.shape, self.d)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta ~ N( mu, Sigma )
        """
        return multivariate_normal(self.mu, self.Sigma, self.shape, self.n)

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta|D ~ N( mu + S M'C^{-1}(D - m - M mu), S )
        S = (Sigma^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        if len(np.shape(self.M)) > 1:
            values = D - self.m - np.einsum("...ja,...a->...j", self.M, self.mu)

            if len(np.shape(self.C)) > 1:
                MinvCM = np.einsum(
                    "...aj,...ab,...bk->...jk", self.M, inv(self.C), self.M
                )
            else:
                MinvCM = np.einsum(
                    "...ja,...kb->...jk", self.M, self.M / np.array(self.C)[:, None]
                )

            if len(np.shape(self.Sigma)) > 1:
                Sigma = inv(inv(self.Sigma) + MinvCM)
            else:
                Sigma = inv(np.eye(self.d) / self.Sigma + MinvCM)

            if len(np.shape(self.C)) > 1:
                mu = self.mu + np.einsum(
                    "...ja,...ba,...bc,...c->...j", Sigma, self.M, inv(self.C), values
                )
            else:
                mu = self.mu + np.einsum(
                    "...ja,...ac,...c->...j",
                    Sigma,
                    self.M / np.array(self.C)[:, None],
                    values,
                )
        else:
            values = D * np.ones(self.d)
            values[: self.n] = values[: self.n] - self.m - self.M * self.mu

            if len(np.shape(self.C)) > 1:
                MinvCM = (
                    np.atleast_1d(self.M)[..., None]
                    * inv(self.C)
                    * np.atleast_1d(self.M)[..., None, :]
                )
                if len(np.shape(self.Sigma)) > 1:
                    Sigma = inv(inv(self.Sigma) + MinvCM)
                else:
                    Sigma = inv(np.eye(self.d) / self.Sigma + MinvCM)

                mu = self.mu + np.einsum(
                    "...ja,...ba,...bc,...c->...j", Sigma, self.M, inv(self.C), values
                )
            else:
                MinvCM = self.M / np.atleast_1d(self.C)[: self.n] * self.M
                if len(np.shape(self.Sigma)) > 1:
                    Sigma = inv(inv(self.Sigma) + np.eye(self.n) * MinvCM)
                else:
                    Sigma = 1 / (1 / self.Sigma + MinvCM)

                mu = self.mu + np.einsum(
                    "...ja,...ac,...c->...j",
                    Sigma,
                    self.M / np.atleast_1d(self.C)[: self.n],
                    values,
                )

        return multivariate_normal(mu, Sigma, self.shape, self.n)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D ~ N( m + M mu, C + M Sigma M' )
        """
        if len(np.shape(self.M)) > 1:
            mu = self.m + np.einsum("...ja,...a->...j", self.M, self.mu)

            if len(np.shape(self.Sigma)) > 1:
                Sigma = np.einsum(
                    "...ja,...ab,...kb->...jk", self.M, self.Sigma, self.M
                )
            else:
                Sigma = np.einsum("...ja,...kb->...jk", self.M, self.Sigma * self.M)
            if len(np.shape(self.C)) > 1:
                Sigma = self.C + Sigma
            else:
                Sigma = self.C * np.eye(self.d) + Sigma
        else:
            mu = self.m * np.ones(self.d)
            mu[: self.n] = mu[: self.n] + self.M * self.mu
            Sigma = self.C

            if len(np.shape(self.Sigma)) > 1 or len(np.shape(self.C)) > 1:
                if len(np.shape(self.C)) <= 1:
                    Sigma = Sigma * np.eye(self.d)
                Sigma[: self.n, : self.n] = (
                    Sigma[: self.n, : self.n]
                    + np.atleast_1d(self.M)[..., None]
                    * self.Sigma
                    * np.atleast_1d(self.M)[..., None, :]
                )
            else:
                Sigma = Sigma * np.ones(self.d)
                Sigma[: self.n] = Sigma[: self.n] + self.M * self.Sigma * self.M

        return multivariate_normal(mu, Sigma, self.shape, self.d)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] | A ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta] |      ( [   mu   ] , [   Sigma M'      Sigma ] )
        """
        evidence = self.evidence()
        prior = self.prior()
        mu = np.block([evidence.mean * np.ones(self.d), prior.mean * np.ones(self.n)])
        corr = np.einsum(
            "...ja,...al->...jl",
            np.atleast_2d(self.M) * np.eye(self.n, self.d),
            np.atleast_2d(self.Sigma) * np.eye(self.n),
        )
        Sigma = np.block(
            [
                [np.atleast_2d(evidence.cov) * np.eye(self.d), corr],
                [np.moveaxis(corr, -1, -2), np.atleast_2d(prior.cov) * np.eye(self.n)],
            ]
        )
        return multivariate_normal(mu, Sigma, self.shape, len(mu))


class LinearMixtureModel(object):
    """A linear mixture model.

    D|theta, A ~ N( m + M theta, C )
    theta|A    ~ N( mu, Sigma )
    A          ~ categorical( exp(logA) )

    Defined by:
        Parameters:          theta (..., n,)
        Data:                D     (..., d,)
        Prior means:         mu    (..., k, n)
        Prior covariances:   Sigma (..., k, n, n)
        Data means:          m     (..., k, d)
        Data covariances:    C     (..., k, d, d)
        log mixture weights: logA  (..., k,)

    Parameters
    ----------
    M : array_like, optional
        if ndim>=2: model matrices
        if ndim==1: model matrix with vector diagonal for all components
        if scalar: scalar * rectangular identity matrix for all components
        Defaults to k copies of rectangular identity matrices
    m : array_like, optional
        if ndim>=1: data means
        if scalar: scalar * unit vector for all components
        Defaults to 0 for all components
    C : array_like, optional
        if ndim>=2: data covariances
        if ndim==1: data covariance with vector diagonal for all components
        if scalar: scalar * identity matrix for all components
        Defaults to k copies of identity matrices
    mu : array_like, optional
        if ndim>=1: prior means
        if scalar: scalar * unit vector for all components
        Defaults to 0 for all components
        Prior mean, defaults to zero vector
    Sigma : array_like, optional
        if ndim>=2: prior covariances
        if ndim==1: prior covariance with vector diagonal for all components
        if scalar: scalar * identity matrix for all components
        Defaults to k copies of identity matrices
    logA : array_like, optional
        if ndim>=1: log mixture weights
        if scalar: scalar * unit vector
        Defaults to uniform weights
    n : int, optional
        Number of parameters, defaults to automatically inferred value
    d : int, optional
        Number of data dimensions, defaults to automatically inferred value
    k : int, optional
        Number of mixture components, defaults to automatically inferred value
    """

    def __init__(self, logA=1, M=1, m=0, C=1, mu=0, Sigma=1, shape=(), n=1, d=1, k=1):
        self.logA = logA
        super().__init__(M=M, m=m, C=C, mu=mu, Sigma=Sigma, shape=shape, n=n, d=d)

    @classmethod
    def from_joint(cls, means, covs, logA, n):
        """Construct model from joint distribution."""
        mu = means[:, -n:]
        Sigma = covs[:, -n:, -n:]
        M = solve(Sigma, covs[:, -n:, :-n]).transpose(0, 2, 1)
        m = means[:, :-n] - np.einsum("ija,ia->ij", M, mu)
        C = covs[:, :-n, :-n] - np.einsum("ija,iab,ikb->ijk", M, Sigma, M)
        return cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma, logA=logA)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.array(self.logA).shape,
            np.atleast_2d(self.M).shape[:-2],
            np.atleast_1d(self.m).shape[:-1],
            np.atleast_2d(self.C).shape[:-2],
            np.atleast_1d(self.mu).shape[:-1],
            np.atleast_2d(self.Sigma).shape[:-2],
            self._shape,
        )

    @property
    def k(self):
        """Number of mixture components of the distribution."""
        return np.shape[-1]

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D|theta,A ~ N( m + M theta, C )
        theta|A   ~ N( mu, Sigma )
        A         ~ categorical(exp(logA))

        Parameters
        ----------
        theta : array_like, shape (n,)
        """
        dist = super().likelihood(theta)
        logA = self.prior().weights(theta)
        return mixture_normal(logA, dist.mean, dist.cov, dist.shape, dist.dim)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta|A ~ N( mu, Sigma )
        A       ~ categorical(exp(logA))
        """
        dist = super().prior()
        return mixture_normal(self.logA, dist.mean, dist.cov, dist.shape, dist.dim)

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta|D, A ~ N( mu + S M'C^{-1}(D - m - M mu), S )
        D|A        ~ N( m + M mu, C + M Sigma M' )
        A          ~ categorical(exp(logA))
        S = (Sigma^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        dist = super().posterior(D)
        logA = self.evidence().weights(D)
        return mixture_normal(logA, dist.mean, dist.cov, dist.shape, dist.dim)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D|A ~ N( m + M mu, C + M Sigma M' )
        A   ~ categorical(exp(logA))
        """
        dist = super().evidence()
        return mixture_normal(self.logA, dist.mean, dist.cov, dist.shape, dist.dim)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] | A ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta] |      ( [   mu   ] , [   Sigma M'      Sigma ] )

        A           ~ categorical(exp(logA))
        """
        dist = super().joint()
        return mixture_normal(self.logA, dist.mean, dist.cov, dist.shape, dist.dim)


class ReducedLinearModel(object):
    """A model with no data.

    If a Likelihood is Gaussian in the parameters, it is sometimes more
    clear/efficient to phrase it in terms of a parameter covariance, parameter
    mean and peak value:

    logL(theta) = logLmax - (theta - mu_L)' Sigma_L^{-1} (theta - mu_L)

    We can link this to a data-based model with the relations:

    Sigma_L = (M' C^{-1} M)^{-1}
    mu_L = Sigma_L M' C^{-1} (D-m)
    logLmax =
    - log|2 pi C|/2 - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

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
        self.mu_L = np.atleast_1d(kwargs.pop("mu_L"))
        self.Sigma_L = np.atleast_2d(kwargs.pop("Sigma_L", None))
        self.logLmax = kwargs.pop("logLmax", 0)
        self.mu_pi = np.atleast_1d(kwargs.pop("mu_pi", np.zeros_like(self.mu_L)))
        self.Sigma_pi = np.atleast_2d(kwargs.pop("Sigma_pi", np.eye(len(self.mu_pi))))
        self.Sigma_P = inv(inv(self.Sigma_pi) + inv(self.Sigma_L))
        self.mu_P = self.Sigma_P @ (
            solve(self.Sigma_pi, self.mu_pi) + solve(self.Sigma_L, self.mu_L)
        )

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
        return (
            self.logLmax
            + multivariate_normal(self.mu_L, self.Sigma_L).logpdf(theta)
            + logdet(2 * np.pi * self.Sigma_L) / 2
        )

    def logZ(self):
        """P(D) as a scalar."""
        return (
            self.logLmax
            + logdet(self.Sigma_P) / 2
            - logdet(self.Sigma_pi) / 2
            - (self.mu_P - self.mu_pi)
            @ solve(self.Sigma_pi, self.mu_P - self.mu_pi)
            / 2
            - (self.mu_P - self.mu_L) @ solve(self.Sigma_L, self.mu_P - self.mu_L) / 2
        )

    def DKL(self):
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence."""
        return (
            logdet(self.Sigma_pi)
            - logdet(self.Sigma_P)
            + np.trace(inv(self.Sigma_pi) @ self.Sigma_P - 1)
            + (self.mu_P - self.mu_pi) @ solve(self.Sigma_pi, self.mu_P - self.mu_pi)
        ) / 2


class ReducedLinearModelUniformPrior(object):
    """A model with no data.

    Gaussian likelihood in the parameters

    logL(theta) = logLmax - (theta - mu_L)' Sigma_L^{-1} (theta - mu_L)

    Uniform prior

    We can link this to a data-based model with the relations:

    Sigma_L = (M' C^{-1} M)^{-1}
    mu_L = Sigma_L M' C^{-1} (D-m)
    logLmax =
    -log|2 pi C|/2 - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

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
        self.mu_L = np.atleast_1d(kwargs.pop("mu_L"))
        self.Sigma_L = np.atleast_2d(kwargs.pop("Sigma_L"))
        self.logLmax = kwargs.pop("logLmax", 0)
        self.logV = kwargs.pop("logV", 0)
        self.Sigma_P = self.Sigma_L
        self.mu_P = self.mu_L

    def posterior(self):
        """P(theta|D) as a scipy distribution object."""
        return multivariate_normal(self.mu_P, self.Sigma_P)

    def logpi(self, theta):
        """P(theta) as a scalar."""
        return -self.logV

    def logP(self, theta):
        """P(theta|D) as a scalar."""
        return self.posterior().logpdf(theta)

    def logL(self, theta):
        """P(D|theta) as a scalar."""
        return (
            self.logLmax
            + logdet(2 * np.pi * self.Sigma_L) / 2
            + multivariate_normal(self.mu_L, self.Sigma_L).logpdf(theta)
        )

    def logZ(self):
        """P(D) as a scalar."""
        return self.logLmax + logdet(2 * np.pi * self.Sigma_P) / 2 - self.logV

    def DKL(self):
        """D_KL(P(theta|D)||P(theta)) the Kullback-Leibler divergence."""
        return self.logV - logdet(2 * np.pi * np.e * self.Sigma_P) / 2
