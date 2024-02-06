"""Gaussian models for linear Bayesian inference."""
import numpy as np
from numpy.linalg import inv, solve

from lsbi.stats import mixture_normal, multivariate_normal
from lsbi.utils import logdet, matrix


def _de_diagonalise(x, diagonal, *args):
    if diagonal:
        return np.atleast_1d(x)[..., None, :] * np.eye(*args)
    else:
        return x


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

    def __init__(
        self,
        M=1,
        m=0,
        C=1,
        mu=0,
        Sigma=1,
        shape=(),
        n=1,
        d=1,
        diagonal_M=False,
        diagonal_C=False,
        diagonal_Sigma=False,
    ):
        self.M = M
        self.diagonal_M = diagonal_M
        if len(np.shape(self.M)) < 2:
            self.diagonal_M = True
        self.m = m
        self.C = C
        self.diagonal_C = diagonal_C
        if len(np.shape(self.C)) < 2:
            self.diagonal_C = True
        self.mu = mu
        self.Sigma = Sigma
        self.diagonal_Sigma = diagonal_Sigma
        if len(np.shape(self.Sigma)) < 2:
            self.diagonal_Sigma = True
        self._shape = shape
        self._n = n
        self._d = d

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.M)[: -2 + self.diagonal_M],
            np.shape(self.m)[:-1],
            np.shape(self.C)[: -2 + self.diagonal_C],
            np.shape(self.mu)[:-1],
            np.shape(self.Sigma)[: -2 + self.diagonal_Sigma],
            self._shape,
        )

    @property
    def n(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.M)[len(np.shape(self.M)) - 1 + self.diagonal_M :],
                *np.shape(self.Sigma)[-2 + self.diagonal_Sigma :],
                *np.shape(self.mu)[-1:],
                self._n,
            ]
        )

    @property
    def d(self):
        """Dimensionality of data space len(D)."""
        return np.max(
            [
                *np.shape(self.M)[-2 + self.diagonal_M : -1],
                *np.shape(self.C)[-2 + self.diagonal_C :],
                *np.shape(self.m)[-1:],
                self._d,
            ]
        )

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D|theta ~ N( m + M theta, C )
        theta   ~ N( mu, Sigma )

        Parameters
        ----------
        theta : array_like, shape (k, n)
        """
        mu = self.m + np.einsum("...ja,...a->...j", self._M, theta)
        return multivariate_normal(mu, self.C, self.shape, self.d, self.diagonal_C)

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta ~ N( mu, Sigma )
        """
        return multivariate_normal(
            self.mu, self.Sigma, self.shape, self.n, self.diagonal_Sigma
        )

    def posterior(self, D):
        """P(theta|D) as a scipy distribution object.

        theta|D ~ N( mu + S M'C^{-1}(D - m - M mu), S )
        S = (Sigma^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        values = (
            D
            - self.m
            - np.einsum("...ja,...a->...j", self._M, self.mu * np.ones(self.n))
        )

        diagonal_Sigma = self.diagonal_C and self.diagonal_Sigma and self.diagonal_M

        if diagonal_Sigma:
            dim = min(self.n, self.d)
            shape = np.broadcast_shapes(self.shape, values.shape[:-1])
            C = np.atleast_1d(self.C)[..., :dim]
            M = np.atleast_1d(self.M)[..., :dim]
            Sigma = np.broadcast_to(self.Sigma, shape + (self.n,)).copy()
            Sigma[..., :dim] = 1 / (1 / Sigma[..., :dim] + M**2 / C)

            mu = np.broadcast_to(self.mu, shape + (self.n,)).copy()
            mu[..., :dim] = mu[..., :dim] + Sigma[..., :dim] * M / C * values[..., :dim]
        else:
            if self.diagonal_C:
                invC = np.eye(self.d) / np.atleast_1d(self.C)[..., None, :]
            else:
                invC = inv(self.C)

            if self.diagonal_Sigma:
                invSigma = np.eye(self.n) / np.atleast_1d(self.Sigma)[..., None, :]
            else:
                invSigma = inv(self.Sigma)

            Sigma = inv(
                invSigma + np.einsum("...aj,...ab,...bk->...jk", self._M, invC, self._M)
            )
            mu = self.mu + np.einsum(
                "...ja,...ba,...bc,...c->...j", Sigma, self._M, invC, values
            )

        return multivariate_normal(mu, Sigma, self.shape, self.n, diagonal_Sigma)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D ~ N( m + M mu, C + M Sigma M' )
        """
        mu = self.m + np.einsum("...ja,...a->...j", self._M, self.mu * np.ones(self.n))
        diagonal_Sigma = self.diagonal_C and self.diagonal_Sigma and self.diagonal_M

        if diagonal_Sigma:
            dim = min(self.n, self.d)
            M = np.atleast_1d(self.M)[..., :dim]
            S = np.atleast_1d(self.Sigma)[..., :dim]
            Sigma = np.broadcast_to(self.C, self.shape + (self.d,)).copy()
            Sigma[..., :dim] += S * M**2
        else:
            Sigma = self._C + np.einsum(
                "...ja,...ab,...kb->...jk", self._M, self._Sigma, self._M
            )

        return multivariate_normal(mu, Sigma, self.shape, self.d, diagonal_Sigma)

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] | A ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta] |      ( [   mu   ] , [   Sigma M'      Sigma ] )
        """
        evidence = self.evidence()
        prior = self.prior()
        a = np.broadcast_to(evidence.mean, self.shape + (self.d,))
        b = np.broadcast_to(prior.mean, self.shape + (self.n,))
        mu = np.block([a, b])
        A = _de_diagonalise(evidence.cov, evidence.diagonal_cov, self.d)
        A = np.broadcast_to(A, self.shape + (self.d, self.d))
        D = _de_diagonalise(prior.cov, prior.diagonal_cov, self.n)
        D = np.broadcast_to(D, self.shape + (self.n, self.n))
        B = np.einsum("...ja,...al->...jl", self._M, self._Sigma)
        B = np.broadcast_to(B, self.shape + (self.d, self.n))
        C = np.moveaxis(B, -1, -2)
        Sigma = np.block([[A, B], [C, D]])
        return multivariate_normal(mu, Sigma, self.shape, self.n + self.d)

    @property
    def _M(self):
        return _de_diagonalise(self.M, self.diagonal_M, self.d, self.n)

    @property
    def _C(self):
        return _de_diagonalise(self.C, self.diagonal_C, self.d)

    @property
    def _Sigma(self):
        return _de_diagonalise(self.Sigma, self.diagonal_Sigma, self.n)


class MixtureModel(LinearModel):
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
    """

    def __init__(self, logA=1, *args):
        super().__init__(*args)
        self.logA = logA

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(np.shape(self.logA), super().shape)

    @property
    def k(self):
        """Number of mixture components of the distribution."""
        return self.shape[-1]

    def likelihood(self, theta):
        """P(D|theta) as a scipy distribution object.

        D|theta,A ~ N( m + M theta, C )
        theta|A   ~ N( mu, Sigma )
        A         ~ categorical(exp(logA))

        Parameters
        ----------
        theta : array_like, shape (n,)
        """
        dist = super().likelihood(np.expand_dims(theta, -2))
        dist.__class__ = mixture_normal
        dist.logA = self.prior().logpdf(theta, broadcast=True, joint=True)
        return dist

    def prior(self):
        """P(theta) as a scipy distribution object.

        theta|A ~ N( mu, Sigma )
        A       ~ categorical(exp(logA))
        """
        dist = super().prior()
        dist.__class__ = mixture_normal
        dist.logA = self.logA
        return dist

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
        dist = super().posterior(np.expand_dims(D, -2))
        dist.__class__ = mixture_normal
        dist.logA = self.evidence().logpdf(D, broadcast=True, joint=True)
        return dist

    def evidence(self):
        """P(D) as a scipy distribution object.

        D|A ~ N( m + M mu, C + M Sigma M' )
        A   ~ categorical(exp(logA))
        """
        dist = super().evidence()
        dist.__class__ = mixture_normal
        dist.logA = self.logA
        return dist

    def joint(self):
        """P(D, theta) as a scipy distribution object.

        [  D  ] | A ~ N( [m + M mu]   [C + M Sigma M'  M Sigma] )
        [theta] |      ( [   mu   ] , [   Sigma M'      Sigma ] )

        A           ~ categorical(exp(logA))
        """
        dist = super().joint()
        dist.__class__ = mixture_normal
        dist.logA = self.logA
        return dist


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
