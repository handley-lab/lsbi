"""Gaussian models for linear Bayesian inference."""

import numpy as np
from numpy.linalg import inv, solve

from lsbi.stats import mixture_normal, multivariate_normal
from lsbi.utils import dediagonalise, logdet


class LinearModel(object):
    """A multilinear model.

    D|θ ~ N( m + M θ, C )
    θ   ~ N( μ, Σ )

    Defined by:
        Parameters:       θ (..., n,)
        Data:             D (..., d,)
        Model:            M (..., d, n)
        Prior mean:       μ (..., n,)
        Prior covariance: Σ (..., n, n)
        Data mean:        m (..., d,)
        Data covariance:  C (..., d, d)

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
    μ : array_like, optional
        if ndim>=1: prior means
        if ndim==0: scalar * unit vector for all components
        Defaults to 0 for all components
        Prior mean, defaults to zero vector
    Σ : array_like, optional
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
        M=1.0,
        m=0.0,
        C=1.0,
        μ=0.0,
        Σ=1.0,
        shape=(),
        n=1,
        d=1,
        diagonal_M=False,
        diagonal_C=False,
        diagonal_Σ=False,
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
        self.μ = μ
        self.Σ = Σ
        self.diagonal_Σ = diagonal_Σ
        if len(np.shape(self.Σ)) < 2:
            self.diagonal_Σ = True
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
            np.shape(self.μ)[:-1],
            np.shape(self.Σ)[: -2 + self.diagonal_Σ],
            self._shape,
        )

    @property
    def n(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.M)[len(np.shape(self.M)) - 1 + self.diagonal_M :],
                *np.shape(self.Σ)[-2 + self.diagonal_Σ :],
                *np.shape(self.μ)[-1:],
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

    def likelihood(self, θ):
        """P(D|θ as a scipy distribution object.

        D|θ ~ N( m + M θ, C )
        θ   ~ N( μ, Σ )

        Parameters
        ----------
        θ : array_like, shape (k, n)
        """
        μ = self.m + np.einsum("...ja,...a->...j", self._M, θ)
        return multivariate_normal(μ, self.C, self.shape, self.d, self.diagonal_C)

    def prior(self):
        """P(θ) as a scipy distribution object.

        θ ~ N( μ, Σ )
        """
        return multivariate_normal(self.μ, self.Σ, self.shape, self.n, self.diagonal_Σ)

    def posterior(self, D):
        """P(θ|D) as a scipy distribution object.

        θ|D ~ N( μ + S M'C^{-1}(D - m - M μ), S )
        S = (Σ^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        values = (
            D
            - self.m
            - np.einsum("...ja,...a->...j", self._M, self.μ * np.ones(self.n))
        )

        diagonal_Σ = self.diagonal_C and self.diagonal_Σ and self.diagonal_M

        if diagonal_Σ:
            dim = min(self.n, self.d)
            shape = np.broadcast_shapes(self.shape, values.shape[:-1])
            C = np.atleast_1d(self.C)[..., :dim]
            M = np.atleast_1d(self.M)[..., :dim]
            Σ = np.broadcast_to(self.Σ, shape + (self.n,)).copy()
            Σ[..., :dim] = 1 / (1 / Σ[..., :dim] + M**2 / C)

            μ = np.broadcast_to(self.μ, shape + (self.n,)).copy()
            μ[..., :dim] = μ[..., :dim] + Σ[..., :dim] * M / C * values[..., :dim]
        else:
            if self.diagonal_C:
                invC = np.eye(self.d) / np.atleast_1d(self.C)[..., None, :]
            else:
                invC = inv(self.C)

            if self.diagonal_Σ:
                invΣ = np.eye(self.n) / np.atleast_1d(self.Σ)[..., None, :]
            else:
                invΣ = inv(self.Σ)

            Σ = inv(
                invΣ + np.einsum("...aj,...ab,...bk->...jk", self._M, invC, self._M)
            )
            μ = self.μ + np.einsum(
                "...ja,...ba,...bc,...c->...j", Σ, self._M, invC, values
            )

        return multivariate_normal(μ, Σ, self.shape, self.n, diagonal_Σ)

    def evidence(self):
        """P(D) as a scipy distribution object.

        D ~ N( m + M μ, C + M Σ M' )
        """
        μ = self.m + np.einsum("...ja,...a->...j", self._M, self.μ * np.ones(self.n))
        diagonal_Σ = self.diagonal_C and self.diagonal_Σ and self.diagonal_M

        if diagonal_Σ:
            dim = min(self.n, self.d)
            M = np.atleast_1d(self.M)[..., :dim]
            S = np.atleast_1d(self.Σ)[..., :dim]
            Σ = np.broadcast_to(self.C, self.shape + (self.d,)).copy()
            Σ[..., :dim] += S * M**2
        else:
            Σ = self._C + np.einsum(
                "...ja,...ab,...kb->...jk", self._M, self._Σ, self._M
            )

        return multivariate_normal(μ, Σ, self.shape, self.d, diagonal_Σ)

    def joint(self):
        """P(θ, D) as a scipy distribution object.

        [θ] ~ N( [   μ   ]   [ Σ      Σ M'   ] )
        [D]    ( [m + M μ] , [M Σ  C + M Σ M'] )
        """
        evidence = self.evidence()
        prior = self.prior()
        b = np.broadcast_to(prior.mean, self.shape + (self.n,))
        a = np.broadcast_to(evidence.mean, self.shape + (self.d,))
        μ = np.block([b, a])
        A = dediagonalise(prior.cov, prior.diagonal, self.n)
        A = np.broadcast_to(A, self.shape + (self.n, self.n))
        D = dediagonalise(evidence.cov, evidence.diagonal, self.d)
        D = np.broadcast_to(D, self.shape + (self.d, self.d))
        C = np.einsum("...ja,...al->...jl", self._M, self._Σ)
        C = np.broadcast_to(C, self.shape + (self.d, self.n))
        B = np.moveaxis(C, -1, -2)
        Σ = np.block([[A, B], [C, D]])
        return multivariate_normal(μ, Σ, self.shape, self.n + self.d)

    @property
    def _M(self):
        return dediagonalise(self.M, self.diagonal_M, self.d, self.n)

    @property
    def _C(self):
        return dediagonalise(self.C, self.diagonal_C, self.d)

    @property
    def _Σ(self):
        return dediagonalise(self.Σ, self.diagonal_Σ, self.n)


class MixtureModel(LinearModel):
    """A linear mixture model.

    D|θ, A ~ N( m + M θ, C )
    θ|A    ~ N( μ, Σ )
    A          ~ categorical( exp(logA) )

    Defined by:
        Parameters:          θ     (..., n,)
        Data:                D     (..., d,)
        Prior means:         μ     (..., k, n)
        Prior covariances:   Σ     (..., k, n, n)
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
    μ : array_like, optional
        if ndim>=1: prior means
        if scalar: scalar * unit vector for all components
        Defaults to 0 for all components
        Prior mean, defaults to zero vector
    Σ : array_like, optional
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

    def __init__(self, logA=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logA = logA

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(np.shape(self.logA), super().shape)

    @property
    def k(self):
        """Number of mixture components."""
        if self.shape == ():
            return 1
        return self.shape[-1]

    def likelihood(self, θ):
        """P(D|θ) as a scipy distribution object.

        D|θ,A ~ N( m + M θ, C )
        θ|A   ~ N( μ, Σ )
        A         ~ categorical(exp(logA))

        Parameters
        ----------
        θ : array_like, shape (n,)
        """
        dist = super().likelihood(np.expand_dims(θ, -2))
        dist.__class__ = mixture_normal
        dist.logA = self.prior().logpdf(θ, broadcast=True, joint=True)
        return dist

    def prior(self):
        """P(θ) as a scipy distribution object.

        θ|A ~ N( μ, Σ )
        A       ~ categorical(exp(logA))
        """
        dist = super().prior()
        dist.__class__ = mixture_normal
        dist.logA = self.logA
        return dist

    def posterior(self, D):
        """P(θ|D) as a scipy distribution object.

        θ|D, A ~ N( μ + S M'C^{-1}(D - m - M μ), S )
        D|A        ~ N( m + M μ, C + M Σ M' )
        A          ~ categorical(exp(logA))
        S = (Σ^{-1} + M'C^{-1}M)^{-1}

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

        D|A ~ N( m + M μ, C + M Σ M' )
        A   ~ categorical(exp(logA))
        """
        dist = super().evidence()
        dist.__class__ = mixture_normal
        dist.logA = self.logA
        return dist

    def joint(self):
        """P(D, θ) as a scipy distribution object.

        [θ] | A ~ N( [   μ   ]   [ Σ      Σ M'   ] )
        [  D  ] |      ( [m + M μ] , [M Σ  C + M Σ M'] )

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

    logL(θ) = logLmax - (θ - μ_L)' Σ_L^{-1} (θ - μ_L)

    We can link this to a data-based model with the relations:

    Σ_L = (M' C^{-1} M)^{-1}
    μ_L = Σ_L M' C^{-1} (D-m)
    logLmax =
    - log|2 pi C|/2 - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

    Parameters
    ----------
    μ_L : array_like
        Likelihood peak
    Σ_L : array_like
        Likelihood covariance
    logLmax : float, optional
        Likelihood maximum, defaults to zero
    μ_pi : array_like, optional
        Prior mean, defaults to zero vector
    Σ_pi : array_like, optional
        Prior covariance, defaults to identity matrix
    """

    def __init__(self, *args, **kwargs):
        self.μ_L = np.atleast_1d(kwargs.pop("μ_L"))
        self.Σ_L = np.atleast_2d(kwargs.pop("Σ_L", None))
        self.logLmax = kwargs.pop("logLmax", 0)
        self.μ_pi = np.atleast_1d(kwargs.pop("μ_pi", np.zeros_like(self.μ_L)))
        self.Σ_pi = np.atleast_2d(kwargs.pop("Σ_pi", np.eye(len(self.μ_pi))))
        self.Σ_P = inv(inv(self.Σ_pi) + inv(self.Σ_L))
        self.μ_P = self.Σ_P @ (solve(self.Σ_pi, self.μ_pi) + solve(self.Σ_L, self.μ_L))

    def prior(self):
        """P(θ) as a scipy distribution object."""
        return multivariate_normal(self.μ_pi, self.Σ_pi)

    def posterior(self):
        """P(θ|D) as a scipy distribution object."""
        return multivariate_normal(self.μ_P, self.Σ_P)

    def logpi(self, θ):
        """P(θ) as a scalar."""
        return self.prior().logpdf(θ)

    def logP(self, θ):
        """P(θ|D) as a scalar."""
        return self.posterior().logpdf(θ)

    def logL(self, θ):
        """P(D|θ) as a scalar."""
        return (
            self.logLmax
            + multivariate_normal(self.μ_L, self.Σ_L).logpdf(θ)
            + logdet(2 * np.pi * self.Σ_L) / 2
        )

    def logZ(self):
        """P(D) as a scalar."""
        return (
            self.logLmax
            + logdet(self.Σ_P) / 2
            - logdet(self.Σ_pi) / 2
            - (self.μ_P - self.μ_pi) @ solve(self.Σ_pi, self.μ_P - self.μ_pi) / 2
            - (self.μ_P - self.μ_L) @ solve(self.Σ_L, self.μ_P - self.μ_L) / 2
        )

    def DKL(self):
        """D_KL(P(θ|D)||P(θ)) the Kullback-Leibler divergence."""
        return (
            logdet(self.Σ_pi)
            - logdet(self.Σ_P)
            + np.trace(inv(self.Σ_pi) @ self.Σ_P - 1)
            + (self.μ_P - self.μ_pi) @ solve(self.Σ_pi, self.μ_P - self.μ_pi)
        ) / 2


class ReducedLinearModelUniformPrior(object):
    """A model with no data.

    Gaussian likelihood in the parameters

    logL(θ) = logLmax - (θ - μ_L)' Σ_L^{-1} (θ - μ_L)

    Uniform prior

    We can link this to a data-based model with the relations:

    Σ_L = (M' C^{-1} M)^{-1}
    μ_L = Σ_L M' C^{-1} (D-m)
    logLmax =
    -log|2 pi C|/2 - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

    Parameters
    ----------
    μ_L : array_like
        Likelihood peak
    Σ_L : array_like
        Likelihood covariance
    logLmax : float, optional
        Likelihood maximum, defaults to zero
    logV : float, optional
        log prior volume, defaults to zero
    """

    def __init__(self, *args, **kwargs):
        self.μ_L = np.atleast_1d(kwargs.pop("μ_L"))
        self.Σ_L = np.atleast_2d(kwargs.pop("Σ_L"))
        self.logLmax = kwargs.pop("logLmax", 0)
        self.logV = kwargs.pop("logV", 0)
        self.Σ_P = self.Σ_L
        self.μ_P = self.μ_L

    def posterior(self):
        """P(θ|D) as a scipy distribution object."""
        return multivariate_normal(self.μ_P, self.Σ_P)

    def logpi(self, θ):
        """P(θ) as a scalar."""
        return -self.logV

    def logP(self, θ):
        """P(θ|D) as a scalar."""
        return self.posterior().logpdf(θ)

    def logL(self, θ):
        """P(D|θ) as a scalar."""
        return (
            self.logLmax
            + logdet(2 * np.pi * self.Σ_L) / 2
            + multivariate_normal(self.μ_L, self.Σ_L).logpdf(θ)
        )

    def logZ(self):
        """P(D) as a scalar."""
        return self.logLmax + logdet(2 * np.pi * self.Σ_P) / 2 - self.logV

    def DKL(self):
        """D_KL(P(θ|D)||P(θ)) the Kullback-Leibler divergence."""
        return self.logV - logdet(2 * np.pi * np.e * self.Σ_P) / 2
