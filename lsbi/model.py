"""Gaussian models for linear Bayesian inference."""

import copy

import numpy as np
from numpy.linalg import inv, solve
from scipy.special import logsumexp

from lsbi.stats import dkl, mixture_normal, multivariate_normal
from lsbi.utils import alias, dediagonalise, logdet


class LinearModel(Model):
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
    μ : (or mu) array_like, optional
        if ndim>=1: prior means
        if ndim==0: scalar * unit vector for all components
        Defaults to 0 for all components
        Prior mean, defaults to zero vector
    Σ : (or Sigma) array_like, optional
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

    def __init__(self, *args, **kwargs):
        self.M = kwargs.pop("M", 1)
        self.diagonal_M = kwargs.pop("diagonal_M", False)
        if len(np.shape(self.M)) < 2:
            self.diagonal_M = True
        self.m = kwargs.pop("m", 0)
        self.C = kwargs.pop("C", 1)
        self.diagonal_C = kwargs.pop("diagonal_C", False)
        if len(np.shape(self.C)) < 2:
            self.diagonal_C = True
        self.μ = kwargs.pop("μ", 0)
        self.μ = kwargs.pop("mu", self.μ)
        self.Σ = kwargs.pop("Σ", 1)
        self.Σ = kwargs.pop("Sigma", self.Σ)
        self.diagonal_Σ = kwargs.pop("diagonal_Σ", False)
        self.diagonal_Σ = kwargs.pop("diagonal_Sigma", self.diagonal_Σ)
        if len(np.shape(self.Σ)) < 2:
            self.diagonal_Σ = True
        self._shape = kwargs.pop("shape", ())
        self._n = kwargs.pop("n", 1)
        self._d = kwargs.pop("d", 1)

        if kwargs:
            raise ValueError(f"Unrecognised arguments: {kwargs}")

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

    def model(self, θ):
        """Model matrix M(θ) for a given parameter vector.

        M(θ) = m + M θ

        Parameters
        ----------
        θ : array_like, shape (..., n,)
        """
        return self.m + np.einsum("...ja,...a->...j", self._M, θ * np.ones(self.n))

    def likelihood(self, θ):
        """P(D|θ) as a distribution object.

        D|θ ~ N( m + M θ, C )
        θ   ~ N( μ, Σ )

        Parameters
        ----------
        θ : array_like, shape (k, n)
        """
        μ = self.model(θ)
        return multivariate_normal(μ, self.C, self.shape, self.d, self.diagonal_C)

    def prior(self):
        """P(θ) as a distribution object.

        θ ~ N( μ, Σ )
        """
        return multivariate_normal(self.μ, self.Σ, self.shape, self.n, self.diagonal_Σ)

    def posterior(self, D):
        """P(θ|D) as a distribution object.

        θ|D ~ N( μ + S M'C^{-1}(D - m - M μ), S )
        S = (Σ^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        values = D - self.model(self.μ)

        diagonal_Σ = self.diagonal_C and self.diagonal_Σ and self.diagonal_M

        if diagonal_Σ:
            dim = min(self.n, self.d)
            shape = np.broadcast_shapes(self.shape, values.shape[:-1])
            C = np.atleast_1d(self.C)[..., :dim]
            M = np.atleast_1d(self.M)[..., :dim]
            Σ = self.Σ * np.ones((*shape, self.n))
            Σ[..., :dim] = 1 / (1 / Σ[..., :dim] + M**2 / C)

            μ = self.μ * np.ones((*shape, self.n))
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
        """P(D) as a distribution object.

        D ~ N( m + M μ, C + M Σ M' )
        """
        diagonal_Σ = self.diagonal_C and self.diagonal_Σ and self.diagonal_M
        if diagonal_Σ:
            dim = min(self.n, self.d)
            M = np.atleast_1d(self.M)[..., :dim]
            S = np.atleast_1d(self.Σ)[..., :dim]
            Σ = self.C * np.ones(
                (
                    *self.shape,
                    self.d,
                )
            )
            Σ[..., :dim] = Σ[..., :dim] + S * M**2
        else:
            Σ = self._C + np.einsum(
                "...ja,...ab,...kb->...jk", self._M, self._Σ, self._M
            )
        μ = self.model(self.μ)
        return multivariate_normal(μ, Σ, self.shape, self.d, diagonal_Σ)

    def joint(self):
        """P(θ, D) as a distribution object.

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

    def update(self, D, inplace=False):
        """Bayesian update of the model with data.

        Parameters
        ----------
        D : array_like, shape (..., d)
        """
        dist = copy.deepcopy(self) if not inplace else self
        posterior = self.posterior(D)
        dist.μ = posterior.mean
        dist.Σ = posterior.cov
        dist.diagonal_Σ = posterior.diagonal
        if not inplace:
            return dist

    def ppd(self, D0):
        """P(D|D0) as a distribution object."""
        return self.update(D0).evidence()

    def dkl(self, D, n=0):
        """KL divergence between the posterior and prior.

        Parameters
        ----------
        D : array_like, shape (..., d)
            Data to form the posterior
        n : int, optional
            Number of samples for a monte carlo estimate, defaults to 0
        """
        return dkl(self.posterior(D), self.prior(), n)

    @property
    def _M(self):
        return dediagonalise(self.M, self.diagonal_M, self.d, self.n)

    @property
    def _C(self):
        return dediagonalise(self.C, self.diagonal_C, self.d)

    @property
    def _Σ(self):
        return dediagonalise(self.Σ, self.diagonal_Σ, self.n)

    def reduce(self, D):
        """Reduce the model to a reduced model.

        Parameters
        ----------
        D : array_like, shape (..., d)
            Data to form the reduced model

        Returns
        -------
        ReducedLinearModel
        """
        # TODO modify this to work with diagonal matrices
        Σ_L = inv(np.einsum("...ja,...ab,...kb->...jk", self._M, inv(self._C), self._M))
        μ_L = np.einsum(
            "...ja,...ab,...bc,...c->...j", Σ_L, self._M, inv(self._C), D - self.m
        )
        logLmax = (
            -logdet(2 * np.pi * self.C, self.diagonal_C) / 2
            - np.einsum("...a,...ab,...b->...", D - self.m, inv(self.C), D - self.m) / 2
        )
        return ReducedLinearModel(
            μ_L=μ_L,
            Σ_L=Σ_L,
            logLmax=logLmax,
            n=self.n,
            shape=self.shape,
            μ=self.μ,
            Σ=self.Σ,
            diagonal_Σ=self.diagonal_Σ,
        )


alias(LinearModel, "μ", "mu")
alias(LinearModel, "Σ", "Sigma")
alias(LinearModel, "diagonal_Σ", "diagonal_Sigma")


class MixtureModel(LinearModel):
    """A linear mixture model.

    D|θ, w ~ N( m + M θ, C )
    θ|w    ~ N( μ, Σ )
    w      ~ categorical( exp(logw) )

    Defined by:
        Parameters:          θ     (..., n,)
        Data:                D     (..., d,)
        Prior means:         μ     (..., k, n)
        Prior covariances:   Σ     (..., k, n, n)
        Data means:          m     (..., k, d)
        Data covariances:    C     (..., k, d, d)
        log mixture weights: logw  (..., k,)

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
    logw : array_like, optional
        if ndim>=1: log mixture weights
        if scalar: scalar * unit vector
        Defaults to uniform weights
    n : int, optional
        Number of parameters, defaults to automatically inferred value
    d : int, optional
        Number of data dimensions, defaults to automatically inferred value
    """

    def __init__(self, *args, **kwargs):
        self.logw = kwargs.pop("logw", 0)
        super().__init__(*args, **kwargs)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(np.shape(self.logw), super().shape)

    @property
    def k(self):
        """Number of mixture components."""
        if self.shape == ():
            return 1
        return self.shape[-1]

    def likelihood(self, θ):
        """P(D|θ) as a distribution object.

        D|θ,w ~ N( m + M θ, C )
        w|θ   ~ categorical(...)

        Parameters
        ----------
        θ : array_like, shape (n,)
        """
        dist = super().likelihood(np.expand_dims(θ, -2))
        dist.__class__ = mixture_normal
        prior = self.prior()
        dist.logw = prior.logpdf(θ, broadcast=True, joint=True)
        dist.logw = dist.logw - logsumexp(dist.logw, axis=-1, keepdims=True)
        return dist

    def prior(self):
        """P(θ) as a distribution object.

        θ|w ~ N( μ, Σ )
        w   ~ categorical(exp(logw))
        """
        dist = super().prior()
        dist.__class__ = mixture_normal
        dist.logw = self.logw
        return dist

    def posterior(self, D):
        """P(θ|D) as a distribution object.

        θ|D, w ~ N( μ + S M'C^{-1}(D - m - M μ), S )
        w|D    ~ P(D|w)P(w)/P(D)
        S = (Σ^{-1} + M'C^{-1}M)^{-1}

        Parameters
        ----------
        D : array_like, shape (d,)
        """
        dist = super().posterior(np.expand_dims(D, -2))
        dist.__class__ = mixture_normal
        dist.logw = self.evidence().logpdf(D, broadcast=True, joint=True)
        dist.logw = dist.logw - logsumexp(dist.logw, axis=-1, keepdims=True)
        return dist

    def evidence(self):
        """P(D) as a distribution object.

        D|w ~ N( m + M μ, C + M Σ M' )
        w   ~ categorical(exp(logw))
        """
        dist = super().evidence()
        dist.__class__ = mixture_normal
        dist.logw = self.logw
        return dist

    def joint(self):
        """P(D, θ) as a distribution object.

        [θ] | w ~ N( [   μ   ]   [ Σ      Σ M'   ] )
        [D] |      ( [m + M μ] , [M Σ  C + M Σ M'] )

        w           ~ categorical(exp(logw))
        """
        dist = super().joint()
        dist.__class__ = mixture_normal
        dist.logw = self.logw
        return dist

    def update(self, D, inplace=False):
        """Bayesian update of the model with data.

        Parameters
        ----------
        D : array_like, shape (..., d)
        """
        dist = copy.deepcopy(self) if not inplace else self
        posterior = self.posterior(D)
        dist.μ = posterior.mean
        dist.Σ = posterior.cov
        dist.diagonal_Σ = posterior.diagonal
        dist.logw = posterior.logw
        if not inplace:
            return dist

    def dkl(self, D, n=0):
        """KL divergence between the posterior and prior.

        Parameters
        ----------
        D : array_like, shape (..., d)
            Data to form the posterior
        n : int, optional
            Number of samples for a monte carlo estimate, defaults to 0
        """
        if n == 0:
            raise ValueError("MixtureModel requires a monte carlo estimate. Use n>0.")

        p = self.posterior(D)
        q = self.prior()
        x = p.rvs(size=(n, *self.shape[:-1]), broadcast=True)
        return (p.logpdf(x, broadcast=True) - q.logpdf(x, broadcast=True)).mean(axis=0)

    def reduce(self, D):
        """Reduce the model to a reduced model.

        Parameters
        ----------
        D : array_like, shape (..., d)
            Data to form the reduced model

        Returns
        -------
        ReducedLinearModel
        """
        dist = super().reduce(np.expand_dims(D, -2))
        dist.__class__ = ReducedMixtureModel
        logw_P = self.evidence().logpdf(D, broadcast=True, joint=True)
        logw_P = logw_P - logsumexp(logw_P, axis=-1, keepdims=True)
        dist.logw = self.logw
        dist.logw_L = logw - self.logw
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
    logLmax = - log|2 π C|/2 - (D-m)'C^{-1}(D-m)/2

    Parameters
    ----------
    logLmax : float, optional
        log likelihood maximum, defaults to zero
    μ_L : array_like
        Likelihood peak
    Σ_L : array_like
        Likelihood covariance
    diagonal_Σ_L : bool, optional
        Whether the likelihood covariance is diagonal, defaults to False
    μ : array_like, optional
        Prior mean, defaults to zero vector
    Σ : array_like, optional
        Prior covariance, defaults to identity matrix
    diagonal_Σ : bool, optional
        Whether the prior covariance is diagonal, defaults to False
    shape : (), optional
        Number of mixture components, defaults to automatically inferred value
    n : int, optional
        Number of parameters, defaults to automatically inferred value
    """

    def __init__(self, *args, **kwargs):
        self.μ_L = kwargs.pop("μ_L", 0)
        self.Σ_L = kwargs.pop("Σ_L", 1)
        self.logLmax = kwargs.pop("logLmax", 0)
        self.μ = kwargs.pop("μ", 0)
        self.Σ = kwargs.pop("Σ", 1)
        self.diagonal_Σ_L = kwargs.pop("diagonal_Σ_L", False)
        self.diagonal_Σ = kwargs.pop("diagonal_Σ", False)
        self._shape = kwargs.pop("shape", ())
        self._n = kwargs.pop("n", 1)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.μ_L)[:-1],
            np.shape(self.Σ_L)[: -2 + self.diagonal_Σ_L],
            np.shape(self.μ)[:-1],
            np.shape(self.Σ)[: -2 + self.diagonal_Σ],
            np.shape(self.logLmax),
            self._shape,
        )

    @property
    def n(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.Σ_L)[-2 + self.diagonal_Σ_L :],
                *np.shape(self.μ_L)[-1:],
                *np.shape(self.Σ)[-2 + self.diagonal_Σ :],
                *np.shape(self.μ)[-1:],
                self._n,
            ]
        )

    def prior(self):
        """P(θ) as a scipy distribution object."""
        return multivariate_normal(self.μ, self.Σ, self.shape, self.n, self.diagonal_Σ)

    def posterior(self):
        """P(θ|D) as a scipy distribution object."""
        diagonal = self.diagonal_Σ_L and self.diagonal_Σ

        if diagonal:
            Σ_P = 1 / (1 / self.Σ + 1 / self.Σ_L)
        elif self.diagonal_Σ_L:
            Σ_P = inv(inv(self.Σ) + np.eye(self.n) / self.Σ_L[..., None, :])
        elif self.diagonal_Σ:
            Σ_P = inv(np.eye(self.n) / self.Σ[..., None, :] + inv(self.Σ_L))
        else:
            Σ_P = inv(inv(self.Σ) + inv(self.Σ_L))

        if self.diagonal_Σ_L:
            x_L = self.μ_L / self.Σ_L
        else:
            x_L = np.einsum(
                "...ij,...j->...i", inv(self.Σ_L), np.ones(self.n) * self.μ_L
            )

        if self.diagonal_Σ:
            x = μ / self.Σ
        else:
            x = np.einsum("...ij,...j->...i", inv(self.Σ), np.ones(self.n) * self.μ)

        μ_P = np.einsum("...ij,...j->...i", Σ_P, x + x_L)
        return multivariate_normal(μ_P, Σ_P, self.shape, self.n, diagonal)

    def logπ(self, θ):
        """P(θ) as a scalar."""
        return self.prior().logpdf(θ)

    def logP(self, θ):
        """P(θ|D) as a scalar."""
        return self.posterior().logpdf(θ)

    def logL(self, θ):
        """P(D|θ) as a scalar."""
        dist = multivariate_normal(
            self.μ_L, self.Σ_L, self.shape, self.n, self.diagonal_Σ_L
        )
        return self.logLmax + dist.logpdf(θ) - dist.logpdf(dist.μ)

    def logZ(self):
        """P(D) as a scalar."""
        return self.logLmax + self.logπ(self.μ_L) - self.logP(self.μ_L)

    def dkl(self):
        """D_KL(P(θ|D)||P(θ)) the Kullback-Leibler divergence."""
        return dkl(self.posterior(), self.prior(), n)


class ReducedLinearModelUniformPrior(object):
    """A model with no data.

    Gaussian likelihood in the parameters

    logL(θ) = logLmax - (θ - μ_L)' Σ_L^{-1} (θ - μ_L)

    Uniform prior

    We can link this to a data-based model with the relations:

    Σ_L = (M' C^{-1} M)^{-1}
    μ_L = Σ_L M' C^{-1} (D-m)
    logLmax = -log|2 π C|/2 - (D-m)'C^{-1}(D-m)/2

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
        self.diagonal = kwargs.pop("diagonal", False)
        self._shape = kwargs.pop("shape", ())
        self._n = kwargs.pop("n", 1)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.μ_L)[:-1],
            np.shape(self.Σ_L)[: -2 + self.diagonal_Σ],
            np.shape(self.logLmax),
            np.shape(self.logV),
            self._shape,
        )

    @property
    def n(self):
        """Dimension of the distribution."""
        return np.max(
            [
                *np.shape(self.Σ_L)[-2 + self.diagonal_Σ_L :],
                *np.shape(self.μ_L)[-1:],
                self._n,
            ]
        )

    def posterior(self):
        """P(θ|D) as a distribution object."""
        return multivariate_normal(
            self.μ_L, self.Σ_L, self.shape, self.n, self.diagonal
        )

    def logπ(self, θ):
        """P(θ) as a scalar."""
        return -self.logV

    def logP(self, θ):
        """P(θ|D) as a scalar."""
        return self.posterior().logpdf(θ)

    def logL(self, θ):
        """P(D|θ) as a scalar."""
        dist = multivariate_normal(
            self.μ_L, self.Σ_L, self.shape, self.n, self.diagonal
        )
        return self.logLmax + dist.logpdf(θ) - dist.logpdf(dist.μ_L)

    def logZ(self):
        """P(D) as a scalar."""
        logZ = np.ones(self.shape) * self.logLmax
        logZ -= self.logV
        logZ += logdet(2 * np.pi * self.Σ_L, self.diagonal) / 2
        return logZ

    def dkl(self):
        """D_KL(P(θ|D)||P(θ)) the Kullback-Leibler divergence."""
        dkl = np.ones(self.shape) * self.logV
        dkl -= logdet(2 * np.pi * np.e * self.Σ_L, self.diagonal) / 2
        return dkl


class ReducedMixtureModel(ReducedLinearModel):
    """A model with no data.

    Gaussian likelihood in the parameters

    logL(θ) = logLmax - (θ - μ_L)' Σ_L^{-1} (θ - μ_L)

    We can link this to a data-based model with the relations:

    Σ_L = (M' C^{-1} M)^{-1}
    μ_L = Σ_L M' C^{-1} (D-m)
    logLmax =
    - log|2 π C|/2 - (D-m)'C^{-1}(C - M (M' C^{-1} M)^{-1} M' )C^{-1}(D-m)/2

    Parameters
    ----------
    μ_L : array_like
        Likelihood peak
    Σ_L : array_like
        Likelihood covariance
    logLmax : float, optional
        Likelihood maximum, defaults to zero
    logw : array_like, optional
        log mixture weights
        if ndim>=1: log mixture weights
        if scalar: scalar * unit vector
        Defaults to uniform weights
    logw_L: array_like, optional
        log mixture weights for the likelihood
        if ndim>=1: log mixture weights
        if scalar: scalar * unit vector
        Defaults to uniform weights
    n : int, optional
        Number of parameters, defaults to automatically inferred value
    """

    def __init__(self, *args, **kwargs):
        self.logw = kwargs.pop("logw", 0)
        self.logw_L = kwargs.pop("logw_L", 0)
        super().__init__(*args, **kwargs)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(
            np.shape(self.logw), np.shape(self.logw_L), super().shape
        )

    @property
    def k(self):
        """Number of mixture components of the distribution."""
        return self.shape[-1]

    def prior(self):
        """P(θ) as a scipy distribution object."""
        dist = super().prior()
        dist.__class__ = mixture_normal
        dist.logw = self.logw
        return dist

    def posterior(self):
        """P(θ|D) as a scipy distribution object."""
        dist = super().posterior()
        dist.__class__ = mixture_normal
        dist.logw = dist.logw + self.logw_L
        return dist


class ReducedMixtureModelUniformPrior(ReducedLinearModelUniformPrior):
    """Fill in docstring."""

    def __init__(self, *args, **kwargs):
        self.logw_L = kwargs.pop("logw_L", 0)
        super().__init__(*args, **kwargs)

    @property
    def shape(self):
        """Shape of the distribution."""
        return np.broadcast_shapes(np.shape(self.logw_L), super().shape)

    @property
    def k(self):
        """Number of mixture components of the distribution."""
        return self.shape[-1]

    def posterior(self):
        """P(θ|D) as a scipy distribution object."""
        dist = super().posterior()
        dist.__class__ = mixture_normal
        dist.logw = self.logw_L
        return dist

    def logπ(self, θ):
        """P(θ) as a scalar."""
        return -self.logV

    def logP(self, θ):
        """P(θ|D) as a scalar."""
        return self.posterior().logpdf(θ)

    def logL(self, θ):
        """P(D|θ) as a scalar."""
        pass

    def logZ(self):
        """To be implemented."""
        pass

    def dkl(self):
        """To be implemented."""
        pass
