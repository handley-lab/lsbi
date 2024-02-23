import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import invwishart

from lsbi.model import (
    LinearModel,
    MixtureModel,
    ReducedLinearModel,
    ReducedLinearModelUniformPrior,
    dediagonalise,
)


def assert_allclose_broadcast(a, b, *args, **kwargs):
    shape = np.broadcast_shapes(np.shape(a), np.shape(b))
    return assert_allclose(
        np.broadcast_to(a, shape), np.broadcast_to(b, shape), *args, **kwargs
    )


shapes = [(2, 3), (3,), ()]
dims = [1, 2, 4]
N = 1000

tests = []
for d in dims:
    for n in dims:
        for diagonal_Sigma in [True, False]:
            for diagonal_C in [True, False]:
                for diagonal_M in [True, False]:
                    for base_shape in shapes + ["scalar"]:
                        shape = base_shape
                        m_shape = base_shape
                        M_shape = base_shape
                        mu_shape = base_shape
                        C_shape = base_shape
                        Sigma_shape = base_shape
                        base_test = (
                            d,
                            n,
                            shape,
                            m_shape,
                            M_shape,
                            mu_shape,
                            C_shape,
                            Sigma_shape,
                            diagonal_Sigma,
                            diagonal_C,
                            diagonal_M,
                        )
                        for alt_shape in shapes + ["scalar"]:
                            for i in range(2, 8):
                                test = base_test[:i] + (alt_shape,) + base_test[i + 1 :]
                                if test[2] == "scalar":
                                    continue
                                tests.append(test)


def test_linear_model_init():
    with pytest.raises(ValueError):
        LinearModel(foo="bar")


@pytest.mark.parametrize(
    "d,n,shape,m_shape,M_shape,mu_shape,C_shape,Sigma_shape,diagonal_Sigma,diagonal_C,diagonal_M",
    tests,
)
class TestLinearModel(object):
    def random(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        if M_shape == "scalar":
            M = np.random.randn()
        elif diagonal_M:
            M = np.random.randn(*M_shape, n)
        else:
            M = np.random.randn(*M_shape, d, n)

        if m_shape == "scalar":
            m = np.random.randn()
        else:
            m = np.random.randn(*m_shape, d)

        if C_shape == "scalar":
            C = np.random.randn() ** 2 + d
        elif diagonal_C:
            C = np.random.randn(*C_shape, d) ** 2 + d
        else:
            C = np.random.randn(*C_shape, d, d)
            C = np.einsum("...ij,...kj->...ik", C, C) + d * np.eye(d)

        if mu_shape == "scalar":
            mu = np.random.randn()
        else:
            mu = np.random.randn(*mu_shape, n)

        if Sigma_shape == "scalar":
            Sigma = np.random.randn() ** 2 + n
        elif diagonal_Sigma:
            Sigma = np.random.randn(*Sigma_shape, n) ** 2 + n
        else:
            Sigma = np.random.randn(*Sigma_shape, n, n)
            Sigma = np.einsum("...ij,...kj->...ik", Sigma, Sigma) + n * np.eye(n)

        model = LinearModel(
            M=M,
            m=m,
            C=C,
            mu=mu,
            Sigma=Sigma,
            shape=shape,
            n=n,
            d=d,
            diagonal_M=diagonal_M,
            diagonal_C=diagonal_C,
            diagonal_Sigma=diagonal_Sigma,
        )
        assert model.d == d
        assert model.n == n
        assert np.all(model.M == M)
        assert np.all(model.m == m)
        assert np.all(model.C == C)
        assert np.all(model.mu == mu)
        assert np.all(model.Sigma == Sigma)
        assert model.diagonal_M == diagonal_M or (
            M_shape == "scalar" and model.diagonal_M
        )
        assert model.diagonal_C == diagonal_C or (
            C_shape == "scalar" and model.diagonal_C
        )
        assert model.diagonal_Sigma == diagonal_Sigma or (
            Sigma_shape == "scalar" and model.diagonal_Sigma
        )
        return model

    @pytest.mark.parametrize("theta_shape", shapes)
    def test_likelihood(
        self,
        theta_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        theta = np.random.randn(*theta_shape, n)
        likelihood = model.likelihood(theta)
        assert likelihood.shape == np.broadcast_shapes(model.shape, theta_shape)
        assert likelihood.dim == model.d

    def test_prior(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        prior = model.prior()
        assert prior.shape == model.shape
        assert prior.dim == model.n

    @pytest.mark.parametrize("D_shape", shapes)
    def test_posterior(
        self,
        D_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        D = np.random.randn(*D_shape, d)
        posterior = model.posterior(D)
        assert posterior.shape == np.broadcast_shapes(model.shape, D_shape)
        assert posterior.dim == model.n

        ppd = model.ppd(D)
        assert ppd.shape == np.broadcast_shapes(model.shape, D_shape)

        update = model.update(D)
        assert_allclose_broadcast(update.m, model.m)
        assert_allclose_broadcast(update.M, model.M)
        assert_allclose_broadcast(update.C, model.C)
        assert_allclose_broadcast(update.mu, posterior.mean)
        assert_allclose_broadcast(update.Sigma, posterior.cov)

        model.update(D, inplace=True)
        assert_allclose(model.m, update.m)
        assert_allclose(model.M, update.M)
        assert_allclose(model.C, update.C)
        assert_allclose(model.mu, update.mu)
        assert_allclose(model.Sigma, update.Sigma)

        dkl = model.dkl(D)
        assert dkl.shape == model.shape
        assert (dkl >= 0).all()

    def test_evidence(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        evidence = model.evidence()
        assert evidence.shape == model.shape
        assert evidence.dim == model.d

    def test_joint(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        joint = model.joint()
        assert joint.shape == model.shape
        assert joint.dim == model.n + model.d

    def test_marginal_conditional(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        atol = 1e-5

        i = np.arange(d + n)[:n]
        model_1 = model.evidence()
        model_2 = model.joint().marginalise(i)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        theta = model.prior().rvs()
        model_1 = model.likelihood(theta)
        model_2 = model.joint().condition(i, theta)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        i = np.arange(d + n)[-d:]
        model_1 = model.prior()
        model_2 = model.joint().marginalise(i)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        D = model.evidence().rvs()
        model_1 = model.posterior(D)
        model_2 = model.joint().condition(i, D)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

    def test_bayes_theorem(
        self,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        atol = 1e-5

        model = self.random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )

        theta_D = model.joint().rvs()
        theta, D = np.split(theta_D, [model.n], axis=-1)
        assert_allclose(
            model.posterior(D).logpdf(theta, broadcast=True)
            + model.evidence().logpdf(D, broadcast=True),
            model.likelihood(theta).logpdf(D, broadcast=True)
            + model.prior().logpdf(theta, broadcast=True),
            atol=atol,
        )


@pytest.mark.parametrize("logw_shape", shapes)
class TestMixtureModel(TestLinearModel):
    def random(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = super().random(
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        logw = np.random.randn(*logw_shape)
        model = MixtureModel(
            logw=logw,
            M=model.M,
            m=model.m,
            C=model.C,
            mu=model.mu,
            Sigma=model.Sigma,
            shape=shape,
            n=n,
            d=d,
            diagonal_M=diagonal_M,
            diagonal_C=diagonal_C,
            diagonal_Sigma=diagonal_Sigma,
        )
        assert np.all(model.logw == logw)
        if model.shape:
            assert model.k == model.shape[-1]
        else:
            assert model.k == 1
        return model

    @pytest.mark.parametrize("theta_shape", shapes)
    def test_likelihood(
        self,
        theta_shape,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        theta = np.random.randn(*theta_shape[:-1], n)
        likelihood = model.likelihood(theta)
        if model.shape != ():
            assert likelihood.shape == np.broadcast_shapes(model.shape, theta_shape)
        assert likelihood.dim == model.d

    def test_prior(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        prior = model.prior()
        assert prior.shape == model.shape
        assert prior.dim == model.n

    @pytest.mark.parametrize("D_shape", shapes)
    def test_posterior(
        self,
        D_shape,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        D = np.random.randn(*D_shape[:-1], d)
        posterior = model.posterior(D)
        if model.shape != ():
            assert posterior.shape == np.broadcast_shapes(model.shape, D_shape)
        assert posterior.dim == model.n

        ppd = model.ppd(D)
        if model.shape != ():
            assert ppd.shape == np.broadcast_shapes(model.shape, D_shape)

        update = model.update(D)
        assert_allclose_broadcast(update.m, model.m)
        assert_allclose_broadcast(update.M, model.M)
        assert_allclose_broadcast(update.C, model.C)
        assert_allclose_broadcast(update.mu, posterior.mean)
        assert_allclose_broadcast(update.Sigma, posterior.cov)

        model.update(D, inplace=True)
        assert_allclose(model.m, update.m)
        assert_allclose(model.M, update.M)
        assert_allclose(model.C, update.C)
        assert_allclose(model.mu, update.mu)
        assert_allclose(model.Sigma, update.Sigma)

        with pytest.raises(ValueError):
            model.dkl(D)

        dkl = model.dkl(D, 10)
        assert dkl.shape == model.shape[:-1]

    def test_evidence(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        evidence = model.evidence()
        assert evidence.shape == model.shape
        assert evidence.dim == model.d

    def test_joint(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        joint = model.joint()
        assert joint.shape == model.shape
        assert joint.dim == model.n + model.d

    def test_marginal_conditional(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )

        atol = 1e-5

        i = np.arange(n + d)[:n]
        model_1 = model.evidence()
        model_2 = model.joint().marginalise(i)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        theta = model.prior().rvs()
        model_1 = model.likelihood(theta)
        model_2 = model.joint().condition(i, theta)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        i = np.arange(n + d)[-d:]
        model_1 = model.prior()
        model_2 = model.joint().marginalise(i)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

        D = model.evidence().rvs()
        model_1 = model.posterior(D)
        model_2 = model.joint().condition(i, D)
        assert_allclose_broadcast(model_1.mean, model_2.mean, atol=atol)
        assert_allclose_broadcast(
            dediagonalise(model_1.cov, model_1.diagonal, model_1.dim),
            model_2.cov,
            atol=atol,
        )

    def test_bayes_theorem(
        self,
        logw_shape,
        M_shape,
        diagonal_M,
        m_shape,
        C_shape,
        diagonal_C,
        mu_shape,
        Sigma_shape,
        diagonal_Sigma,
        shape,
        n,
        d,
    ):
        atol = 1e-5

        model = self.random(
            logw_shape,
            M_shape,
            diagonal_M,
            m_shape,
            C_shape,
            diagonal_C,
            mu_shape,
            Sigma_shape,
            diagonal_Sigma,
            shape,
            n,
            d,
        )
        theta_D = model.joint().rvs()
        theta, D = np.split(theta_D, [model.n], axis=-1)
        assert_allclose(
            model.posterior(D).logpdf(theta, broadcast=True)
            + model.evidence().logpdf(D, broadcast=True),
            model.likelihood(theta).logpdf(D, broadcast=True)
            + model.prior().logpdf(theta, broadcast=True),
            atol=atol,
        )


@pytest.mark.parametrize("n", np.arange(1, 6))
class TestReducedLinearModel(object):
    def random(self, n):
        mu_pi = np.random.randn(n)
        Sigma_pi = invwishart(scale=np.eye(n)).rvs()
        mu_L = np.random.randn(n)
        Sigma_L = invwishart(scale=np.eye(n)).rvs()
        logLmax = np.random.randn()

        return ReducedLinearModel(
            mu_pi=mu_pi, Sigma_pi=Sigma_pi, logLmax=logLmax, mu_L=mu_L, Sigma_L=Sigma_L
        )

    def test_bayes_theorem(self, n):
        model = self.random(n)
        theta = model.prior().rvs()
        assert_allclose(
            model.logP(theta) + model.logZ(), model.logL(theta) + model.logpi(theta)
        )


@pytest.mark.parametrize("n", np.arange(1, 6))
class TestReducedLinearModelUniformPrior(object):
    def random(self, n):
        mu_L = np.random.randn(n)
        Sigma_L = invwishart(scale=np.eye(n)).rvs()
        logLmax = np.random.randn()
        logV = np.random.randn()

        return ReducedLinearModelUniformPrior(
            logLmax=logLmax, logV=logV, mu_L=mu_L, Sigma_L=Sigma_L
        )

    def test_model(self, n):
        model = self.random(n)
        theta = model.posterior().rvs(N)
        assert_allclose(
            model.logpi(theta) + model.logL(theta), model.logP(theta) + model.logZ()
        )

        logV = 50
        Sigma_pi = np.exp(2 * logV / n) / (2 * np.pi) * np.eye(n)

        reduced_model = ReducedLinearModel(
            logLmax=model.logLmax,
            mu_L=model.mu_L,
            Sigma_L=model.Sigma_L,
            Sigma_pi=Sigma_pi,
        )

        model = ReducedLinearModelUniformPrior(
            logLmax=model.logLmax, mu_L=model.mu_L, Sigma_L=model.Sigma_L, logV=logV
        )

        assert_allclose(reduced_model.logZ(), model.logZ())
        assert_allclose(reduced_model.DKL(), model.DKL())

    def test_bayes_theorem(self, n):
        model = self.random(n)
        theta = model.posterior().rvs()
        assert_allclose(
            model.logP(theta) + model.logZ(), model.logL(theta) + model.logpi(theta)
        )
