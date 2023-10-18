from lsbi.model import (LinearModel,
                        ReducedLinearModel,
                        ReducedLinearModelUniformPrior)

import numpy as np
import scipy.stats
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("n", np.arange(1, 6))
@pytest.mark.parametrize("d", np.arange(1, 6))
class TestLinearModel(object):

    def random_model(self, d, n):
        M = np.random.rand(d, n)
        m = np.random.rand(d)
        C = scipy.stats.wishart(scale=np.eye(d)).rvs()
        mu = np.random.rand(n)
        Sigma = scipy.stats.wishart(scale=np.eye(n)).rvs()
        return LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    def _test_shape(self, model, d, n):
        assert model.n == n
        assert model.d == d
        assert model.M.shape == (d, n)
        assert model.m.shape == (d,)
        assert model.C.shape == (d, d)
        assert model.mu.shape == (n,)
        assert model.Sigma.shape == (n, n)

    def test_init_M(self, d, n):
        model = LinearModel(M=np.random.rand())
        self._test_shape(model, 1, 1)

        model = LinearModel(M=np.random.rand(n))
        self._test_shape(model, 1, n)

        model = LinearModel(M=np.random.rand(d, n))
        self._test_shape(model, d, n)

    def test_init_m_mu(self, d, n):
        model = LinearModel(m=np.random.rand(), mu=np.random.rand())
        self._test_shape(model, 1, 1)

        model = LinearModel(m=np.random.rand(), mu=np.random.rand(n))
        self._test_shape(model, 1, n)

        model = LinearModel(m=np.random.rand(d), mu=np.random.rand())
        self._test_shape(model, d, 1)

        model = LinearModel(m=np.random.rand(d), mu=np.random.rand(n))
        self._test_shape(model, d, n)

    def test_failure(self, d, n):
        with pytest.raises(ValueError) as excinfo:
            LinearModel(m=np.random.rand(5))
        string = "Unable to determine number of parameters n"
        assert string in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            LinearModel(mu=np.random.rand(3))
        string = "Unable to determine data dimensions d"
        assert string in str(excinfo.value)

    def test_joint(self, d, n):
        N = 100
        model = self.random_model(d, n)
        prior = model.prior()
        evidence = model.evidence()
        joint = model.joint()

        samples_1 = prior.rvs(N)
        samples_2 = joint.rvs(N)[:, -n:]

        if n == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(n):
            p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = scipy.stats.kstest(prior.logpdf(samples_2),
                               prior.logpdf(samples_1)).pvalue
        assert p > 1e-5

        samples_1 = evidence.rvs(N)
        samples_2 = joint.rvs(N)[:, :d]

        if d == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(d):
            p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = scipy.stats.kstest(evidence.logpdf(samples_2),
                               evidence.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_likelihood_posterior(self, d, n):
        N = 100
        model = self.random_model(d, n)
        joint = model.joint()

        samples = []
        model.prior()
        theta = np.atleast_1d(model.prior().rvs())
        for _ in range(N):
            data = np.atleast_1d(model.likelihood(theta).rvs())
            theta = np.atleast_1d(model.posterior(data).rvs())
            samples.append(np.concatenate([data, theta])[:])
        samples_1 = np.array(samples)[::10]
        samples_2 = joint.rvs(len(samples_1))

        for i in range(n+d):
            p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = scipy.stats.kstest(joint.logpdf(samples_2),
                               joint.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_DKL(self, d, n):
        N = 1000
        model = self.random_model(d, n)

        data = model.evidence().rvs()
        posterior = model.posterior(data)
        prior = model.prior()

        samples = posterior.rvs(N)
        Info = (posterior.logpdf(samples) - prior.logpdf(samples))
        assert_allclose(Info.mean(), model.DKL(data),
                        atol=5*Info.std()/np.sqrt(N))


def test_reduce():
    model = LinearModel(M=np.random.rand(5, 3))
    data = model.evidence().rvs()
    reduced_model = model.reduce(data)
    assert isinstance(reduced_model, ReducedLinearModel)
    reduced_model.prior().mean
    assert_allclose(reduced_model.prior().mean, model.prior().mean)
    assert_allclose(reduced_model.prior().cov, model.prior().cov)
    assert_allclose(reduced_model.posterior().mean, model.posterior(data).mean)
    assert_allclose(reduced_model.posterior().cov, model.posterior(data).cov)
    assert_allclose(model.evidence().logpdf(data), reduced_model.logZ())
    assert_allclose(model.DKL(data), reduced_model.DKL())


@pytest.mark.parametrize("n", np.arange(1, 6))
class TestReducedLinearModel(object):
    def random_model(self, n):
        mu_pi = np.random.randn(n)
        Sigma_pi = scipy.stats.wishart(scale=np.eye(n)).rvs()
        mu_L = np.random.randn(n)
        Sigma_L = scipy.stats.wishart(scale=np.eye(n)).rvs()
        logLmax = np.random.randn()

        return ReducedLinearModel(mu_pi=mu_pi, Sigma_pi=Sigma_pi,
                                  logLmax=logLmax,
                                  mu_L=mu_L, Sigma_L=Sigma_L)

    def test_model(self, n):
        model = self.random_model(n)
        theta = model.posterior().rvs(1000)
        assert_allclose(model.logpi(theta) + model.logL(theta),
                        model.logP(theta) + model.logZ())


@pytest.mark.parametrize("n", np.arange(1, 6))
class TestReducedLinearModelUniformPrior(object):
    def random_model(self, n):
        mu_L = np.random.randn(n)
        Sigma_L = scipy.stats.wishart(scale=np.eye(n)).rvs()
        logLmax = np.random.randn()
        logV = np.random.randn()

        return ReducedLinearModelUniformPrior(logLmax=logLmax, logV=logV,
                                              mu_L=mu_L, Sigma_L=Sigma_L)

    def test_model(self, n):
        model = self.random_model(n)
        theta = model.posterior().rvs(1000)
        assert_allclose(model.logpi(theta) + model.logL(theta),
                        model.logP(theta) + model.logZ())

        logV = 50
        Sigma_pi = np.exp(2*logV/n)/(2*np.pi)*np.eye(n)

        reduced_model = ReducedLinearModel(logLmax=model.logLmax,
                                           mu_L=model.mu_L,
                                           Sigma_L=model.Sigma_L,
                                           Sigma_pi=Sigma_pi)

        model = ReducedLinearModelUniformPrior(logLmax=model.logLmax,
                                               mu_L=model.mu_L,
                                               Sigma_L=model.Sigma_L,
                                               logV=logV)

        assert_allclose(reduced_model.logZ(), model.logZ())
        assert_allclose(reduced_model.DKL(), model.DKL())
