import numpy as np
import pytest
from numpy.random import rand
from numpy.testing import assert_allclose
from scipy.stats import invwishart, kstest

from lsbi.model import (
    LinearMixtureModel,
    LinearModel,
    ReducedLinearModel,
    ReducedLinearModelUniformPrior,
)

N = 1000


@pytest.mark.parametrize("n", [1, 2, 5, 10])
@pytest.mark.parametrize("d", [1, 2, 5, 10])
class TestLinearModel(object):
    cls = LinearModel

    def random(self, d, n):
        M = rand(d, n)
        m = rand(d)
        C = invwishart(scale=np.eye(d), df=10 * d).rvs()
        mu = rand(n)
        Sigma = invwishart(scale=np.eye(n), df=10 * n).rvs()
        return self.cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    def _test_shape(self, model, d, n):
        assert model.n == n
        assert model.d == d
        assert model.M.shape == (d, n)
        assert model.m.shape == (d,)
        assert model.C.shape == (d, d)
        assert model.mu.shape == (n,)
        assert model.Sigma.shape == (n, n)

    def test_init_M(self, d, n):
        self._test_shape(self.cls(M=rand()), 1, 1)
        self._test_shape(self.cls(M=rand(), n=n), 1, n)
        self._test_shape(self.cls(M=rand(), d=d), d, 1)
        self._test_shape(self.cls(M=rand(), d=d, n=n), d, n)
        self._test_shape(self.cls(M=rand(n)), 1, n)
        self._test_shape(self.cls(M=rand(n), d=d), d, n)
        self._test_shape(self.cls(M=rand(d, 1)), d, 1)
        self._test_shape(self.cls(M=rand(d, 1), n=n), d, n)
        self._test_shape(self.cls(M=rand(d, n)), d, n)

        M = rand()
        model = self.cls(M=M, d=d, n=n)
        assert_allclose(np.diag(model.M), M)

        M = rand(n)
        model = self.cls(M=M, d=d)
        assert_allclose(np.diag(model.M), M[: min(d, n)])

        M = rand(d, n)
        model = self.cls(M=M)
        assert_allclose(model.M, M)

    def test_init_mu(self, d, n):
        self._test_shape(self.cls(mu=rand(), d=d), d, 1)
        self._test_shape(self.cls(mu=rand(), d=d, n=n), d, n)
        self._test_shape(self.cls(mu=rand(n), d=d), d, n)

        mu = rand()
        model = self.cls(mu=mu, d=d, n=n)
        assert_allclose(model.mu, mu)

        mu = rand(n)
        model = self.cls(mu=mu, d=d)
        assert_allclose(model.mu, mu)

    def test_init_Sigma(self, d, n):
        self._test_shape(self.cls(Sigma=rand(), d=d), d, 1)
        self._test_shape(self.cls(Sigma=rand(), d=d, n=n), d, n)
        self._test_shape(self.cls(Sigma=rand(n), d=d), d, n)
        self._test_shape(self.cls(Sigma=rand(n, n), d=d), d, n)

        Sigma = rand()
        model = self.cls(Sigma=Sigma, d=d, n=n)
        assert_allclose(np.diag(model.Sigma), Sigma)

        Sigma = rand(n)
        model = self.cls(Sigma=Sigma, d=d)
        assert_allclose(np.diag(model.Sigma), Sigma)

        Sigma = rand(n, n)
        model = self.cls(Sigma=Sigma, d=d)
        assert_allclose(model.Sigma, Sigma)

    def test_init_m(self, d, n):
        self._test_shape(self.cls(m=rand(), n=n), 1, n)
        self._test_shape(self.cls(m=rand(), d=d, n=n), d, n)
        self._test_shape(self.cls(m=rand(d), n=n), d, n)

        m = rand()
        model = self.cls(m=m, d=d, n=n)
        assert_allclose(model.m, m)

        m = rand(d)
        model = self.cls(m=m, n=n)
        assert_allclose(model.m, m)

    def test_init_C(self, d, n):
        self._test_shape(self.cls(C=rand(), n=n), 1, n)
        self._test_shape(self.cls(C=rand(), d=d, n=n), d, n)
        self._test_shape(self.cls(C=rand(d), n=n), d, n)
        self._test_shape(self.cls(C=rand(d, d), n=n), d, n)

        C = rand()
        model = self.cls(C=C, d=d, n=n)
        assert_allclose(np.diag(model.C), C)

        C = rand(d)
        model = self.cls(C=C, n=n)
        assert_allclose(np.diag(model.C), C)

        C = rand(d, d)
        model = self.cls(C=C, n=n)
        assert_allclose(model.C, C)

    def test_failure(self, d, n):
        with pytest.raises(ValueError) as excinfo:
            self.cls(m=rand(d))
        string = "Unable to determine number of parameters n"
        assert string in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            self.cls(mu=rand(n))
        string = "Unable to determine data dimensions d"
        assert string in str(excinfo.value)

    def test_joint(self, d, n):
        model = self.random(d, n)
        prior = model.prior()
        evidence = model.evidence()
        joint = model.joint()

        samples_1 = prior.rvs(N)
        samples_2 = joint.rvs(N)[:, -n:]

        if n == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(n):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(prior.logpdf(samples_2), prior.logpdf(samples_1)).pvalue
        assert p > 1e-5

        samples_1 = evidence.rvs(N)
        samples_2 = joint.rvs(N)[:, :d]

        if d == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(d):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(evidence.logpdf(samples_2), evidence.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_likelihood_posterior(self, d, n):
        model = self.random(d, n)
        joint = model.joint()

        samples = []
        model.prior()
        theta = model.prior().rvs()
        for _ in range(N):
            data = np.atleast_1d(model.likelihood(theta).rvs())
            theta = np.atleast_1d(model.posterior(data).rvs())
            samples.append(np.concatenate([data, theta])[:])
        samples_1 = np.array(samples)[::100]
        samples_2 = joint.rvs(len(samples_1))

        for i in range(n + d):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(joint.logpdf(samples_2), joint.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_DKL(self, d, n):
        model = self.random(d, n)

        data = model.evidence().rvs()
        posterior = model.posterior(data)
        prior = model.prior()

        samples = posterior.rvs(N)
        Info = posterior.logpdf(samples) - prior.logpdf(samples)
        assert_allclose(Info.mean(), model.DKL(data), atol=5 * Info.std() / np.sqrt(N))

    def test_from_joint(self, d, n):
        model = self.random(d, n)
        joint = model.joint()
        mean = joint.mean
        cov = joint.cov
        model2 = self.cls.from_joint(mean, cov, n)
        assert model2.n == model.n
        assert model2.d == model.d
        assert_allclose(model2.M, model.M)
        assert_allclose(model2.m, model.m)
        assert_allclose(model2.C, model.C)
        assert_allclose(model2.mu, model.mu)
        assert_allclose(model2.Sigma, model.Sigma)

    def test_reduce(self, d, n):
        if n > d:
            pytest.skip("n > d")
        model = self.cls(M=rand(d, n))
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

    def test_marginal_conditional(self, d, n):
        model = self.random(d, n)
        i = np.arange(d + n)[-n:]
        model_1 = model.evidence()
        model_2 = model.joint().marginalise(i)
        assert_allclose(model_1.mean, model_2.mean)
        assert_allclose(model_1.cov, model_2.cov, rtol=1e-6, atol=1e-6)

        theta = model.prior().rvs()
        model_1 = model.likelihood(theta)
        model_2 = model.joint().condition(i, theta)
        assert_allclose(model_1.mean, model_2.mean)
        assert_allclose(model_1.cov, model_2.cov, rtol=1e-6, atol=1e-6)

        i = np.arange(d + n)[:d]
        model_1 = model.prior()
        model_2 = model.joint().marginalise(i)
        assert_allclose(model_1.mean, model_2.mean)
        assert_allclose(model_1.cov, model_2.cov, rtol=1e-6, atol=1e-6)

        D = model.evidence().rvs()
        model_1 = model.posterior(D)
        model_2 = model.joint().condition(i, D)
        assert_allclose(model_1.mean, model_2.mean)
        assert_allclose(model_1.cov, model_2.cov, rtol=1e-6, atol=1e-6)

    def test_bayes_theorem(self, d, n):
        model = self.random(d, n)
        theta = model.prior().rvs()
        D = model.evidence().rvs()
        assert_allclose(
            model.posterior(D).logpdf(theta) + model.evidence().logpdf(D),
            model.likelihood(theta).logpdf(D) + model.prior().logpdf(theta),
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


@pytest.mark.parametrize("k", np.arange(1, 6))
@pytest.mark.parametrize("d", np.arange(1, 6))
@pytest.mark.parametrize("n", np.arange(1, 6))
class TestLinearMixtureModel(object):
    cls = LinearMixtureModel

    def random(self, k, d, n):
        M = rand(k, d, n)
        m = rand(k, d)
        C = np.array(
            [
                np.atleast_2d(invwishart(scale=np.eye(d), df=d * 10).rvs())
                for _ in range(k)
            ]
        )

        mu = rand(k, n)
        Sigma = np.array(
            [
                np.atleast_2d(invwishart(scale=np.eye(n), df=d * 10).rvs())
                for _ in range(k)
            ]
        )
        logA = np.log(rand(k))
        return self.cls(M=M, m=m, C=C, mu=mu, Sigma=Sigma, logA=logA)

    def _test_shape(self, model, k, d, n):
        assert model.n == n
        assert model.d == d
        assert model.k == k
        assert model.M.shape == (k, d, n)
        assert model.m.shape == (
            k,
            d,
        )
        assert model.C.shape == (k, d, d)
        assert model.mu.shape == (
            k,
            n,
        )
        assert model.Sigma.shape == (k, n, n)
        assert model.logA.shape == (k,)

    def test_init_M(self, k, d, n):
        self._test_shape(self.cls(M=rand()), 1, 1, 1)
        self._test_shape(self.cls(M=rand(), n=n), 1, 1, n)
        self._test_shape(self.cls(M=rand(), d=d), 1, d, 1)
        self._test_shape(self.cls(M=rand(), k=k), k, 1, 1)
        self._test_shape(self.cls(M=rand(), d=d, n=n), 1, d, n)
        self._test_shape(self.cls(M=rand(), k=k, n=n), k, 1, n)
        self._test_shape(self.cls(M=rand(), k=k, d=d), k, d, 1)
        self._test_shape(self.cls(M=rand(), k=k, d=d, n=n), k, d, n)
        self._test_shape(self.cls(M=rand(n)), 1, 1, n)
        self._test_shape(self.cls(M=rand(n), d=d), 1, d, n)
        self._test_shape(self.cls(M=rand(n), k=k), k, 1, n)
        self._test_shape(self.cls(M=rand(n), k=k, d=d), k, d, n)

        self._test_shape(self.cls(M=rand(d, 1)), 1, d, 1)
        self._test_shape(self.cls(M=rand(d, 1), k=k), k, d, 1)
        self._test_shape(self.cls(M=rand(d, 1), n=n), 1, d, n)
        self._test_shape(self.cls(M=rand(d, 1), k=k, n=n), k, d, n)

        self._test_shape(self.cls(M=rand(k, 1, 1)), k, 1, 1)
        self._test_shape(self.cls(M=rand(k, 1, 1), d=d), k, d, 1)
        self._test_shape(self.cls(M=rand(k, 1, 1), n=n), k, 1, n)
        self._test_shape(self.cls(M=rand(k, 1, 1), d=d, n=n), k, d, n)

        self._test_shape(self.cls(M=rand(k, d, 1)), k, d, 1)
        self._test_shape(self.cls(M=rand(k, d, 1), n=n), k, d, n)

        self._test_shape(self.cls(M=rand(k, 1, n)), k, 1, n)
        self._test_shape(self.cls(M=rand(k, 1, n), d=d), k, d, n)

        self._test_shape(self.cls(M=rand(1, d, n)), 1, d, n)
        self._test_shape(self.cls(M=rand(1, d, n), k=k), k, d, n)

        self._test_shape(self.cls(M=rand(d, n)), 1, d, n)
        self._test_shape(self.cls(M=rand(d, n), k=k), k, d, n)

        self._test_shape(self.cls(M=rand(k, d, n)), k, d, n)

        M = rand()
        model = self.cls(M=M, d=d, n=n)
        assert_allclose(np.diag(model.M[0]), M)

        M = rand(n)
        model = self.cls(M=M, d=d)
        assert_allclose(np.diag(model.M[0]), M[: min(d, n)])

        M = rand(d, n)
        model = self.cls(M=M)
        assert_allclose(model.M[0], M)

        M = rand()
        model = self.cls(M=M, k=k, d=d, n=n)
        for M_ in model.M:
            assert_allclose(M_, model.M[0])
            assert_allclose(np.diag(M_), M)

        M = rand(n)
        model = self.cls(M=M, d=d)
        for M_ in model.M:
            assert_allclose(M_, model.M[0])
            assert_allclose(np.diag(M_), M[: min(d, n)])

        M = rand(d, n)
        model = self.cls(M=M)
        for M_ in model.M:
            assert_allclose(M_, model.M[0])
            assert_allclose(M_, M)

        M = rand(k, d, n)
        model = self.cls(M=M)
        assert_allclose(model.M, M)

    def test_init_mu(self, k, d, n):
        self._test_shape(self.cls(mu=rand(), d=d), 1, d, 1)
        self._test_shape(self.cls(mu=rand(), k=k, d=d), k, d, 1)
        self._test_shape(self.cls(mu=rand(), d=d, n=n), 1, d, n)
        self._test_shape(self.cls(mu=rand(), k=k, d=d, n=n), k, d, n)
        self._test_shape(self.cls(mu=rand(n), d=d), 1, d, n)
        self._test_shape(self.cls(mu=rand(n), k=k, d=d), k, d, n)
        self._test_shape(self.cls(mu=rand(k, n), d=d), k, d, n)

        mu = rand()
        model = self.cls(mu=mu, d=d, n=n)
        assert_allclose(model.mu, mu)

        mu = rand(n)
        model = self.cls(mu=mu, d=d)
        assert_allclose(model.mu[0], mu)

        mu = rand()
        model = self.cls(mu=mu, k=k, d=d, n=n)
        assert_allclose(model.mu, mu)

        mu = rand(n)
        model = self.cls(mu=mu, k=k, d=d)
        for mu_ in model.mu:
            assert_allclose(mu_, mu)

        mu = rand(k, n)
        model = self.cls(mu=mu, d=d)
        assert_allclose(model.mu, mu)

    def test_init_Sigma(self, k, d, n):
        self._test_shape(self.cls(Sigma=rand(), d=d), 1, d, 1)
        self._test_shape(self.cls(Sigma=rand(), k=k, d=d), k, d, 1)
        self._test_shape(self.cls(Sigma=rand(), d=d, n=n), 1, d, n)
        self._test_shape(self.cls(Sigma=rand(), k=k, d=d, n=n), k, d, n)
        self._test_shape(self.cls(Sigma=rand(n), d=d), 1, d, n)
        self._test_shape(self.cls(Sigma=rand(n), k=k, d=d), k, d, n)
        self._test_shape(self.cls(Sigma=rand(n, n), d=d), 1, d, n)
        self._test_shape(self.cls(Sigma=rand(n, n), k=k, d=d), k, d, n)
        self._test_shape(self.cls(Sigma=rand(k, n, n), d=d), k, d, n)

        Sigma = rand()
        model = self.cls(Sigma=Sigma, d=d, n=n)
        assert_allclose(np.diag(model.Sigma[0]), Sigma)

        Sigma = rand(n)
        model = self.cls(Sigma=Sigma, d=d)
        assert_allclose(np.diag(model.Sigma[0]), Sigma)

        Sigma = rand(n, n)
        model = self.cls(Sigma=Sigma, d=d)
        assert_allclose(model.Sigma[0], Sigma)

        Sigma = rand()
        model = self.cls(Sigma=Sigma, k=k, d=d, n=n)
        for Sigma_ in model.Sigma:
            assert_allclose(np.diag(Sigma_), Sigma)

        Sigma = rand(n)
        model = self.cls(Sigma=Sigma, k=k, d=d)
        for Sigma_ in model.Sigma:
            assert_allclose(np.diag(Sigma_), Sigma)

        Sigma = rand(n, n)
        model = self.cls(Sigma=Sigma, k=k, d=d)
        for Sigma_ in model.Sigma:
            assert_allclose(Sigma_, Sigma)

        Sigma = rand(k, n, n)
        model = self.cls(Sigma=Sigma, d=d)
        assert_allclose(model.Sigma, Sigma)

    def test_init_m(self, k, d, n):
        self._test_shape(self.cls(m=rand(), n=n), 1, 1, n)
        self._test_shape(self.cls(m=rand(), k=k, n=n), k, 1, n)
        self._test_shape(self.cls(m=rand(), n=n, d=d), 1, d, n)
        self._test_shape(self.cls(m=rand(), k=k, n=n, d=d), k, d, n)
        self._test_shape(self.cls(m=rand(d), n=n), 1, d, n)
        self._test_shape(self.cls(m=rand(d), k=k, n=n), k, d, n)
        self._test_shape(self.cls(m=rand(k, d), n=n), k, d, n)

        m = rand()
        model = self.cls(m=m, d=d, n=n)
        assert_allclose(model.m, m)

        m = rand(d)
        model = self.cls(m=m, n=n)
        assert_allclose(model.m[0], m)

        m = rand()
        model = self.cls(m=m, k=k, d=d, n=n)
        assert_allclose(model.m, m)

        m = rand(d)
        model = self.cls(m=m, k=k, n=n)
        for mu_ in model.m:
            assert_allclose(mu_, m)

        m = rand(k, d)
        model = self.cls(m=m, n=n)
        assert_allclose(model.m, m)

    def test_init_C(self, k, d, n):
        self._test_shape(self.cls(C=rand(), n=n), 1, 1, n)
        self._test_shape(self.cls(C=rand(), k=k, n=n), k, 1, n)
        self._test_shape(self.cls(C=rand(), n=n, d=d), 1, d, n)
        self._test_shape(self.cls(C=rand(), k=k, n=n, d=d), k, d, n)
        self._test_shape(self.cls(C=rand(d), n=n), 1, d, n)
        self._test_shape(self.cls(C=rand(d), k=k, n=n), k, d, n)
        self._test_shape(self.cls(C=rand(d, d), n=n), 1, d, n)
        self._test_shape(self.cls(C=rand(d, d), k=k, n=n), k, d, n)
        self._test_shape(self.cls(C=rand(k, d, d), n=n), k, d, n)

        C = rand()
        model = self.cls(C=C, d=d, n=n)
        assert_allclose(np.diag(model.C[0]), C)

        C = rand(d)
        model = self.cls(C=C, n=n)
        assert_allclose(np.diag(model.C[0]), C)

        C = rand(d, d)
        model = self.cls(C=C, n=n)
        assert_allclose(model.C[0], C)

        C = rand()
        model = self.cls(C=C, k=k, d=d, n=n)
        for Sigma_ in model.C:
            assert_allclose(np.diag(Sigma_), C)

        C = rand(d)
        model = self.cls(C=C, k=k, n=n)
        for Sigma_ in model.C:
            assert_allclose(np.diag(Sigma_), C)

        C = rand(d, d)
        model = self.cls(C=C, k=k, n=n)
        for Sigma_ in model.C:
            assert_allclose(Sigma_, C)

        C = rand(k, d, d)
        model = self.cls(C=C, n=n)
        assert_allclose(model.C, C)

    def test_init_logA(self, k, d, n):
        self._test_shape(self.cls(logA=rand(), d=d, n=n), 1, d, n)
        self._test_shape(self.cls(logA=rand(), k=k, d=d, n=n), k, d, n)
        self._test_shape(self.cls(logA=rand(k), d=d, n=n), k, d, n)

        logA = rand()
        model = self.cls(logA=logA, d=d, n=n)
        assert_allclose(model.logA, logA)

        logA = rand(k)
        model = self.cls(logA=logA, d=d, n=n)
        assert_allclose(model.logA, logA)

    def test_failure(self, k, d, n):
        with pytest.raises(ValueError) as excinfo:
            self.cls(m=rand(d))
        string = "Unable to determine number of parameters n"
        assert string in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            self.cls(mu=rand(n))
        string = "Unable to determine data dimensions d"
        assert string in str(excinfo.value)

    def test_joint(self, k, d, n):
        model = self.random(k, d, n)
        prior = model.prior()
        evidence = model.evidence()
        joint = model.joint()

        samples_1 = prior.rvs(N)
        samples_2 = joint.rvs(N)[:, -n:]

        if n == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(n):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(prior.logpdf(samples_2), prior.logpdf(samples_1)).pvalue
        assert p > 1e-5

        samples_1 = evidence.rvs(N)
        samples_2 = joint.rvs(N)[:, :d]

        if d == 1:
            samples_1 = np.atleast_2d(samples_1).T

        for i in range(d):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(evidence.logpdf(samples_2), evidence.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_likelihood_posterior(self, k, d, n):
        model = self.random(k, d, n)
        joint = model.joint()

        samples = []
        model.prior()
        theta = model.prior().rvs()
        for _ in range(N):
            data = np.atleast_1d(model.likelihood(theta).rvs())
            theta = np.atleast_1d(model.posterior(data).rvs())
            samples.append(np.concatenate([data, theta])[:])
        samples_1 = np.array(samples)[::100]
        samples_2 = joint.rvs(len(samples_1))

        for i in range(n + d):
            p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(joint.logpdf(samples_2), joint.logpdf(samples_1)).pvalue
        assert p > 1e-5

    def test_from_joint(self, k, d, n):
        model = self.random(k, d, n)
        joint = model.joint()
        means = joint.means
        covs = joint.covs
        logA = joint.logA
        model2 = self.cls.from_joint(means, covs, logA, n)
        assert model2.n == model.n
        assert model2.d == model.d
        assert_allclose(model2.M, model.M)
        assert_allclose(model2.m, model.m)
        assert_allclose(model2.C, model.C)
        assert_allclose(model2.mu, model.mu)
        assert_allclose(model2.Sigma, model.Sigma)
        assert_allclose(model2.logA, model.logA)

    def test_marginal_conditional(self, k, d, n):
        model = self.random(k, d, n)
        i = np.arange(d + n)[-n:]
        model_1 = model.evidence()
        model_2 = model.joint().marginalise(i)
        assert_allclose(model_1.means, model_2.means)
        assert_allclose(model_1.covs, model_2.covs, rtol=1e-6, atol=1e-6)
        assert_allclose(model_1.logA, model_2.logA)

        theta = model.prior().rvs()
        model_1 = model.likelihood(theta)
        model_2 = model.joint().condition(i, theta)
        assert_allclose(model_1.means, model_2.means)
        assert_allclose(model_1.covs, model_2.covs, rtol=1e-6, atol=1e-6)
        assert_allclose(model_1.logA, model_2.logA)

        i = np.arange(d + n)[:d]
        model_1 = model.prior()
        model_2 = model.joint().marginalise(i)
        assert_allclose(model_1.means, model_2.means)
        assert_allclose(model_1.covs, model_2.covs, rtol=1e-6, atol=1e-6)
        assert_allclose(model_1.logA, model_2.logA)

        D = model.evidence().rvs()
        model_1 = model.posterior(D)
        model_2 = model.joint().condition(i, D)
        assert_allclose(model_1.means, model_2.means)
        assert_allclose(model_1.covs, model_2.covs, rtol=1e-6, atol=1e-6)
        assert_allclose(model_1.logA, model_2.logA)

    def test_bayes_theorem(self, k, d, n):
        model = self.random(k, d, n)
        theta = model.prior().rvs()
        D = model.evidence().rvs()
        assert_allclose(
            model.posterior(D).logpdf(theta) + model.evidence().logpdf(D),
            model.likelihood(theta).logpdf(D) + model.prior().logpdf(theta),
        )
