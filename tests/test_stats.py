import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from scipy.stats import invwishart, kstest

from lsbi.stats import mixture_multivariate_normal, multivariate_normal

N = 1000


@pytest.mark.parametrize("d", [1, 2, 5, 10])
@pytest.mark.parametrize("k", [1, 2, 5, 10])
class TestMixtureMultivariateNormal(object):
    cls = mixture_multivariate_normal

    def random(self, k, d):
        means = np.random.randn(k, d)
        covs = invwishart(scale=np.eye(d), df=d * 10).rvs(k)
        if k == 1:
            covs = np.array([covs])
        logA = np.log(scipy.stats.dirichlet(np.ones(k)).rvs())[0] + 10
        return self.cls(means, covs, logA)

    def test_rvs(self, k, d):
        dist = self.random(k, d)
        logA = dist.logA
        logA -= scipy.special.logsumexp(logA)
        mvns = [
            scipy.stats.multivariate_normal(dist.means[i], dist.covs[i])
            for i in range(k)
        ]

        samples_1, logpdfs_1 = [], []
        for _ in range(N):
            i = np.random.choice(k, p=np.exp(logA))
            x = mvns[i].rvs()
            samples_1.append(x)
            logpdf = scipy.special.logsumexp(
                [mvns[j].logpdf(x) + logA[j] for j in range(k)]
            )
            assert_allclose(logpdf, dist.logpdf(x))
            logpdfs_1.append(logpdf)
        samples_1, logpdfs_1 = np.array(samples_1), np.array(logpdfs_1)

        samples_2 = dist.rvs(N)
        logpdfs_2 = dist.logpdf(samples_2)

        for i in range(d):
            if d == 1:
                p = kstest(samples_1, samples_2).pvalue
            else:
                p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(logpdfs_1, logpdfs_2).pvalue
        assert p > 1e-5

        for shape in [(d,), (3, d), (3, 4, d)]:
            x = np.random.rand(*shape)
            assert mvns[0].logpdf(x).shape == dist.logpdf(x).shape

    def test_bijector(self, k, d):
        dist = self.random(k, d)

        # Test inversion
        x = np.random.rand(N, d)
        theta = dist.bijector(x)
        assert_allclose(dist.bijector(theta, inverse=True), x, atol=1e-6)

        # Test sampling
        samples = dist.rvs(N)
        for i in range(d):
            if d == 1:
                p = kstest(np.squeeze(theta), samples).pvalue
            else:
                p = kstest(theta[:, i], samples[:, i]).pvalue
            assert p > 1e-5

        p = kstest(dist.logpdf(samples), dist.logpdf(theta)).pvalue
        assert p > 1e-5

        # Test shapes
        x = np.random.rand(d)
        theta = dist.bijector(x)
        assert theta.shape == x.shape
        assert dist.bijector(theta, inverse=True).shape == x.shape

        x = np.random.rand(3, 4, d)
        theta = dist.bijector(x)
        assert theta.shape == x.shape
        assert dist.bijector(theta, inverse=True).shape == x.shape

    @pytest.mark.parametrize("p", np.arange(1, 5))
    def test_marginalise_condition(self, d, k, p):
        if d <= p:
            pytest.skip("d <= p")
        i = np.random.choice(d, p, replace=False)
        j = np.array([x for x in range(d) if x not in i])
        dist = self.random(k, d)
        mixture_2 = dist.marginalise(i)
        assert mixture_2.means.shape == (k, d - p)
        assert mixture_2.covs.shape == (k, d - p, d - p)
        assert_allclose(dist.means[:, j], mixture_2.means)
        assert_allclose(dist.covs[:, j][:, :, j], mixture_2.covs)

        v = np.random.randn(k, p)
        mixture_3 = dist.condition(i, v)
        assert mixture_3.means.shape == (k, d - p)
        assert mixture_3.covs.shape == (k, d - p, d - p)

        v = np.random.randn(p)
        mixture_3 = dist.condition(i, v)
        assert mixture_3.means.shape == (k, d - p)
        assert mixture_3.covs.shape == (k, d - p, d - p)


@pytest.mark.parametrize("d", [1, 2, 5, 10])
class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, d):
        mean = np.random.randn(d)
        cov = invwishart(scale=np.eye(d), df=d * 10).rvs()
        return self.cls(mean, cov)

    def test_rvs(self, d):
        dist = self.random(d)
        mvn = scipy.stats.multivariate_normal(dist.mean, dist.cov)

        samples_1 = mvn.rvs(N)
        logpdfs_1 = mvn.logpdf(samples_1)
        assert_allclose(logpdfs_1, dist.logpdf(samples_1))
        samples_2 = dist.rvs(N)
        logpdfs_2 = dist.logpdf(samples_2)

        for i in range(d):
            if d == 1:
                p = kstest(samples_1, samples_2).pvalue
            else:
                p = kstest(samples_1[:, i], samples_2[:, i]).pvalue
            assert p > 1e-5

        p = kstest(logpdfs_1, logpdfs_2).pvalue
        assert p > 1e-5

        for shape in [(), (d,), (3, d), (3, 4, d)]:
            x = np.random.rand(*shape)
            assert mvn.logpdf(x).shape == dist.logpdf(x).shape

    def test_bijector(self, d):
        dist = self.random(d)
        # Test inversion
        x = np.random.rand(N, d)
        theta = dist.bijector(x)
        assert_allclose(dist.bijector(theta, inverse=True), x, atol=1e-6)

        # Test sampling
        samples = dist.rvs(N)
        for i in range(d):
            if d == 1:
                p = kstest(np.squeeze(theta), samples).pvalue
            else:
                p = kstest(theta[:, i], samples[:, i]).pvalue
            assert p > 1e-5

        p = kstest(dist.logpdf(samples), dist.logpdf(theta)).pvalue
        assert p > 1e-5

        # Test shapes
        x = np.random.rand(d)
        theta = dist.bijector(x)
        assert theta.shape == x.shape
        assert dist.bijector(theta, inverse=True).shape == x.shape

        x = np.random.rand(3, 4, d)
        theta = dist.bijector(x)
        assert theta.shape == x.shape
        assert dist.bijector(theta, inverse=True).shape == x.shape

    @pytest.mark.parametrize("p", np.arange(1, 5))
    def test_marginalise_condition_multivariate_normal(self, d, p):
        if d <= p:
            pytest.skip("d <= p")
        i = np.random.choice(d, p, replace=False)
        j = np.array([x for x in range(d) if x not in i])
        dist_1 = self.random(d)
        dist_2 = dist_1.marginalise(i)
        assert dist_2.mean.shape == (d - p,)
        assert dist_2.cov.shape == (d - p, d - p)
        assert_allclose(dist_1.mean[j], dist_2.mean)
        assert_allclose(dist_1.cov[j][:, j], dist_2.cov)

        v = np.random.randn(p)
        dist_3 = dist_1.condition(i, v)
        assert dist_3.mean.shape == (d - p,)
        assert dist_3.cov.shape == (d - p, d - p)
