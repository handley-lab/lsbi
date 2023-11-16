import pytest
from lsbi.stats import (mixture_multivariate_normal,
                        multivariate_normal)
from numpy.testing import assert_allclose
import numpy as np
import scipy.stats
import scipy.special


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("d", [1, 2, 5])
def test_mixture_multivariate_normal(k, d):
    N = 1000
    means = np.random.randn(k, d)
    covs = scipy.stats.wishart(scale=np.eye(d)).rvs(k)
    if k == 1:
        covs = np.array([covs])
    logA = np.log(scipy.stats.dirichlet(np.ones(k)).rvs())[0] + 10
    mixture = mixture_multivariate_normal(means, covs, logA)
    logA -= scipy.special.logsumexp(logA)

    samples_1, logpdfs_1 = [], []
    mvns = [scipy.stats.multivariate_normal(means[i], covs[i])
            for i in range(k)]
    for _ in range(N):
        i = np.random.choice(k, p=np.exp(logA))
        x = mvns[i].rvs()
        samples_1.append(x)
        logpdf = scipy.special.logsumexp([mvns[j].logpdf(x) + logA[j]
                                          for j in range(k)])
        assert_allclose(logpdf, mixture.logpdf(x))
        logpdfs_1.append(logpdf)
    samples_1, logpdfs_1 = np.array(samples_1), np.array(logpdfs_1)

    samples_2 = mixture.rvs(N)
    logpdfs_2 = mixture.logpdf(samples_2)

    for i in range(d):
        if d == 1:
            p = scipy.stats.kstest(samples_1, samples_2).pvalue
        else:
            p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(logpdfs_1, logpdfs_2).pvalue
    assert p > 1e-5

    for shape in [(d,), (3, d), (3, 4, d)]:
        x = np.random.rand(*shape)
        assert mvns[0].logpdf(x).shape == mixture.logpdf(x).shape


def test_mixture_multivariate_normal_bijector():
    k = 4
    d = 10
    covs = scipy.stats.wishart.rvs(d, np.eye(d), size=k)
    means = np.random.randn(k, d)
    logA = np.log(scipy.stats.dirichlet.rvs(np.ones(k))[0])
    model = mixture_multivariate_normal(means, covs, logA)

    # Test inversion
    x = np.random.rand(1000, d)
    theta = model.bijector(x)
    assert_allclose(model.bijector(theta, inverse=True), x, atol=1e-6)

    # Test sampling
    samples = model.rvs(1000)
    for i in range(d):
        p = scipy.stats.kstest(theta[:, i], samples[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(model.logpdf(samples), model.logpdf(theta)).pvalue
    assert p > 1e-5

    # Test shapes
    x = np.random.rand(d)
    theta = model.bijector(x)
    assert theta.shape == x.shape
    assert model.bijector(theta, inverse=True).shape == x.shape

    x = np.random.rand(3, 4, d)
    theta = model.bijector(x)
    assert theta.shape == x.shape
    assert model.bijector(theta, inverse=True).shape == x.shape


def test_multivariate_normal_bijector():
    d = 10
    cov = scipy.stats.wishart.rvs(d, np.eye(d))
    mean = np.random.randn(d)
    model = multivariate_normal(mean, cov)

    # Test inversion
    x = np.random.rand(1000, d)
    theta = model.bijector(x)
    assert_allclose(model.bijector(theta, inverse=True), x, atol=1e-6)

    # Test sampling
    samples = model.rvs(1000)
    for i in range(d):
        p = scipy.stats.kstest(theta[:, i], samples[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(model.logpdf(samples), model.logpdf(theta)).pvalue
    assert p > 1e-5

    # Test shapes
    x = np.random.rand(d)
    theta = model.bijector(x)
    assert theta.shape == x.shape
    assert model.bijector(theta, inverse=True).shape == x.shape

    x = np.random.rand(3, 4, d)
    theta = model.bijector(x)
    assert theta.shape == x.shape
    assert model.bijector(theta, inverse=True).shape == x.shape


def test_marginalise_condition_multivariate_normal():
    d = 5
    mean = np.random.randn(d)
    cov = scipy.stats.wishart(scale=np.eye(d)).rvs()
    dist_1 = multivariate_normal(mean, cov)
    dist_2 = dist_1.marginalise([0, 2, 4])
    assert dist_2.mean.shape == (2,)
    assert dist_2.cov.shape == (2, 2)
    assert_allclose(dist_1.mean[[1, 3]], dist_2.mean)
    assert_allclose(dist_1.cov[[1, 3]][:, [1, 3]], dist_2.cov)

    dist_3 = dist_1.condition([0, 2, 4], [1, 2, 3])
    assert dist_3.mean.shape == (2,)
    assert dist_3.cov.shape == (2, 2)


@pytest.mark.parametrize("k", np.arange(1, 5))
@pytest.mark.parametrize("d", np.arange(1, 5))
@pytest.mark.parametrize("p", np.arange(1, 5))
def test_marginalise_condition_mixtures(d, k, p):
    if d <= p:
        pytest.skip("d <= p")
    i = np.random.choice(d, p, replace=False)
    j = np.array([x for x in range(d) if x not in i])
    means = np.random.randn(k, d)
    covs = scipy.stats.wishart(scale=np.eye(d)).rvs(k)
    if k == 1:
        covs = np.array([covs])
    covs.shape
    means.shape
    logA = np.log(scipy.stats.dirichlet(np.ones(k)).rvs())[0] + 10
    mixture = mixture_multivariate_normal(means, covs, logA)
    mixture_2 = mixture.marginalise(i)
    assert mixture_2.means.shape == (k, d-p)
    assert mixture_2.covs.shape == (k, d-p, d-p)
    assert_allclose(mixture.means[:, j], mixture_2.means)
    assert_allclose(mixture.covs[:, j][:, :, j], mixture_2.covs)

    v = np.random.randn(k, p)
    mixture_3 = mixture.condition(i, v)
    assert mixture_3.means.shape == (k, d-p)
    assert mixture_3.covs.shape == (k, d-p, d-p)

    v = np.random.randn(p)
    mixture_3 = mixture.condition(i, v)
    assert mixture_3.means.shape == (k, d-p)
    assert mixture_3.covs.shape == (k, d-p, d-p)
