import pytest
from lsbi.stats import mixture_multivariate_normal
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
