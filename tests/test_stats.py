import pytest
from lsbi.stats import mixture_multivariate_normal
import numpy as np
import scipy.stats
import scipy.special


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("d", [1, 2, 5])
def test_mixture_multivariate_normal(k, d):
    k = 5
    d = 2
    N = 1000
    means = np.random.randn(k, d)
    covs = scipy.stats.wishart(scale=np.eye(d)).rvs(k)
    if k == 1:
        covs = np.array([covs])
    logA = np.log(scipy.stats.dirichlet(np.ones(k)).rvs())[0] + 10
    mixture = mixture_multivariate_normal(means, covs, logA)
    logA -= scipy.special.logsumexp(logA)

    samples_1, logpdfs_1 = [], []
    for _ in range(N):
        i = np.random.choice(k, p=np.exp(logA))
        mvn = scipy.stats.multivariate_normal(means[i], covs[i])
        x = mvn.rvs()
        samples_1.append(x)
        logpdfs_1.append(mixture.logpdf(x))
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
        assert mvn.logpdf(x).shape == mixture.logpdf(x).shape
