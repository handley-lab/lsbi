from lsbi.model import LinearModel
import numpy as np
import scipy.stats
from numpy.testing import assert_allclose
import pytest


def _test_shape(model, d, n):
    assert model.n == n
    assert model.d == d
    assert model.M.shape == (d, n)
    assert model.m.shape == (d,)
    assert model.C.shape == (d, d)
    assert model.mu.shape == (n,)
    assert model.Sigma.shape == (n, n)


def test_M():
    model = LinearModel(M=np.random.rand())
    _test_shape(model, 1, 1)

    model = LinearModel(M=np.random.rand(1))
    _test_shape(model, 1, 1)

    model = LinearModel(M=np.random.rand(1, 5))
    _test_shape(model, 1, 5)

    model = LinearModel(M=np.random.rand(3, 1))
    _test_shape(model, 3, 1)

    model = LinearModel(M=np.random.rand(3, 5))
    _test_shape(model, 3, 5)


def test_m_mu():
    model = LinearModel(m=np.random.rand(), mu=np.random.rand())
    _test_shape(model, 1, 1)

    model = LinearModel(m=np.random.rand(1), mu=np.random.rand(1))
    _test_shape(model, 1, 1)

    model = LinearModel(m=np.random.rand(1), mu=np.random.rand(5))
    _test_shape(model, 1, 5)

    model = LinearModel(m=np.random.rand(3), mu=np.random.rand(1))
    _test_shape(model, 3, 1)

    model = LinearModel(m=np.random.rand(3), mu=np.random.rand(5))
    _test_shape(model, 3, 5)


def test_failure():
    with pytest.raises(ValueError) as excinfo:
        LinearModel(m=np.random.rand(5))
    assert "Unable to determine number of parameters n" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        LinearModel(mu=np.random.rand(3))
    assert "Unable to determine data dimensions d" in str(excinfo.value)


def random_model(d, n):
    M = np.random.rand(d, n)
    m = np.random.rand(d)
    C = scipy.stats.wishart(scale=np.eye(d)).rvs()
    mu = np.random.rand(n)
    Sigma = scipy.stats.wishart(scale=np.eye(n)).rvs()
    return LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)


def test_joint():
    d = 5
    n = 3
    N = 100
    model = random_model(d, n)
    prior = model.prior()
    evidence = model.evidence()
    joint = model.joint()

    samples_1 = prior.rvs(N)
    samples_2 = joint.rvs(N)[:, -n:]

    for i in range(n):
        p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(prior.logpdf(samples_2),
                           prior.logpdf(samples_1)).pvalue
    assert p > 1e-5

    samples_1 = evidence.rvs(N)
    samples_2 = joint.rvs(N)[:, :d]

    for i in range(d):
        p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(evidence.logpdf(samples_2),
                           evidence.logpdf(samples_1)).pvalue
    assert p > 1e-5


def test_likelihood_posterior():
    d = 5
    n = 3
    N = 1000
    model = random_model(d, n)
    joint = model.joint()

    samples = []
    theta = model.prior().rvs()
    for _ in range(N):
        data = model.likelihood(theta).rvs()
        theta = model.posterior(data).rvs()
        samples.append(np.concatenate([data, theta])[:])
    samples_1 = np.array(samples)[::100]
    samples_2 = joint.rvs(len(samples_1))

    for i in range(n+d):
        p = scipy.stats.kstest(samples_1[:, i], samples_2[:, i]).pvalue
        assert p > 1e-5

    p = scipy.stats.kstest(joint.logpdf(samples_2),
                           joint.logpdf(samples_1)).pvalue
    assert p > 1e-5


def test_DKL():
    d = 5
    n = 3
    N = 1000
    model = random_model(d, n)

    data = model.evidence().rvs()
    posterior = model.posterior(data)
    prior = model.prior()

    samples = posterior.rvs(N)
    Info = (posterior.logpdf(samples) - prior.logpdf(samples))
    assert_allclose(Info.mean(), model.DKL(data), atol=5*Info.std()/np.sqrt(N))
