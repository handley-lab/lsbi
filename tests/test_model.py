from lsbi.model import LinearModel
import numpy as np


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
