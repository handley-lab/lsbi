import numpy as np
import pytest

from lsbi.model_1 import LinearMixtureModel, LinearModel

shapes = [(2, 3), (3,), ()]
sizes = [(6, 5), (5,), ()]
dims = [1, 2, 4]


@pytest.mark.parametrize("d", dims)
@pytest.mark.parametrize("n", dims)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("m_shape", shapes + ["scalar"])
@pytest.mark.parametrize("mu_shape", shapes + ["scalar"])
@pytest.mark.parametrize("M_shape", shapes + ["scalar", "vector"])
@pytest.mark.parametrize("C_shape", shapes + ["scalar", "vector"])
@pytest.mark.parametrize("Sigma_shape", shapes + ["scalar", "vector"])
class TestLinearModel(object):
    cls = LinearModel

    def random(self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d):
        if M_shape == "scalar":
            M = np.random.randn()
        elif M_shape == "vector":
            M = np.random.randn(n)
        else:
            M = np.random.randn(*M_shape, d, n)

        if m_shape == "scalar":
            m = np.random.randn()
        else:
            m = np.random.randn(*m_shape, d)

        if C_shape == "scalar":
            C = np.random.randn() ** 2
        elif C_shape == "vector":
            C = np.random.randn(d) ** 2
        else:
            C = np.random.randn(*C_shape, d, d)
            C = np.einsum("...ij,...kj->...ik", C, C) + d * np.eye(d)

        if mu_shape == "scalar":
            mu = np.random.randn()
        else:
            mu = np.random.randn(*mu_shape, n)

        if Sigma_shape == "scalar":
            Sigma = np.random.randn() ** 2
        elif Sigma_shape == "vector":
            Sigma = np.random.randn(n) ** 2
        else:
            Sigma = np.random.randn(*Sigma_shape, n, n)
            Sigma = np.einsum("...ij,...kj->...ik", Sigma, Sigma) + n * np.eye(n)

        model = self.cls(M, m, C, mu, Sigma, shape, n, d)

        assert model.d == d
        assert model.n == n
        model.prior()
        assert model.shape == np.broadcast_shapes(
            shape,
            np.shape(np.atleast_2d(M))[:-2],
            np.shape(np.atleast_1d(m))[:-1],
            np.shape(np.atleast_1d(mu))[:-1],
            np.shape(np.atleast_2d(C))[:-2],
            np.shape(np.atleast_2d(Sigma))[:-2],
        )
        assert np.all(model.M == M)
        assert np.all(model.m == m)
        assert np.all(model.C == C)
        assert np.all(model.mu == mu)
        assert np.all(model.Sigma == Sigma)
        return model

    @pytest.mark.parametrize("theta_shape", shapes)
    def test_likelihood(
        self, theta_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        theta = np.random.randn(*theta_shape, n)
        dist = model.likelihood(theta)
        assert dist.shape == np.broadcast_shapes(shape, theta_shape)
