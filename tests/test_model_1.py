import numpy as np
import pytest

from lsbi.model_1 import LinearMixtureModel, LinearModel

shapes = [(2, 3), (3,), ()]
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

        model = LinearModel(M, m, C, mu, Sigma, shape, n, d)
        assert model.d == d
        assert model.n == n
        assert np.all(model.M == M)
        assert np.all(model.m == m)
        assert np.all(model.C == C)
        assert np.all(model.mu == mu)
        assert np.all(model.Sigma == Sigma)
        return model

    def test_init(self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        assert model.shape == np.broadcast_shapes(
            shape,
            np.shape(np.atleast_2d(model.M))[:-2],
            np.shape(np.atleast_1d(model.m))[:-1],
            np.shape(np.atleast_1d(model.mu))[:-1],
            np.shape(np.atleast_2d(model.C))[:-2],
            np.shape(np.atleast_2d(model.Sigma))[:-2],
        )

    @pytest.mark.parametrize("theta_shape", shapes)
    def test_likelihood(
        self, theta_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        theta = np.random.randn(*theta_shape, n)
        dist = model.likelihood(theta)
        assert dist.shape == np.broadcast_shapes(model.shape, theta_shape)
        assert dist.dim == model.d

    def test_prior(self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.prior()
        assert dist.shape == model.shape
        assert dist.dim == model.n

    @pytest.mark.parametrize("D_shape", shapes)
    def test_posterior(
        self, D_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        D = np.random.randn(*D_shape, d)
        dist = model.posterior(D)
        assert dist.shape == np.broadcast_shapes(model.shape, D_shape)
        assert dist.dim == model.n

    def test_evidence(
        self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.evidence()
        assert dist.shape == model.shape
        assert dist.dim == model.d

    def test_joint(self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d):
        model = self.random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.joint()
        assert dist.shape == model.shape
        assert dist.dim == model.n + model.d


@pytest.mark.parametrize("logA_shape", shapes)
class TestLinearMixtureModel(TestLinearModel):
    def random(
        self, logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = super().random(
            M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        logA = np.random.randn(*logA_shape)
        model = LinearMixtureModel(
            logA, model.M, model.m, model.C, model.mu, model.Sigma, shape, n, d
        )
        assert np.all(model.logA == logA)
        return model

    def test_init(
        self, logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        assert model.shape == np.broadcast_shapes(
            shape,
            np.shape(np.atleast_2d(model.M))[:-2],
            np.shape(np.atleast_1d(model.m))[:-1],
            np.shape(np.atleast_1d(model.mu))[:-1],
            np.shape(np.atleast_2d(model.C))[:-2],
            np.shape(np.atleast_2d(model.Sigma))[:-2],
            np.shape(model.logA),
        )

    @pytest.mark.parametrize("theta_shape", shapes)
    def test_likelihood(
        self,
        theta_shape,
        M_shape,
        m_shape,
        C_shape,
        mu_shape,
        Sigma_shape,
        logA_shape,
        shape,
        n,
        d,
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        theta = np.random.randn(*theta_shape[:-1], n)
        dist = model.likelihood(theta)
        if model.shape != ():
            assert dist.shape == np.broadcast_shapes(model.shape, theta_shape)
        assert dist.dim == model.d

    def test_prior(
        self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, logA_shape, shape, n, d
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.prior()
        assert dist.shape == model.shape
        assert dist.dim == model.n

    @pytest.mark.parametrize("D_shape", shapes)
    def test_posterior(
        self,
        D_shape,
        M_shape,
        m_shape,
        C_shape,
        mu_shape,
        Sigma_shape,
        logA_shape,
        shape,
        n,
        d,
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        D = np.random.randn(*D_shape[:-1], d)
        dist = model.posterior(D)
        if model.shape != ():
            assert dist.shape == np.broadcast_shapes(model.shape, D_shape)
        assert dist.dim == model.n

    def test_evidence(
        self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, logA_shape, shape, n, d
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.evidence()
        assert dist.shape == model.shape
        assert dist.dim == model.d

    def test_joint(
        self, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, logA_shape, shape, n, d
    ):
        model = self.random(
            logA_shape, M_shape, m_shape, C_shape, mu_shape, Sigma_shape, shape, n, d
        )
        dist = model.joint()
        assert dist.shape == model.shape
        assert dist.dim == model.n + model.d
