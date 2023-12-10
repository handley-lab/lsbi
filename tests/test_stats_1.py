import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from scipy.stats import kstest

from lsbi.stats_1 import mixture_normal, multivariate_normal

shapes = [(2, 3, 4), (3, 4), (4,), ()]
sizes = [(8, 7, 6), (7, 6), (6,), ()]
dims = [1, 2, 5]


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("mean_shape", shapes)
@pytest.mark.parametrize("cov_shape", shapes)
class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, dim, shape, mean_shape, cov_shape):
        mean = np.random.randn(*mean_shape, dim)
        cov = np.random.randn(*cov_shape, dim, dim)
        cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)
        return self.cls(mean, cov, shape)

    def test_properties(self, dim, shape, mean_shape, cov_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        assert dist.dim == dim
        assert dist.shape == np.broadcast_shapes(shape, mean_shape, cov_shape)

    @pytest.mark.parametrize("size", sizes)
    def test_logpdf(self, dim, shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape

    @pytest.mark.parametrize("size", sizes)
    def test_rvs(self, dim, shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        x = dist.rvs(size)
        assert x.shape == size + dist.shape + (dim,)

    @pytest.mark.parametrize("A_shape", shapes)
    @pytest.mark.parametrize("b_shape", shapes)
    @pytest.mark.parametrize("k", dims)
    def test_predict(self, dim, shape, mean_shape, cov_shape, k, A_shape, b_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        A = np.random.randn(*A_shape, k, dim)
        b = np.random.randn(*b_shape, k)

        dist_2 = dist.predict(A, b)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, A.shape[:-2], b.shape[:-1]
        )
        assert dist_2.cov.shape[:-2] == np.broadcast_shapes(
            dist.cov.shape[:-2], A.shape[:-2]
        )
        assert dist_2.mean.shape[:-1] == np.broadcast_shapes(
            dist.mean.shape[:-1], A.shape[:-2], b.shape[:-1]
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, A.shape[:-2])
        assert dist_2.cov.shape[:-2] == np.broadcast_shapes(
            dist.cov.shape[:-2], A.shape[:-2]
        )
        assert dist_2.mean.shape[:-1] == np.broadcast_shapes(
            dist.mean.shape[:-1], A.shape[:-2]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize("p", dims)
    def test_marginalise(self, dim, shape, mean_shape, cov_shape, p):
        if dim <= p:
            pytest.skip("dim <= p")
        i = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, mean_shape, cov_shape)
        dist_2 = dist.marginalise(i)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert dist_2.cov.shape[:-2] == dist.cov.shape[:-2]
        assert dist_2.mean.shape[:-1] == dist.mean.shape[:-1]
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("p", dims)
    def test_condition(self, dim, shape, mean_shape, cov_shape, p, values_shape):
        if dim <= p:
            pytest.skip("dim <= p")
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape, p)
        dist = self.random(dim, shape, mean_shape, cov_shape)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape)
        assert dist_2.cov.shape[:-2] == dist.cov.shape[:-2]
        assert dist_2.mean.shape[:-1] == np.broadcast_shapes(
            dist.mean.shape[:-1], dist.cov.shape[:-2], values_shape
        )
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    def test_bijector(self, dim, shape, mean_shape, cov_shape, x_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        x = np.random.rand(*x_shape, dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)

        y = np.random.rand(*x_shape, dim)
        x = dist.bijector(y, inverse=True)

        assert x.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("logA_shape", shapes)
@pytest.mark.parametrize("mean_shape", shapes)
@pytest.mark.parametrize("cov_shape", shapes)
@pytest.mark.parametrize("dim", dims)
class TestMixtureNormal(object):
    cls = mixture_normal

    def random(self, dim, shape, logA_shape, mean_shape, cov_shape):
        logA = np.random.randn(*logA_shape)
        mean = np.random.randn(*mean_shape, dim)
        cov = np.random.randn(*cov_shape, dim, dim)
        cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)
        return self.cls(logA, mean, cov, shape)

    def test_properties(self, dim, shape, logA_shape, mean_shape, cov_shape):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        assert dist.dim == dim
        assert dist.shape == np.broadcast_shapes(
            shape, mean_shape, cov_shape, logA_shape
        )

    @pytest.mark.parametrize("size", sizes)
    def test_logpdf(self, dim, shape, logA_shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape[:-1]

    @pytest.mark.parametrize("size", sizes)
    def test_rvs(self, dim, shape, logA_shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        x = dist.rvs(size)
        assert x.shape == size + dist.shape[:-1] + (dim,)

    @pytest.mark.parametrize("p", dims)
    def test_marginalise(self, dim, shape, logA_shape, mean_shape, cov_shape, p):
        if dim <= p:
            pytest.skip("dim <= p")
        i = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        dist_2 = dist.marginalise(i)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert dist_2.cov.shape[:-2] == dist.cov.shape[:-2]
        assert dist_2.mean.shape[:-1] == dist.mean.shape[:-1]
        assert dist_2.logA.shape == dist.logA.shape
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("p", dims)
    def test_condition(
        self, dim, shape, logA_shape, mean_shape, cov_shape, p, values_shape
    ):
        if dim <= p:
            pytest.skip("dim <= p")
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape[:-1], p)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape[:-1] + (1,))
        assert dist_2.cov.shape[:-2] == dist.cov.shape[:-2]
        assert dist_2.mean.shape[:-1] == np.broadcast_shapes(
            dist.mean.shape[:-1], dist.cov.shape[:-2], values_shape[:-1] + (1,)
        )
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    def test_bijector(self, dim, shape, logA_shape, mean_shape, cov_shape, x_shape):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        x = np.random.rand(*x_shape[:-1], dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(x.shape, dist.shape[:-1] + (dim,))

        y = np.random.rand(*x_shape[:-1], dim)
        x = dist.bijector(y, inverse=True)
        assert x.shape == np.broadcast_shapes(y.shape, dist.shape[:-1] + (dim,))
