import numpy as np
import pytest

from lsbi.stats_1 import mixture_normal, multivariate_normal

shapes = [(2, 3, 4), (3, 4), (4,), ()]
sizes = [(8, 7, 6), (7, 6), (6,), ()]
dims = [0, 1, 2, 5]


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("mean_shape", shapes + ["scalar"])
@pytest.mark.parametrize("cov_shape", shapes + ["scalar", "vector"])
class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, dim, shape, mean_shape, cov_shape):
        if mean_shape == "scalar":
            mean = np.random.randn()
            mean_shape = shape
        else:
            mean = np.random.randn(*mean_shape, dim)
        if cov_shape == "scalar":
            cov = np.random.randn() ** 2
            cov_shape = shape
        elif cov_shape == "vector":
            cov = np.random.randn(dim) ** 2
            cov_shape = shape
        else:
            cov = np.random.randn(*cov_shape, dim, dim)
            cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)
        dist = self.cls(mean, cov, shape, dim)

        assert dist.dim == dim
        assert dist.shape == np.broadcast_shapes(shape, mean_shape, cov_shape)
        assert np.all(dist.mean == mean)
        assert np.all(dist.cov == cov)
        return dist

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
        assert np.shape(dist_2.cov)[:-2] == np.broadcast_shapes(
            np.shape(dist.cov)[:-2], A.shape[:-2]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], A.shape[:-2], b.shape[:-1]
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, A.shape[:-2])
        assert np.shape(dist_2.cov)[:-2] == np.broadcast_shapes(
            np.shape(dist.cov)[:-2], A.shape[:-2]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], A.shape[:-2]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize("p", dims)
    def test_marginalise(self, dim, shape, mean_shape, cov_shape, p):
        if dim < p:
            pytest.skip("dim < p")
        i = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, mean_shape, cov_shape)
        dist_2 = dist.marginalise(i)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert np.shape(dist_2.cov)[:-2] == np.shape(dist.cov)[:-2]
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("p", dims)
    def test_condition(self, dim, shape, mean_shape, cov_shape, p, values_shape):
        if dim < p:
            pytest.skip("dim < p")

        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape, p)
        dist = self.random(dim, shape, mean_shape, cov_shape)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape)
        assert np.shape(dist_2.cov)[:-2] == np.shape(dist.cov)[:-2]
        if cov_shape == "scalar" or cov_shape == "vector":
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1], np.shape(dist.cov)[:-2], values_shape
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
@pytest.mark.parametrize("mean_shape", shapes + ["scalar"])
@pytest.mark.parametrize("cov_shape", shapes + ["scalar", "vector"])
@pytest.mark.parametrize("dim", dims)
class TestMixtureNormal(object):
    cls = mixture_normal

    def random(self, dim, shape, logA_shape, mean_shape, cov_shape):
        logA = np.random.randn(*logA_shape)
        if mean_shape == "scalar":
            mean = np.random.randn()
            mean_shape = shape
        else:
            mean = np.random.randn(*mean_shape, dim)

        if cov_shape == "scalar":
            cov = np.random.randn() ** 2
            cov_shape = shape
        elif cov_shape == "vector":
            cov = np.random.randn(dim) ** 2
            cov_shape = shape
        else:
            cov = np.random.randn(*cov_shape, dim, dim)
            cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)

        dist = self.cls(logA, mean, cov, shape, dim)

        assert dist.dim == dim
        assert dist.shape == np.broadcast_shapes(
            shape, logA_shape, mean_shape, cov_shape
        )
        assert np.all(dist.logA == logA)
        assert np.all(dist.mean == mean)
        assert np.all(dist.cov == cov)
        return dist

    @pytest.mark.parametrize("size", sizes)
    def test_logpdf(self, dim, shape, logA_shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape[:-1]

    @pytest.mark.parametrize("size", sizes)
    def test_rvs(self, dim, shape, logA_shape, mean_shape, cov_shape, size):
        dim = 5
        shape = (2, 3, 4)
        logA_shape = (2, 3, 4)
        mean_shape = (2, 3, 4)
        cov_shape = "vector"
        size = (8, 7, 6)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        x = dist.rvs(size)
        assert x.shape == size + dist.shape[:-1] + (dim,)

    @pytest.mark.parametrize("A_shape", shapes)
    @pytest.mark.parametrize("b_shape", shapes)
    @pytest.mark.parametrize("k", dims)
    def test_predict(
        self, dim, shape, logA_shape, mean_shape, cov_shape, k, A_shape, b_shape
    ):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        A = np.random.randn(*A_shape[:-1], k, dim)
        b = np.random.randn(*b_shape[:-1], k)

        dist_2 = dist.predict(A, b)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, A.shape[:-2] + (1,), b.shape[:-1] + (1,)
        )
        assert np.shape(dist_2.cov)[:-2] == np.broadcast_shapes(
            np.shape(dist.cov)[:-2], A.shape[:-2] + (1,)
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], A.shape[:-2] + (1,), b.shape[:-1] + (1,)
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, A.shape[:-2] + (1,))
        assert np.shape(dist_2.cov)[:-2] == np.broadcast_shapes(
            np.shape(dist.cov)[:-2], A.shape[:-2] + (1,)
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], A.shape[:-2] + (1,)
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize("p", dims)
    def test_marginalise(self, dim, shape, logA_shape, mean_shape, cov_shape, p):
        if dim < p:
            pytest.skip("dim < p")
        i = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        dist_2 = dist.marginalise(i)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert np.shape(dist_2.cov)[:-2] == np.shape(dist.cov)[:-2]
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert np.shape(dist_2.logA) == np.shape(dist.logA)
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("p", dims)
    def test_condition(
        self, dim, shape, logA_shape, mean_shape, cov_shape, p, values_shape
    ):
        if dim < p:
            pytest.skip("dim < p")
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape[:-1], p)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape[:-1] + (1,))
        assert np.shape(dist_2.cov)[:-2] == np.shape(dist.cov)[:-2]
        if cov_shape == "scalar" or cov_shape == "vector":
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1],
                np.shape(dist.cov)[:-2],
                values_shape[:-1] + (1,),
            )
        assert np.shape(dist_2.logA) == dist_2.shape
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
