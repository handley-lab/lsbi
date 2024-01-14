import numpy as np
import pytest

from lsbi.stats_1 import mixture_normal, multivariate_normal

shapes = [(2, 3), (3,), ()]
sizes = [(6, 5), (5,), ()]
dims = [1, 2, 4]

tests = []
A_tests = []
p_tests = []

for dim in dims:
    for shape in shapes:
        for mean_shape in shapes + ["scalar"]:
            for cov_shape in shapes + ["scalar"]:
                for diagonal_cov in [True, False]:
                    tests.append((dim, shape, mean_shape, cov_shape, diagonal_cov))
                    for A_shape in shapes + ["scalar"]:
                        for diagonal_A in [True, False]:
                            for b_shape in shapes + ["scalar"]:
                                for k in dims:
                                    if (diagonal_A or A_shape == "scalar") and (
                                        b_shape != "scalar" or k != dim
                                    ):
                                        continue
                                    A_tests.append(
                                        (
                                            dim,
                                            shape,
                                            mean_shape,
                                            cov_shape,
                                            diagonal_cov,
                                            A_shape,
                                            diagonal_A,
                                            b_shape,
                                            k,
                                        )
                                    )

                    for p in dims:
                        if dim < p:
                            continue
                        p_tests.append(
                            (dim, shape, mean_shape, cov_shape, diagonal_cov, p)
                        )


class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, dim, shape, mean_shape, cov_shape, diagonal_cov):
        if mean_shape == "scalar":
            mean = np.random.randn()
        else:
            mean = np.random.randn(*mean_shape, dim)

        if cov_shape == "scalar":
            cov = np.random.randn() ** 2
        elif diagonal_cov:
            cov = np.random.randn(*cov_shape, dim) ** 2
        else:
            cov = np.random.randn(*cov_shape, dim, dim)
            cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)

        dist = multivariate_normal(mean, cov, shape, dim, diagonal_cov)

        assert dist.dim == dim
        assert np.all(dist.mean == mean)
        assert np.all(dist.cov == cov)
        return dist

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_logpdf(self, dim, shape, mean_shape, cov_shape, diagonal_cov, size):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_rvs(self, dim, shape, mean_shape, cov_shape, diagonal_cov, size):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        x = dist.rvs(size)
        assert x.shape == size + dist.shape + (dim,)

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, A_shape, diagonal_A, b_shape, k",
        A_tests,
    )
    def test_predict(
        self,
        dim,
        shape,
        mean_shape,
        cov_shape,
        diagonal_cov,
        k,
        A_shape,
        diagonal_A,
        b_shape,
    ):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)

        if b_shape == "scalar":
            b = np.random.randn()
        else:
            b = np.random.randn(*b_shape, k)

        if A_shape == "scalar":
            A = np.random.randn()
        elif diagonal_A:
            A = np.random.randn(*A_shape, dim)
        else:
            A = np.random.randn(*A_shape, k, dim)

        dist_2 = dist.predict(A, b, diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, np.shape(A)[: -2 + diagonal_A], np.shape(b)[:-1]
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal_cov], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A], np.shape(b)[:-1]
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A, diagonal_A=diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal_cov], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, p", p_tests
    )
    def test_marginalise(self, dim, shape, mean_shape, cov_shape, diagonal_cov, p):
        indices = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        dist_2 = dist.marginalise(indices)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov]
            == np.shape(dist.cov)[: -2 + diagonal_cov]
        )
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, p", p_tests
    )
    def test_condition(
        self, dim, shape, mean_shape, cov_shape, diagonal_cov, p, values_shape
    ):
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape, p)
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape)
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov]
            == np.shape(dist.cov)[: -2 + diagonal_cov]
        )
        if cov_shape == "scalar" or diagonal_cov:
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1],
                np.shape(dist.cov)[: -2 + diagonal_cov],
                values_shape,
            )
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_bijector(self, dim, shape, mean_shape, cov_shape, diagonal_cov, x_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        x = np.random.rand(*x_shape, dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)

        y = np.random.rand(*x_shape, dim)
        x = dist.bijector(y, inverse=True)

        assert x.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)


@pytest.mark.parametrize("logA_shape", shapes)
class TestMixtureNormal(TestMultivariateNormal):
    cls = mixture_normal

    def random(self, dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov):
        dist = super().random(dim, shape, mean_shape, cov_shape, diagonal_cov)
        logA = np.random.randn(*logA_shape)
        dist = mixture_normal(
            logA, dist.mean, dist.cov, dist.shape, dist.dim, dist.diagonal_cov
        )
        assert np.all(dist.logA == logA)
        return dist

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_logpdf(
        self, dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov, size
    ):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape[:-1]

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_rvs(
        self, dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov, size
    ):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)
        x = dist.rvs(size)
        assert x.shape == size + dist.shape[:-1] + (dim,)

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, A_shape, diagonal_A, b_shape, k",
        A_tests,
    )
    def test_predict(
        self,
        dim,
        shape,
        logA_shape,
        mean_shape,
        cov_shape,
        diagonal_cov,
        A_shape,
        diagonal_A,
        b_shape,
        k,
    ):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)

        if b_shape == "scalar":
            b = np.random.randn()
        else:
            b = np.random.randn(*b_shape, k)

        if A_shape == "scalar":
            A = np.random.randn()
        elif diagonal_A:
            A = np.random.randn(*A_shape, dim)
        else:
            A = np.random.randn(*A_shape, k, dim)

        dist_2 = dist.predict(A, b, diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape,
            np.shape(A)[: -2 + diagonal_A],
            np.shape(b)[:-1],
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal_cov], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1],
            np.shape(A)[: -2 + diagonal_A],
            np.shape(b)[:-1],
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A, diagonal_A=diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal_cov], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, p", p_tests
    )
    def test_marginalise(
        self, dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov, p
    ):
        indices = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)
        dist_2 = dist.marginalise(indices)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov]
            == np.shape(dist.cov)[: -2 + diagonal_cov]
        )
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert np.shape(dist_2.logA) == np.shape(dist.logA)
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal_cov, p", p_tests
    )
    def test_condition(
        self,
        dim,
        shape,
        logA_shape,
        mean_shape,
        cov_shape,
        diagonal_cov,
        p,
        values_shape,
    ):
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape[:-1], p)
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape[:-1] + (1,))
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal_cov]
            == np.shape(dist.cov)[: -2 + diagonal_cov]
        )
        if cov_shape == "scalar" or diagonal_cov:
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1],
                np.shape(dist.cov)[: -2 + diagonal_cov],
                values_shape[:-1] + (1,),
            )
        assert np.shape(dist_2.logA) == dist_2.shape
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal_cov", tests)
    def test_bijector(
        self, dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov, x_shape
    ):
        dist = self.random(dim, shape, logA_shape, mean_shape, cov_shape, diagonal_cov)
        x = np.random.rand(*x_shape[:-1], dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(x.shape, dist.shape[:-1] + (dim,))

        y = np.random.rand(*x_shape[:-1], dim)
        x = dist.bijector(y, inverse=True)
        assert x.shape == np.broadcast_shapes(y.shape, dist.shape[:-1] + (dim,))
