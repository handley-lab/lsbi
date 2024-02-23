import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as scipy_multivariate_normal

from lsbi.stats import dkl, mixture_normal, multivariate_normal

shapes = [(2, 3), (3,), ()]
sizes = [(6, 5), (5,), ()]
dims = [1, 2, 4]
pvalue = 1e-7

tests = []
A_tests = []
p_tests = []

for dim in dims:
    for shape in shapes:
        for mean_shape in shapes + ["scalar"]:
            for cov_shape in shapes + ["scalar"]:
                for diagonal in [True, False]:
                    tests.append((dim, shape, mean_shape, cov_shape, diagonal))
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
                                            diagonal,
                                            A_shape,
                                            diagonal_A,
                                            b_shape,
                                            k,
                                        )
                                    )

                    for p in dims:
                        if dim < p:
                            continue
                        p_tests.append((dim, shape, mean_shape, cov_shape, diagonal, p))


def flatten(dist):
    """Convert a multivariate_normal to a list of scipy.stats.multivariate_normal"""
    mean = np.broadcast_to(dist.mean, dist.shape + (dist.dim,)).reshape(-1, dist.dim)
    if dist.diagonal:
        cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim,)).reshape(-1, dist.dim)
    else:
        cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim, dist.dim)).reshape(
            -1, dist.dim, dist.dim
        )

    flat_dist = [
        scipy_multivariate_normal(m, c, allow_singular=True)
        for (m, c) in zip(mean, cov)
    ]
    return flat_dist


class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, dim, shape, mean_shape, cov_shape, diagonal):
        if mean_shape == "scalar":
            mean = np.random.randn()
        else:
            mean = np.random.randn(*mean_shape, dim)

        if cov_shape == "scalar":
            cov = np.random.randn() ** 2 + dim
        elif diagonal:
            cov = np.random.randn(*cov_shape, dim) ** 2 + dim
        else:
            cov = np.random.randn(*cov_shape, dim, dim)
            cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)

        dist = multivariate_normal(mean, cov, shape, dim, diagonal)

        assert dist.dim == dim
        assert np.all(dist.mean == mean)
        assert np.all(dist.cov == cov)
        return dist

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_getitem(self, dim, shape, mean_shape, cov_shape, diagonal):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)

        if len(dist.shape) > 0:
            dist_2 = dist[0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[1:]
            assert dist_2.dim == dim

        if len(dist.shape) > 1:
            dist_2 = dist[0, 0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[2:]
            assert dist_2.dim == dim

            dist_2 = dist[0, :]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[1:]
            assert dist_2.dim == dim

            dist_2 = dist[:, 0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[:-1]
            assert dist_2.dim == dim

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_logpdf(self, dim, shape, mean_shape, cov_shape, diagonal, size):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape

        flat_dist = flatten(dist)
        flat_logpdf = np.array([d.logpdf(x) for d in flat_dist])
        flat_logpdf = np.moveaxis(flat_logpdf, 0, -1).reshape(logpdf.shape)
        assert_allclose(logpdf, flat_logpdf)

        assert_allclose(np.exp(logpdf), dist.pdf(x))

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_rvs_shape(self, dim, shape, mean_shape, cov_shape, diagonal, size):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        rvs = dist.rvs(size)
        assert rvs.shape == size + dist.shape + (dim,)

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_rvs(self, dim, shape, mean_shape, cov_shape, diagonal):
        size = 100
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        rvs = dist.rvs(size)

        mean = np.broadcast_to(dist.mean, dist.shape + (dist.dim,)).reshape(
            -1, dist.dim
        )
        if dist.diagonal:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim,)).reshape(
                -1, dist.dim
            )
        else:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim, dist.dim)).reshape(
                -1, dist.dim, dist.dim
            )

        rvs_ = np.array(
            [
                scipy_multivariate_normal(ms, cs, allow_singular=True).rvs(size)
                for ms, cs in zip(mean, cov)
            ]
        ).reshape(-1, size, dim)

        rvs = np.moveaxis(rvs.reshape(size, -1, dim), 1, 0)

        for a, b in zip(rvs, rvs_):
            for i in range(dim):
                assert scipy.stats.kstest(a[:, i], b[:, i]).pvalue > pvalue

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal, A_shape, diagonal_A, b_shape, k",
        A_tests,
    )
    def test_predict(
        self,
        dim,
        shape,
        mean_shape,
        cov_shape,
        diagonal,
        k,
        A_shape,
        diagonal_A,
        b_shape,
    ):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)

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
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A], np.shape(b)[:-1]
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A, diagonal=diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal, p", p_tests)
    def test_marginalise(self, dim, shape, mean_shape, cov_shape, diagonal, p):
        indices = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        dist_2 = dist.marginalise(indices)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal]
            == np.shape(dist.cov)[: -2 + diagonal]
        )
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal, p", p_tests)
    def test_condition(
        self, dim, shape, mean_shape, cov_shape, diagonal, p, values_shape
    ):
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape, p)
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape)
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal]
            == np.shape(dist.cov)[: -2 + diagonal]
        )
        if cov_shape == "scalar" or diagonal:
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1],
                np.shape(dist.cov)[: -2 + diagonal],
                values_shape,
            )
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_bijector(self, dim, shape, mean_shape, cov_shape, diagonal, x_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape, diagonal)
        x = np.random.rand(*x_shape, dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)

        y = np.random.rand(*x_shape, dim)
        x = dist.bijector(y, inverse=True)

        assert x.shape == np.broadcast_shapes(dist.shape + (dim,), x.shape)

        x = np.random.rand(*x_shape, dim)
        y = dist.bijector(x)
        assert_allclose(np.broadcast_to(x, y.shape), dist.bijector(y, inverse=True))


@pytest.mark.parametrize("logw_shape", shapes)
class TestMixtureNormal(TestMultivariateNormal):
    cls = mixture_normal

    def random(self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal):
        dist = super().random(dim, shape, mean_shape, cov_shape, diagonal)
        logw = np.random.randn(*logw_shape)
        dist = mixture_normal(
            logw, dist.mean, dist.cov, dist.shape, dist.dim, dist.diagonal
        )
        assert np.all(dist.logw == logw)
        if dist.shape:
            assert dist.k == dist.shape[-1]
        else:
            assert dist.k == 1
        return dist

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_getitem(self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal):
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)

        if len(dist.shape) > 0:
            dist_2 = dist[0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[1:]
            assert dist_2.dim == dim

        if len(dist.shape) > 1:
            dist_2 = dist[0, 0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[2:]
            assert dist_2.dim == dim

            dist_2 = dist[0, :]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[1:]
            assert dist_2.dim == dim

            dist_2 = dist[:, 0]
            assert isinstance(dist_2, self.cls)
            assert dist_2.shape == dist.shape[:-1]
            assert dist_2.dim == dim

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_logpdf(
        self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal, size
    ):
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        x = np.random.randn(*size, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == size + dist.shape[:-1]

        assert_allclose(np.exp(logpdf), dist.pdf(x))

        logw = np.broadcast_to(dist.logw, dist.shape).reshape(-1, dist.k).copy()
        logw -= logsumexp(logw, axis=-1, keepdims=True)
        mean = np.broadcast_to(dist.mean, dist.shape + (dist.dim,)).reshape(
            -1, dist.k, dist.dim
        )
        if dist.diagonal:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim,)).reshape(
                -1, dist.k, dist.dim
            )
        else:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim, dist.dim)).reshape(
                -1, dist.k, dist.dim, dist.dim
            )

        flat_dist = [
            [
                scipy_multivariate_normal(m, c, allow_singular=True)
                for (m, c) in zip(ms, cs)
            ]
            for (ms, cs) in zip(mean, cov)
        ]
        flat_logpdf = np.array(
            [
                logsumexp([la + d.logpdf(x) for la, d in zip(las, ds)], axis=0)
                for las, ds in zip(logw, flat_dist)
            ]
        )
        flat_logpdf = np.moveaxis(flat_logpdf, 0, -1).reshape(logpdf.shape)
        assert_allclose(logpdf, flat_logpdf)

    @pytest.mark.parametrize("size", sizes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_rvs_shape(
        self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal, size
    ):
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        rvs = dist.rvs(size)
        assert rvs.shape == size + dist.shape[:-1] + (dim,)

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_rvs(self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal):
        size = 100
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        rvs = dist.rvs(size)
        logw = np.broadcast_to(dist.logw, dist.shape).reshape(-1, dist.k).copy()
        logw -= logsumexp(logw, axis=-1, keepdims=True)
        p = np.exp(logw)
        mean = np.broadcast_to(dist.mean, dist.shape + (dist.dim,)).reshape(
            -1, dist.k, dist.dim
        )
        if dist.diagonal:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim,)).reshape(
                -1, dist.k, dist.dim
            )
        else:
            cov = np.broadcast_to(dist.cov, dist.shape + (dist.dim, dist.dim)).reshape(
                -1, dist.k, dist.dim, dist.dim
            )

        rvs_ = np.array(
            [
                [
                    scipy_multivariate_normal(ms[j], cs[j], allow_singular=True).rvs()
                    for j in np.random.choice(len(ms), p=ps, size=size)
                ]
                for ms, cs, ps in zip(mean, cov, p)
            ]
        ).reshape(-1, size, dim)
        rvs = np.moveaxis(rvs, -2, 0).reshape(-1, size, dim)

        for a, b in zip(rvs, rvs_):
            for i in range(dim):
                assert scipy.stats.kstest(a[:, i], b[:, i]).pvalue > pvalue

    @pytest.mark.parametrize(
        "dim, shape, mean_shape, cov_shape, diagonal, A_shape, diagonal_A, b_shape, k",
        A_tests,
    )
    def test_predict(
        self,
        dim,
        shape,
        logw_shape,
        mean_shape,
        cov_shape,
        diagonal,
        A_shape,
        diagonal_A,
        b_shape,
        k,
    ):
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)

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
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1],
            np.shape(A)[: -2 + diagonal_A],
            np.shape(b)[:-1],
        )
        assert dist_2.dim == k

        dist_2 = dist.predict(A, diagonal=diagonal_A)
        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(
            dist.shape, np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.cov)[: -2 + dist_2.diagonal] == np.broadcast_shapes(
            np.shape(dist.cov)[: -2 + diagonal], np.shape(A)[: -2 + diagonal_A]
        )
        assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
            np.shape(dist.mean)[:-1], np.shape(A)[: -2 + diagonal_A]
        )
        assert dist_2.dim == k

    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal, p", p_tests)
    def test_marginalise(
        self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal, p
    ):
        indices = np.random.choice(dim, p, replace=False)
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        dist_2 = dist.marginalise(indices)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == dist.shape
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal]
            == np.shape(dist.cov)[: -2 + diagonal]
        )
        assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        assert np.shape(dist_2.logw) == np.shape(dist.logw)
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("values_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal, p", p_tests)
    def test_condition(
        self,
        dim,
        shape,
        logw_shape,
        mean_shape,
        cov_shape,
        diagonal,
        p,
        values_shape,
    ):
        indices = np.random.choice(dim, p, replace=False)
        values = np.random.randn(*values_shape[:-1], p)
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        dist_2 = dist.condition(indices, values)

        assert isinstance(dist_2, self.cls)
        assert dist_2.shape == np.broadcast_shapes(dist.shape, values_shape[:-1] + (1,))
        assert (
            np.shape(dist_2.cov)[: -2 + dist_2.diagonal]
            == np.shape(dist.cov)[: -2 + diagonal]
        )
        if cov_shape == "scalar" or diagonal:
            assert np.shape(dist_2.mean)[:-1] == np.shape(dist.mean)[:-1]
        else:
            assert np.shape(dist_2.mean)[:-1] == np.broadcast_shapes(
                np.shape(dist.mean)[:-1],
                np.shape(dist.cov)[: -2 + diagonal],
                values_shape[:-1] + (1,),
            )
        assert np.shape(dist_2.logw) == dist_2.shape
        assert dist_2.dim == dim - p

    @pytest.mark.parametrize("x_shape", shapes)
    @pytest.mark.parametrize("dim, shape, mean_shape, cov_shape, diagonal", tests)
    def test_bijector(
        self, dim, shape, logw_shape, mean_shape, cov_shape, diagonal, x_shape
    ):
        dist = self.random(dim, shape, logw_shape, mean_shape, cov_shape, diagonal)
        x = np.random.rand(*x_shape[:-1], dim)
        y = dist.bijector(x)
        assert y.shape == np.broadcast_shapes(x.shape, dist.shape[:-1] + (dim,))

        y = np.random.rand(*x_shape[:-1], dim)
        x = dist.bijector(y, inverse=True)
        assert x.shape == np.broadcast_shapes(y.shape, dist.shape[:-1] + (dim,))

        x = np.random.rand(*x_shape[:-1], dim)
        y = dist.bijector(x)
        assert_allclose(
            np.broadcast_to(x, y.shape), dist.bijector(y, inverse=True), atol=1e-4
        )


@pytest.mark.parametrize("dim_p, shape_p, mean_shape_p, cov_shape_p, diagonal_p", tests)
@pytest.mark.parametrize("dim_q, shape_q, mean_shape_q, cov_shape_q, diagonal_q", tests)
def test_dkl(
    dim_p,
    shape_p,
    mean_shape_p,
    cov_shape_p,
    diagonal_p,
    dim_q,
    shape_q,
    mean_shape_q,
    cov_shape_q,
    diagonal_q,
):
    p = TestMultivariateNormal().random(
        dim, shape_p, mean_shape_p, cov_shape_p, diagonal_p
    )
    q = TestMultivariateNormal().random(
        dim, shape_q, mean_shape_q, cov_shape_q, diagonal_q
    )

    dkl_pq = dkl(p, q)

    assert_allclose(dkl(p, p), 0, atol=1e-10)
    assert_allclose(dkl(q, q), 0, atol=1e-10)

    assert (dkl_pq >= 0).all()
    assert dkl_pq.shape == np.broadcast_shapes(p.shape, q.shape)

    dkl_mc = dkl(p, q, 1000)
    assert dkl_mc.shape == np.broadcast_shapes(p.shape, q.shape)

    assert_allclose(dkl_pq, dkl_mc, atol=1)
