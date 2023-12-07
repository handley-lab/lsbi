import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from scipy.stats import kstest

from lsbi.stats_1 import multivariate_normal

shapes = [(2, 3, 4), (3, 4), (4,), ()]


@pytest.mark.parametrize("mean_shape", shapes)
@pytest.mark.parametrize("cov_shape", shapes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dim", [1, 5])
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

    @pytest.mark.parametrize("x_shape", [(8, 7, 6), (8, 7), (8,), ()])
    def test_logpdf(self, dim, shape, mean_shape, cov_shape, x_shape):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        x = np.random.randn(*x_shape, dim)
        logpdf = dist.logpdf(x)
        assert logpdf.shape == x_shape + dist.shape

    @pytest.mark.parametrize("size", [(8, 7, 6), (8, 7), (8,)])
    def test_rvs(self, dim, shape, mean_shape, cov_shape, size):
        dist = self.random(dim, shape, mean_shape, cov_shape)
        x = dist.rvs(size)
        assert x.shape == tuple(np.atleast_1d(size)) + dist.shape + (dim,)
