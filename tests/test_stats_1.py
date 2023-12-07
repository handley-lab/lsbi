import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from scipy.stats import kstest

from lsbi.stats_1 import multivariate_normal


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3), (2,), ()])
@pytest.mark.parametrize("dim", [1, 5])
class TestMultivariateNormal(object):
    cls = multivariate_normal

    def random(self, dim, shape):
        mean = np.random.randn(*shape, dim)
        cov = np.random.randn(*shape, dim, dim)
        cov = np.einsum("...ij,...kj->...ik", cov, cov) + dim * np.eye(dim)
        return self.cls(mean, cov)

    def test_properties(self, dim, shape):
        dist = self.random(dim, shape)
        assert dist.dim == dim
        assert dist.shape == shape

    @pytest.mark.parametrize("xshape", [(8, 7, 6), (8, 7), (8,), ()])
    def test_logpdf(self, dim, shape, xshape):
        dist = self.random(dim, shape)
        x = np.random.randn(*xshape, dim)
        assert dist.logpdf(x).shape == xshape + shape

    @pytest.mark.parametrize("size", [(8, 7, 6), (8, 7), (8,), 8])
    def test_rvs(self, dim, shape, size):
        dist = self.random(dim, shape)
        x = dist.rvs(size)
        assert x.shape == tuple(np.atleast_1d(size)) + shape + (dim,)
