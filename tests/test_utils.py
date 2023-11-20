import pytest
from numpy.testing import assert_allclose

from lsbi.utils import bisect


def test_bisect():
    def f(x):
        return x - 5

    assert bisect(f, 0, 10) == 5

    with pytest.raises(ValueError):
        bisect(f, 0, 4)

    def f(x):
        return x - [1, 2]

    assert_allclose(bisect(f, 0, 10), [1, 2])
