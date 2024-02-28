import numpy as np
import pytest
from numpy.testing import assert_allclose

from lsbi.utils import alias, bisect


def test_bisect():
    def f(x):
        return x - 5

    assert bisect(f, 0, 10) == 5

    with pytest.raises(ValueError):
        bisect(f, 0, 4)

    def f(x):
        return x - [1, 2]

    assert_allclose(bisect(f, 0, 10), [1, 2])


def test_alias():
    class A:
        pass

    a = A()
    with pytest.raises(AttributeError):
        a.x
    a.x = 1
    assert a.x == 1
    with pytest.raises(AttributeError):
        a.y
    alias(A, "x", "y")
    assert a.y == 1
    a.y = 2
    assert a.x == 2
    a.x = np.eye(3)
    assert_allclose(a.y, np.eye(3))
    a.y[0, 0] = 0
    assert a.x[0, 0] == 0
