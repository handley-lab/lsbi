"""Utility functions for lsbi."""

import numpy as np


def logdet(A, diagonal=False):
    """log(abs(det(A)))."""
    if diagonal:
        return np.log(np.abs(A)).sum(axis=-1)
    else:
        return np.linalg.slogdet(A)[1]


def quantise(f, x, tol=1e-8):
    """Quantise f(x) to zero within tolerance tol."""
    y = np.atleast_1d(f(x))
    return np.where(np.abs(y) < tol, 0, y)


def bisect(f, a, b, args=(), tol=1e-8):
    """Vectorised simple bisection search.

    The shape of the output is the broadcasted shape of a and b.

    Parameters
    ----------
    f : callable
        Function to find the root of.
    a : array_like
        Lower bound of the search interval.
    b : array_like
        Upper bound of the search interval.
    args : tuple, optional
        Extra arguments to `f`.
    tol : float, optional
        (absolute) tolerance of the solution

    Returns
    -------
    x : ndarray
        Solution to the equation f(x) = 0.
    """
    a = np.array(a)
    b = np.array(b)
    while np.abs(a - b).max() > tol:
        fa = quantise(f, a, tol)
        fb = quantise(f, b, tol)
        a = np.where(fb == 0, b, a)
        b = np.where(fa == 0, a, b)

        if np.any(fa * fb > 0):
            raise ValueError("f(a) and f(b) must have opposite signs")
        q = (a + b) / 2
        fq = quantise(f, q, tol)

        a = np.where(fq == 0, q, a)
        a = np.where(fa * fq > 0, q, a)

        b = np.where(fq == 0, q, b)
        b = np.where(fb * fq > 0, q, b)
    return (a + b) / 2


def dediagonalise(x, diagonal, *args):
    """Optionally construct a dense matrix with x on the diagonal."""
    if diagonal:
        return np.atleast_1d(x)[..., None, :] * np.eye(*args)
    else:
        return x


def alias(cls, name, alias):
    """Create an alias for a property.

    Parameters
    ----------
    cls : class
        Class to add the alias to.
    name : str
        Name of the property to alias.
    alias : str
        Name of the alias.

    Examples
    --------
    >>> class MyCls:
    ...     def __init__(self, name):
    ...         self.name = name
    ...
    >>> alias(MyCls, 'name', 'n')
    >>> obj = MyCls('will')
    >>> obj.name
    'will'
    >>> obj.n
    'will'
    >>> obj.n = 'bill'
    >>> obj.name
    'bill'
    """

    @property
    def f(self):
        return getattr(self, name)

    @f.setter
    def f(self, x):
        setattr(self, name, x)

    setattr(cls, alias, f)
