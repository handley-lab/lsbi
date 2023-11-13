"""Utility functions for lsbi."""
import numpy as np


def logdet(A):
    """log(abs(det(A)))."""
    return np.linalg.slogdet(A)[1]


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
    while np.abs(a-b).max() > tol:
        fa = f(a)
        fb = f(b)
        q = (a+b)/2
        fq = f(q)
        a = np.where(np.sign(fq) == np.sign(fa), q, a)
        b = np.where(np.sign(fq) == np.sign(fb), q, b)
    return (a+b)/2
