"""Utility functions for lsbi."""
import numpy as np


def logdet(A, diag=False):
    """log(abs(det(A)))."""
    if diag:
        return np.sum(np.log(np.abs(A)), axis=-1)
    else:
        return np.linalg.slogdet(A)[1]


def quantise(f, x, tol=1e-8):
    """Quantise f(x) to zero within tolerance tol."""
    y = np.atleast_1d(f(x))
    return np.where(np.abs(y) < tol, 0, y)


def matrix(M, *args):
    """Convert M to a matrix."""
    if len(np.shape(M)) > 1:
        return M
    else:
        return M * np.eye(*args)


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


def choice(size, p):
    """Vectorised choice function.

    Parameters
    ----------
    size : int or tuple of ints
        Shape of the output.
    p : array_like
        Probability array

    Returns
    -------
    out : ndarray
        Output array of shape (*size, *p.shape[:-1]).
    """
    cump = np.cumsum(p, axis=-1)
    u = np.random.rand(*size, *p.shape)
    return np.argmin(u > cump, axis=-1)
