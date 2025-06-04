"""Utility functions for lsbi."""

from typing import Union, Callable, Tuple, Any, Type

import numpy as np
from numpy.typing import NDArray

# Type aliases
ArrayLike = Union[float, int, NDArray[np.floating], list, tuple]


def logdet(A: ArrayLike, diagonal: bool = False) -> Union[float, NDArray[np.floating]]:
    """Compute log absolute determinant of matrix A.
    
    Parameters
    ----------
    A : array_like
        Input matrix or matrices. If diagonal=True, can be 1D array
        representing diagonal elements.
    diagonal : bool, optional
        If True, A represents diagonal elements only. Default is False.
        
    Returns
    -------
    float or ndarray
        Log absolute determinant of A.
        
    Examples
    --------
    >>> logdet(np.eye(2))  # Returns 0.0
    >>> logdet([2, 3], diagonal=True)  # Returns log(6)
    """
    if diagonal:
        return np.log(np.abs(A)).sum(axis=-1)
    else:
        return np.linalg.slogdet(A)[1]


def quantise(
    f: Callable[[ArrayLike], ArrayLike], 
    x: ArrayLike, 
    tol: float = 1e-8
) -> NDArray[np.floating]:
    """Quantise f(x) to zero within tolerance tol."""
    y = np.atleast_1d(f(x))
    return np.where(np.abs(y) < tol, 0, y)


def bisect(
    f: Callable[[ArrayLike], ArrayLike],
    a: ArrayLike,
    b: ArrayLike, 
    args: Tuple[Any, ...] = (),
    tol: float = 1e-8
) -> NDArray[np.floating]:
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


def dediagonalise(
    x: ArrayLike, 
    diagonal: bool, 
    *args: int
) -> Union[ArrayLike, NDArray[np.floating]]:
    """Convert diagonal representation to full matrix if needed.
    
    Parameters
    ----------
    x : array_like
        Input array. If diagonal=True, represents diagonal elements.
    diagonal : bool
        If True, construct full matrix with x on diagonal.
        If False, return x unchanged.
    *args : int
        Dimensions for the identity matrix when diagonal=True.
        
    Returns
    -------
    array_like or ndarray
        Full matrix if diagonal=True, otherwise x unchanged.
        
    Examples
    --------
    >>> dediagonalise([1, 2], True, 2)  # Returns 2x2 matrix with [1,2] on diagonal
    >>> dediagonalise(np.eye(2), False)  # Returns input unchanged
    """
    if diagonal:
        return np.atleast_1d(x)[..., None, :] * np.eye(*args)
    else:
        return x


def alias(cls: Type[Any], name: str, alias: str) -> None:
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
