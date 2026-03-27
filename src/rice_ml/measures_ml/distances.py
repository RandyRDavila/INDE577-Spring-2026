"""
Distance measures for ``rice_ml``.

This module provides basic vector distance functions with input validation
and doctestable NumPy-style docstrings.

Examples
--------
>>> euclidean([0, 0], [3, 4])
5.0
>>> taxicab([0, 0], [3, 4])
7.0
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    'euclidean',
    'taxicab',
]

def _validate_1d_numeric_pair(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and coerce two inputs into compatible 1D numeric NumPy arrays.

    Parameters
    ----------
    x : Sequence[float]
        First input vector.
    y : Sequence[float]
        Second input vector.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A pair of validated 1D float arrays.

    Raises
    ------
    TypeError
        If either input is not array-like or contains non-numeric data.
    ValueError
        If either input is empty, not one-dimensional, or the vectors do not
        have the same length.

    Examples
    --------
    >>> a, b = _validate_1d_numeric_pair([1, 2], [3, 4])
    >>> a.tolist(), b.tolist()
    ([1.0, 2.0], [3.0, 4.0])
    """
    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Inputs must be array-like and contain only numeric values.") from exc

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Inputs must be one-dimensional vectors.")

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("Inputs must be non-empty vectors.")

    if x_arr.shape != y_arr.shape:
        raise ValueError("Input vectors must have the same length.")

    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Inputs must contain only finite numeric values.")

    return x_arr, y_arr


def euclidean(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute the Euclidean distance between two vectors.

    The Euclidean distance between vectors :math:`x` and :math:`y` is

    .. math::
       \\sqrt{\\sum_i (x_i - y_i)^2}

    Parameters
    ----------
    x : Sequence[float]
        First input vector.
    y : Sequence[float]
        Second input vector.

    Returns
    -------
    float
        Euclidean distance between ``x`` and ``y``.

    Raises
    ------
    TypeError
        If either input is not array-like or contains non-numeric data.
    ValueError
        If either input is empty, not one-dimensional, contains non-finite
        values, or if the vectors have different lengths.

    Examples
    --------
    >>> euclidean([0, 0], [3, 4])
    5.0
    >>> euclidean((1, 2, 3), (1, 2, 3))
    0.0
    >>> round(euclidean([1, 1], [4, 5]), 6)
    5.0
    """
    x_arr, y_arr = _validate_1d_numeric_pair(x, y)
    return float(np.linalg.norm(x_arr - y_arr))


def taxicab(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute the taxicab (Manhattan) distance between two vectors.

    The taxicab distance between vectors :math:`x` and :math:`y` is

    .. math::
       \\sum_i |x_i - y_i|

    Parameters
    ----------
    x : Sequence[float]
        First input vector.
    y : Sequence[float]
        Second input vector.

    Returns
    -------
    float
        Taxicab distance between ``x`` and ``y``.

    Raises
    ------
    TypeError
        If either input is not array-like or contains non-numeric data.
    ValueError
        If either input is empty, not one-dimensional, contains non-finite
        values, or if the vectors have different lengths.

    Examples
    --------
    >>> taxicab([0, 0], [3, 4])
    7.0
    >>> taxicab((1, 2, 3), (1, 2, 3))
    0.0
    >>> taxicab([1, -1], [4, 5])
    9.0
    """
    x_arr, y_arr = _validate_1d_numeric_pair(x, y)
    return float(np.sum(np.abs(x_arr - y_arr)))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
