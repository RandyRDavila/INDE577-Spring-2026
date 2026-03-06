import math

import numpy as np
import pytest

from rice_ml.measures_ml.distances import euclidean, taxicab


def test_euclidean_basic():
    assert euclidean([0, 0], [3, 4]) == 5.0


def test_taxicab_basic():
    assert taxicab([0, 0], [3, 4]) == 7.0


def test_zero_distance():
    assert euclidean([1, 2, 3], [1, 2, 3]) == 0.0
    assert taxicab([1, 2, 3], [1, 2, 3]) == 0.0


def test_accepts_numpy_arrays():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 6.0, 3.0])
    assert math.isclose(euclidean(x, y), 5.0)
    assert math.isclose(taxicab(x, y), 7.0)


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        euclidean([1, 2], [1, 2, 3])

    with pytest.raises(ValueError, match="same length"):
        taxicab([1, 2], [1, 2, 3])


def test_empty_inputs_raise_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        euclidean([], [])

    with pytest.raises(ValueError, match="non-empty"):
        taxicab([], [])


def test_non_1d_inputs_raise_value_error():
    with pytest.raises(ValueError, match="one-dimensional"):
        euclidean([[1, 2], [3, 4]], [1, 2])

    with pytest.raises(ValueError, match="one-dimensional"):
        taxicab([1, 2], [[1, 2], [3, 4]])


def test_non_numeric_inputs_raise_type_error():
    with pytest.raises(TypeError, match="numeric"):
        euclidean(["a", "b"], [1, 2])

    with pytest.raises(TypeError, match="numeric"):
        taxicab([1, 2], ["x", "y"])


def test_non_finite_inputs_raise_value_error():
    with pytest.raises(ValueError, match="finite"):
        euclidean([1, np.nan], [1, 2])

    with pytest.raises(ValueError, match="finite"):
        taxicab([1, np.inf], [1, 2])
