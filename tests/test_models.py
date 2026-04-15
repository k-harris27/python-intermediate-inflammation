"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation import models

@pytest.mark.parametrize(
        "data, expected",
        [
            ([[0,0], [0,0], [0,0]], [0,0]),
            ([[1,2], [3,4], [5,6]], [3,4]),
            ([[0.5,1.5], [2.5,3.5], [4.5,5.5]], [2.5,3.5])
        ]
)
def test_daily_mean(data, expected):
    """Test that mean function works for a variety of inputs"""
    npt.assert_array_almost_equal(models.daily_mean(data), expected)

@pytest.mark.parametrize(
        "data, expected",
        [
            ([[0,0], [0,0], [0,0]], [0,0]),
            ([[1,6], [3,4], [5,2]], [1,2]),
            ([[0.5,1.5], [2.5,3.5], [4.5,5.5]], [0.5,1.5])
        ]
)
def test_daily_min(data, expected):
    """Test that min function works for an array of positive integers."""

    npt.assert_array_almost_equal(models.daily_min(data), expected)

@pytest.mark.parametrize(
        "data, expected",
        [
            ([[0,0], [0,0], [0,0]], [0,0]),
            ([[1,6], [3,4], [5,2]], [5,6]),
            ([[0.5,1.5], [2.5,3.5], [4.5,5.5]], [4.5,5.5])
        ]
)
def test_daily_max(data, expected):
    """Test that min function works for an array of positive integers."""

    npt.assert_array_almost_equal(models.daily_max(data), expected)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[-1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
    ])
def test_patient_normalise(test, expected):
    """Test normalisation for parameterised arrays of integers.

    Args:
        test (np.ndarray): Test data as a 2D numpy array.
        expected (np.ndarray): Expected normalised result as a 2D numpy array.
    """

    result = models.patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
