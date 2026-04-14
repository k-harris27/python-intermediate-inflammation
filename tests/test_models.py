"""Tests for statistics functions within the Model layer."""

import pytest
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
