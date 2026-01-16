"""This module provides math adaptations for array operations.

These functions provide a consistent interface for common array operations.
"""

import numpy as np


def sqrt(x):
    """Return an array of the same shape containing the element square
    root of `x`."""
    return x**0.5


def zero_like(x):
    """Return an array of the same shape as `x` containing only zeros."""
    return x * 0.0


def ones_like(x):
    """Return an array of the same shape as `x` containing only ones."""
    return x * 0.0 + 1.0


def clip(a, a_min, a_max):
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to the interval
    edges. For example, if an interval of [0, 1] is specified, values smaller
    than 0 become 0, and values larger than 1 become 1."""
    return np.clip(a, a_min, a_max)


def sign(x):
    """Returns an element-wise indication of the sign of a number."""
    return np.sign(x)


def where(x, *args):
    """Delegate to numpy.where."""
    return np.where(x, *args)


def random_like(x):
    """Return an array of the same shape as `x` filled with random numbers from
    the interval [0, 1)."""
    return np.random.random(x.shape)


def random_normal_like(x, loc, scale):
    """Return an array of the same shape as `x` filled with random numbers from
    a normal distribution."""
    return np.random.normal(loc, scale, x.shape)


def assert_numpy(x):
    """Given a numpy array x, return a copy of the array."""
    return x.copy()


def scalar(x):
    if isinstance(x, float):
        return x
    if not x.size == 1:
        raise ValueError("size is %i instead of 1" % x.size)
    return x.reshape((1,))[0]


def isnan(x):
    """Delegate to numpy.isnan."""
    return np.isnan(x)
