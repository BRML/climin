"""This module provides functionality which is usable for
coding towards gnumpy and numpy: the idea is to avoid if clauses
in the optimizer code.

"""

import numpy as np

try:
    import gnumpy as gp
except ImportError:
    pass


def sqrt(x):
    """Return an array of the same shape containing the element square
    root of `x`."""
    return x ** 0.5


def zero_like(x):
    """Return an array of the same shape as `x` containing only zeros."""
    return x * 0.


def ones_like(x):
    """Return an array of the same shape as `x` containing only ones."""
    return x * 0. + 1.


def clip(a, a_min, a_max):
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to the interval
    edges. For example, if an interval of [0, 1] is specified, values smaller
    than 0 become 0, and values larger than 1 become 1."""
    if not isinstance(a, np.ndarray):
        max_mask = (a > a_max)
        max_tar = gp.ones(a.shape) * a_max
        min_mask = (a < a_min)
        min_tar = gp.ones(a.shape) * a_min
        a_clipped = (
            a * (1 - max_mask - min_mask)
            + max_tar * max_mask + min_tar * min_mask)
        return a_clipped
    else:
        return np.clip(a, a_min, a_max)


def sign(x):
    """Returns an element-wise indication of the sign of a number."""
    if not isinstance(x, np.ndarray):
        return gp.sign(x)
    else:
        return np.sign(x)


def random_like(x):
    """Return an array of the same shape as `x` filled with random numbers from
    the interval [0, 1)."""
    if not isinstance(x, np.ndarray):
        return gp.rand(x.shape)
    else:
        return np.random.random(x.shape)
