"""This module provides functionality which is usable for
coding towards gnumpy and numpy: the idea is to avoid if clauses
in the optimizer code.

"""


def sqrt(x):
    """Return an array of the same shape containing the element square
    root of `x`."""
    return x**(0.5)


def zero_like(x):
    """Return an array of the same shape as `x` containing only zeros."""
    return x * 0.


def ones_like(x):
    """Return an array of the same shape as `x` containing only ones."""
    return x * 0. + 1.
