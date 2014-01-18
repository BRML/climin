# -*- coding: utf-8 -*-

"""Module that contains projection operators."""


import numpy as np

from mathadapt import sqrt


def project_to_simplex(v, scale=1.):
    """ Project v into the probability simplex, return the result.
    If a different sum is desired for the entries, use scale.

    The orthogonal projection of v into the simplex is of form

       non_negative_part(v-a)

    for some a.

    The function a->sum(non_negative_part(v-a)) is decreasing and convex.
    Then we can use a newton's iteration, and piecewise linearity give
    finite convergence.
    This is faster than the O(n log(n)) using binary search,
    and there exist more complicated O(n) algorithms.

    """
    a = min(v) - scale
    f = sum(non_negative_part(v - a)) - scale

    while f > 1e-8:
        diff = v - a
        f = non_negative_part(diff).sum() - scale
        df = (1.0 * (diff > 0)).sum()
        #print((a, f, df))
        a += f / (df + 1e-6)

    return non_negative_part(v - a)


def non_negative_part(v):
    """ A copy of v with negative entries replaced by zero. """
    return (v + abs(v)) / 2


def max_length_columns(arr, max_length):
    """Project the columns of an array below a certain length.

    Works in place.

    Parameters
    ----------

    arr : array_like
        2D array.

    max_length : int
        Maximum length of a column.
    """
    if arr.ndim != 2:
        raise ValueError('only 2d arrays allowed')

    max_length = float(max_length)

    lengths = sqrt((arr ** 2).sum(axis=0))
    too_big_by = lengths / max_length
    divisor = too_big_by
    non_violated = lengths < max_length

    if isinstance(arr, np.ndarray):
        divisor[np.where(non_violated)] = 1.
    else:
        # Gnumpy implementation.
        # TODO: can this be done more efficiently?
        for i, nv in enumerate(non_violated):
            if nv:
                divisor[i] = 1.

    arr /= divisor[np.newaxis]
