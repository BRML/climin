# -*- coding: utf-8 -*-

"""Module that contains projection operators."""


import numpy as np

from mathadapt import sqrt


def project_to_simplex(v, scale=1.):
    """ Project a vector into the probability simplex, return the result.

    The result is the closest non-negative vector whose elements sum to scale.

    Parameters
    ----------
    v : np.array((n))
        The vector in R^n to project onto probability simplex

    scale : the sum of the elements in the result will be scale.

    If a different sum is desired for the entries, set scale accordingly.

    The orthogonal projection of v into the simplex is of form

       non_negative_part(v-adjustment)

    for some a (see)

    The function adjustment->sum(non_negative_part(v-adjustment)) is decreasing and convex.
    Then we can use a newton's iteration, and piecewise linearity gives
    finite convergence.
    This is faster than the O(n log(1/epsilon)) using binary search,
    and there exist more complicated O(n) algorithms.

    """
    adjustment = np.min(v) - scale
    sum_deviation = np.sum(non_negative_part(v - adjustment)) - scale

    while sum_deviation > 1e-8:
        diff = v - adjustment
        sum_deviation = non_negative_part(diff).sum() - scale
        df = (1.0 * (diff > 0)).sum()
        #print((a, f, df))
        adjustment += sum_deviation / (df + 1e-6)

    return non_negative_part(v - adjustment)


def non_negative_part(v):
    """ A copy of v with negative entries replaced by zero. """
    return (v + np.abs(v)) / 2


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
