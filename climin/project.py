# -*- coding: utf-8 -*-

"""Module that contains projection operators."""


import numpy as np

from mathadapt import sqrt


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
