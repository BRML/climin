# -*- coding: utf-8 -*-

import random

import numpy as np


def sparsify_columns(arr, n_non_zero):
    """Set all but `n_non_zero` entries to zero for each column of `arr`."""
    colsize = arr.shape[0]
    for i in range(arr.shape[1]):
        idxs = xrange(colsize)
        zeros = random.sample(idxs, colsize - n_non_zero)
        arr[zeros, i] *= 0


def bound_spectral_radius(arr, bound=1.2):
    """Rescale the highest eigenvalue of the square matrix `arr` to `bound`."""
    vals, vecs = np.linalg.eig(arr)

    vals /= abs(vals).max()
    vals *= 1.2
    arr[...] = np.dot(vecs, np.dot(np.diag(vals), np.linalg.inv(vecs)))
