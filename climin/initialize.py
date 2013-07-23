# -*- coding: utf-8 -*-

import random

import numpy as np

import climin.mathadapt as ma


def sparsify_columns(arr, n_non_zero):
    """Set all but `n_non_zero` entries to zero for each column of `arr`."""
    colsize = arr.shape[0]

    # In case it's gnumpy, copy to numpy array first. The sparsifying loop will
    # run in numpy.
    arr_np = arr if isinstance(arr, np.ndarray) else arr.as_numpy_array()

    mask = np.ones_like(arr_np)
    for i in range(arr.shape[1]):
        idxs = xrange(colsize)
        zeros = random.sample(idxs, colsize - n_non_zero)
        mask[zeros, i] *= 0

    arr *= mask


def bound_spectral_radius(arr, bound=1.2):
    """Rescale the highest eigenvalue of the square matrix `arr` to `bound`."""
    vals, vecs = np.linalg.eig(ma.assert_numpy(arr))

    vals /= abs(vals).max()
    vals *= 1.2
    arr[...] = np.dot(vecs, np.dot(np.diag(vals), np.linalg.inv(vecs)))


def randomize_normal(arr, loc=0, scale=1):
    """Populate an array with random numbers from a normal distribution with
    mean `loc` and standard deviation `scale`."""
    sample = random.normal(arr.shape, loc, scale)
    arr[...] = sample
