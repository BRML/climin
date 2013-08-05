# -*- coding: utf-8 -*-

"""Module that contains functionality to initialize parameters to starting
values."""

import random

import numpy as np

import climin.mathadapt as ma


def sparsify_columns(arr, n_non_zero):
    """Set all but ``n_non_zero`` entries to zero for each column of ``arr``.

    This is a common technique to find better starting points for learning
    deep and/or recurrent networks.

    Parameters
    ----------

    arr : array_like, two dimensional
      Array to work upon in place.

    n_non_zero : integer
      Amount of non zero entries to keep.

    Examples
    --------

    >>> import numpy as np
    >>> from climin.initialize import sparsify_columns
    >>> arr = np.arange(9).reshape((3, 3))
    >>> sparsify_columns(arr, 1)
    >>> arr                                         # doctest: +SKIP
    array([[0, 0, 0],
           [0, 4, 5],
           [6, 0, 0]])
    """
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
    """Set the spectral radius of the square matrix ``arr`` to ``bound``.

    This is performed by making an Eigendecomposition of ``arr``, rescale all
    Eigenvalues such that the absolute value of the greatest matches ``bound``
    and recompose it again.

    Parameters
    ----------

    arr : array_like, two dimensional
        Array to work upon in place.

    bound : float, optional, default: 1.2

    Examples
    --------

    >>> import numpy as np
    >>> from climin.initialize import bound_spectral_radius
    >>> arr = np.arange(9).reshape((3, 3)).astype('float64')
    >>> bound_spectral_radius(arr, 1.1)
    >>> arr                                 # doctest: +SKIP
    array([[ -7.86816957e-17,   8.98979486e-02,   1.79795897e-01],
           [  2.69693846e-01,   3.59591794e-01,   4.49489743e-01],
           [  5.39387691e-01,   6.29285640e-01,   7.19183588e-01]])
    """
    vals, vecs = np.linalg.eig(ma.assert_numpy(arr))

    vals /= abs(vals).max()
    vals *= 1.2
    arr[...] = np.dot(vecs, np.dot(np.diag(vals), np.linalg.inv(vecs)))


def randomize_normal(arr, loc=0, scale=1):
    """Populate an array with random numbers from a normal distribution with
    mean `loc` and standard deviation `scale`.

    Parameters
    ----------

    arr : array_like
      Array to work upon in place.

    loc : float
      Mean of the random numbers.

    scale : float
      Standard deviation of the random numbers.

    Examples
    --------

    >>> import numpy as np
    >>> from climin.initialize import randomize_normal
    >>> arr = np.empty((3, 3))
    >>> randomize_normal(arr)
    >>> arr                                 # doctest: +SKIP
    array([[ 0.18076413,  0.60880657,  1.20855691],
           [ 1.7799948 , -0.82565481,  0.53875307],
           [-0.67056028, -1.46257419,  1.17033425]])
    >>> randomize_normal(arr, 10, 0.1)
    >>> arr                                 # doctest: +SKIP
    array([[ 10.02221481,  10.0982449 ,  10.02495358],
          [  9.99867829,   9.99410111,   9.8242318 ],
          [  9.9383779 ,   9.94880091,  10.03179085]])
    """
    sample = np.random.normal(loc, scale, arr.shape)
    if isinstance(arr, np.ndarray):
        arr[...] = sample.astype(arr.dtype)
    else:
        # Assume gnumpy.
        arr[:] = sample.astype('float32')
