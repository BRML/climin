# -*- coding: utf-8 -*-

import random


def sparsify_columns(arr, n_non_zero):
    """Set all but `n_non_zero` entries to zero for each column of `arr`."""
    colsize = arr.shape[0]
    for i in range(arr.shape[1]):
        idxs = xrange(colsize)
        zeros = random.sample(idxs, colsize - n_non_zero)
        print zeros
        arr[zeros, i] *= 0
