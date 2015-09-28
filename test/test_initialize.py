# -*- coding: utf-8 -*-

import numpy as np

from climin.initialize import sparsify_columns


def test_sparsify_columns():
    pars = np.ones((8, 10))
    sparsify_columns(pars, 3)
    assert (pars.sum(axis=0) == [3] * 10).all(), 'sparsify_columns did not work'
