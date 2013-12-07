# -*- coding: utf-8 -*-

import climin.project as pr
from numpy import array, allclose

def test_simplex_projection():
    point = array([1., 2, 3])
    scale = 1.43
    distribution = pr.project_to_simplex(point, scale=scale)
    assert allclose(distribution.sum(), scale, atol = 1e-13), 'projection failed to achieve requested sum of elements'
    assert (distribution >= 0.).all(), 'projection produced negative elements'
