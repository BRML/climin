# -*- coding: utf-8 -*-

import itertools

import nose

from climin.stops import rising


def test_rising():
    func = iter([0, 0, 0.1, 1.2]).next
    is_rising = rising(func, 1, 0.1)
    assert is_rising(None) == False, 'returns True although not enough data'
    assert is_rising(None) == False, 'returns True although not rising'
    assert is_rising(None) == False, 'returns True although rising in tolerance'
    assert is_rising(None) == True, 'returns False although rising'
