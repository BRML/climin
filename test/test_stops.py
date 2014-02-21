# -*- coding: utf-8 -*-

import itertools

import nose

from climin.stops import patience, rising


def test_rising():
    func = iter([0, 0, 0.1, 1.2]).next
    is_rising = rising(func, 1, 0.1)
    assert not is_rising(None), 'returns True although not enough data'
    assert not is_rising(None), 'returns True although not rising'
    assert not is_rising(None), 'returns True although rising in tolerance'
    assert is_rising(None), 'returns False although rising'


def test_patience_constant():
    func = iter([10, 10, 10, 10]).next
    stopper = patience(func, 3, 2)

    assert not stopper({'n_iter': 0})
    assert not stopper({'n_iter': 1})
    assert not stopper({'n_iter': 2})
    assert stopper({'n_iter': 3}), 'initial patience should be over'


def test_patience_increase():
    func = iter([10, 10, 10, 5, 10, 10, 10]).next
    stopper = patience(func, 3, 2)

    assert not stopper({'n_iter': 0})
    assert not stopper({'n_iter': 1})
    assert not stopper({'n_iter': 2})
    assert not stopper({'n_iter': 3})
    assert not stopper({'n_iter': 4})
    assert not stopper({'n_iter': 5})
    assert stopper({'n_iter': 6})
