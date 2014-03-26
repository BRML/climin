# -*- coding: utf-8 -*-

import itertools

import nose

from climin.stops import Patience


def test_patience_constant():
    func = iter([10, 10, 10, 10]).next
    stopper = Patience(func, 3, 2)

    assert not stopper({'n_iter': 0})
    assert not stopper({'n_iter': 1})
    assert not stopper({'n_iter': 2})
    assert stopper({'n_iter': 3}), 'initial patience should be over'


def test_patience_increase():
    func = iter([10, 10, 10, 5, 10, 10, 10]).next
    stopper = Patience(func, 3, 2)

    assert not stopper({'n_iter': 0})
    assert not stopper({'n_iter': 1})
    assert not stopper({'n_iter': 2})
    assert not stopper({'n_iter': 3})
    assert not stopper({'n_iter': 4})
    assert not stopper({'n_iter': 5})
    assert stopper({'n_iter': 6})
