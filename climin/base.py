# -*- coding: utf-8 -*-

import itertools

import numpy as np


def repeat_or_iter(obj):
    try:
        return iter(obj)
    except TypeError:
        return itertools.repeat(obj)


def dummylogfunc(*args, **kwargs): pass


class Minimizer(object):

    def __init__(self, wrt, args=None, logfunc=None):
        self.wrt = wrt
        self.logfunc = logfunc if logfunc is not None else dummylogfunc
        if args is None:
            self.args = itertools.repeat(([], {}))
        else:
            self.args = args

    def some(self, min_iter=None, max_iter=None, min_improv=None):
        """Minimize for some time.

        Minimization will be done for at least `min_iter` and at most `max_iter`
        steps. If in between, the improvement of the loss (which is a positive
        number if the loss gets lower) is less than min_improv, minimization is
        stopped."""
        loss_m1 = float('inf')
        info = None
        for i, info in enumerate(self):
            # This is an ugly hack, but it will go anyway.
            if min_improv is not None:
                loss = info['loss']
                improvement = loss_m1 - loss
            else:
                improvement = 1
            if improvement < min_improv and i > min_iter:
                break
            if i == max_iter:
                break
            loss_m1 = loss

        return info


def is_nonzerofinite(arr):
    """Return True if the array is neither zero, NaN or infinite."""
    return (arr != 0).any() and np.isfinite(arr).all()
