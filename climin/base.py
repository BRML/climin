# -*- coding: utf-8 -*-

import itertools
import collections
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

    def minimize_until(self, criterions):
        """Minimize until one of the supplied `criterions` is met.

        Each criterion is a callable that, given the info object yielded by
        an optimizer, returns a boolean indicating whether to stop. False means
        to continue, True means to stop."""
        if not criterions:
            raise ValueError('need to supply at least one criterion')
        
        # if criterions is a single criterion, wrap it in iterable list
        if not isinstance(criterions, collections.Iterable):
            criterions = [criterions]
        
        for info in self:
            for criterion in criterions:
                if criterion(info):
                    break
        return info


def is_nonzerofinite(arr):
    """Return True if the array is neither zero, NaN or infinite."""
    return (arr != 0).any() and np.isfinite(arr).all()
