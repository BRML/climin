# -*- coding: utf-8 -*-

import itertools

from logging import deadend


def repeat_or_iter(obj):
    try:
        return iter(obj)
    except TypeError:
        return itertools.repeat(obj)


class Minimizer(object):

    def __init__(self, wrt, args=None, stop=1, logger=None):
        self.wrt = wrt
        self.stop = stop
        self.logger = logger if logger is not None else deadend()
        self.verbose = verbose
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
        info = {}
        for i, info in enumerate(self):
            loss = info['loss']
            improvement = loss_m1 - loss
            if improvement < min_improv and i > min_iter:
                break
            if i == max_iter:
                break
            loss_m1 = loss

        return info
