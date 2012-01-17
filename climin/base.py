# -*- coding: utf-8 -*-

import itertools


def repeat_or_iter(obj):
    try:
        return iter(obj)
    except TypeError:
        return itertools.repeat(obj)


class Minimizer(object):

    def __init__(self, wrt, args=None, stop=1, verbose=False):
        self.wrt = wrt
        self.stop = stop
        self.verbose = verbose
        if args is None:
            self.args = itertools.repeat(([], {}))
        else:
            self.args = args

    def some(self, min_iter=None, max_iter=None, min_improv=None, log=None):
        if log is None:
            def log(*args, **kwargs): pass

        loss_m1 = float('inf')
        info = None

        for i, info in enumerate(self):
            log(info)
            loss = info['loss']
            improvement = loss_m1 - loss
            if improvement < min_improv and i > min_iter:
                break
            if i == max_iter:
                break
            loss_m1 = loss

        return info
