# -*- coding: utf-8 -*-


import scipy.optimize


class LineSearch(object):

    def __init__(self, wrt):
        self.wrt = wrt


class ExponentialDistant(LineSearch):

    def __init__(self, wrt, f, factor=0.8, granularity=50):
        super(ExponentialDistant, self).__init__(wrt)
        self.f = f
        self.factor = factor
        self.granularity = granularity

    def search(self, direction, args, kwargs):
        f_by_steplength = lambda steplength: self.f(
            self.wrt + steplength * direction, *args, **kwargs)
        distances = [self.factor**i for i in range(self.granularity)]

        loss0 = f_by_steplength(0)      # Loss at the current position.

        for d in distances:
            loss = f_by_steplength(d)
            cur_reduction = loss0 - loss
            if cur_reduction > 0:
                # We are better! So stop.
                return d
        return 0


class ScipyLineSearch(LineSearch):

    def __init__(self, wrt, f, fprime):
        super(ScipyLineSearch, self).__init__(wrt)
        self.f = f
        self.fprime = fprime

    def search(self, direction, args, kwargs):
        if kwargs:
            raise ValueError('keyword arguments not supported')
        gfk = self.fprime(self.wrt, *args)
        return scipy.optimize.line_search(
            self.f, self.fprime, self.wrt, direction, gfk, args=args)[0]
