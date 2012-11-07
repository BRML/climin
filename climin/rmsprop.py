# -*- coding: utf-8 -*-


import numpy as np

from base import Minimizer


class RmsProp(Minimizer):

    def __init__(self, wrt, fprime, steprate, decay,
                 args=None, logfunc=None):
        super(RmsProp, self).__init__(wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprate = steprate
        self.decay = decay

    def __iter__(self):
        moving_mean_squared = None

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)
            if moving_mean_squared is None:
                moving_mean_squared = gradient**2
            moving_mean_squared = (
                self.decay * moving_mean_squared
                + (1 - self.decay) * gradient**2)
            step = self.steprate * gradient / np.sqrt(moving_mean_squared + 1e-8)
            self.wrt -= step

            yield dict(args=args, kwargs=kwargs, gradient=gradient,
                       n_iter=i,
                       moving_mean_squared=moving_mean_squared,
                       step=step)
